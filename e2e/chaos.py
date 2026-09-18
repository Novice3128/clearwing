#!/usr/bin/env python3
"""cw-e2e chaos proxy — parameterized byte-pipe reverse proxy + synthetic
HTTP-status fault injector.

Modes (first N client connections are hit, the rest healthy relay):
  refuse    : accept then immediately close          (close-before-response)
  reset     : read 1KB then RST via SO_LINGER 0       (mid-request reset)
  stall     : read request head, hold 75s, relay      (read-timeout path)
  ratelimit : answer HTTP 429 + Retry-After: 1        (synthetic status fault)

The upstream host/port/path are resolved DYNAMICALLY from the live config's
provider.base_url (never hardcode the vendor — SPEC §7). Host header is
rewritten to the upstream host (vendor edge rejects mismatched Host with 421).
Key discipline: request lines only are logged; bodies never parsed or stored.
"""
from __future__ import annotations

import asyncio
import re
import socket
import ssl
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import yaml

LISTEN_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


def resolve_direct_endpoint(url: str | None = None) -> dict:
    """Parse an explicit base_url (preferred: pre-surgery snapshot) or the
    live config provider.base_url into upstream parts. NEVER call this against
    the live config while a config surgery is ACTIVE — pass the surgery's
    original_url instead (review P0-3: the proxy would otherwise relay to
    itself)."""
    if not url:
        cfg = yaml.safe_load((Path.home() / ".clearwing/config.yaml").read_text())
        url = cfg["provider"]["base_url"]
    u = urlsplit(url)
    return {"url": url, "host": u.hostname, "port": u.port or (443 if u.scheme == "https" else 80),
            "path": u.path.rstrip("/"), "scheme": u.scheme}


class ProxyHandle:
    """Runs the proxy on a dedicated thread+event loop; stop() closes the
    server so sequential scenarios can reuse the same port."""

    def __init__(self, mode: str, limit: int, log_path: Path, direct: dict, port: int):
        self.mode, self.limit, self.log_path = mode, limit, log_path
        self.direct, self.port = direct, port
        self.loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.base_events.Server | None = None
        self._hits = 0
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError(f"chaos proxy failed to listen on {port}")

    # ---------------------------------------------------------------- thread

    def _log(self, msg: str) -> None:
        line = f"{time.strftime('%T')} {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", buffering=1) as fh:
            fh.write(line + "\n")

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except BaseException as e:  # noqa: BLE001 — CancelledError is BaseException
            self._log(f"# proxy exit: {type(e).__name__}")
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(self.loop) if not t.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self.loop.close()

    async def _serve(self) -> None:
        self._server = await asyncio.start_server(self._handle, LISTEN_HOST, self.port)
        self._log(f"chaos proxy listening {LISTEN_HOST}:{self.port} mode={self.mode} "
                  f"first {self.limit} -> {self.direct['host']}:{self.direct['port']}{self.direct['path']}")
        self._ready.set()
        async with self._server:
            await self._server.serve_forever()

    # ---------------------------------------------------------------- logic

    async def _relay(self, reader, writer) -> None:
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _handle(self, cr, cw) -> None:
        n = self._hits = self._hits + 1
        hit = n <= self.limit
        head = b""
        try:
            head = await cr.read(1024)
            reqline = head.split(b"\r\n", 1)[0][:80].decode("latin1", "replace")
        except Exception:
            reqline = "<no request>"
        if hit and self.mode == "refuse":
            self._log(f"#{n} CHAOS refuse: closing ({reqline})")
            cw.close()
            return
        if hit and self.mode == "reset":
            self._log(f"#{n} CHAOS reset: RST after 1KB ({reqline})")
            sock = cw.get_extra_info("socket")
            if sock is not None:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            cw.close()
            return
        if hit and self.mode == "ratelimit":
            self._log(f"#{n} CHAOS 429: synthetic rate-limit ({reqline})")
            body = b"{\"error\":\"rate_limit_exceeded\"}"
            cw.write(b"HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\n"
                     b"content-type: application/json\r\ncontent-length: " + str(len(body)).encode() +
                     b"\r\nconnection: close\r\n\r\n" + body)
            await cw.drain()
            cw.close()
            return
        if hit and self.mode == "stall":
            self._log(f"#{n} CHAOS stall: holding 75s ({reqline})")
            await asyncio.sleep(75)
        head = re.sub(rb"(?im)^Host:.*\r\n",
                      f"Host: {self.direct['host']}\r\n".encode(), head, count=1)
        try:
            ssl_ctx = ssl.create_default_context() if self.direct.get("scheme", "https") == "https" else None
            ur, uw = await asyncio.open_connection(self.direct["host"], self.direct["port"], ssl=ssl_ctx)
        except Exception as e:
            self._log(f"#{n} upstream connect failed: {type(e).__name__}")
            cw.close()
            return
        self._log(f"#{n} relay{' (post-chaos)' if not hit else ''}: {reqline}")
        if head:
            uw.write(head)
            await uw.drain()
        await asyncio.gather(self._relay(cr, uw), self._relay(ur, cw))

    # ----------------------------------------------------------------- ctrl

    def refusals(self) -> int:
        return self._hits if self.limit >= 10**6 else min(self._hits, self.limit)

    def stop(self) -> None:
        if self.loop and self._server:
            async def _close():
                self._server.close()
                await self._server.wait_closed()
                raise asyncio.CancelledError()
            try:
                asyncio.run_coroutine_threadsafe(_close(), self.loop).result(timeout=5)
            except BaseException:
                pass
        self._thread.join(timeout=6)
