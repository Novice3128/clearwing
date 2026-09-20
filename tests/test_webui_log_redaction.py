"""Regression tests for issue #84: the operator api_key must never reach the
webui log file.

Two bugs in the old ``_ApiKeyRedactionFilter``:

1. It was only installed on ``uvicorn.access`` — but the WebSocket
   accept/reject lines (``... "WebSocket /ws/agent?api_key=..." [accepted]``)
   are emitted on ``uvicorn.error`` (all three uvicorn WS protocol impls
   route their logger through it), so the key leaked in the clear.
2. The rewrite check ran on the ALREADY-redacted message (``api_key=``
   is a substring of ``api_key=[REDACTED]``), so every matching record was
   collapsed to a pre-rendered msg with ``args=None`` — and uvicorn's
   ``AccessFormatter.formatMessage`` unpacks ``record.args`` into five
   values, turning each access line into a "--- Logging error ---"
   traceback instead of a log line.
"""

from __future__ import annotations

import logging

import pytest
from uvicorn.logging import AccessFormatter

from clearwing.ui.commands.webui import _ApiKeyRedactionFilter, _install_api_key_redaction

# Faithful uvicorn 0.44 shapes: the access line carries a 5-tuple
# (client, method, full_path, http_version, status) and the WS accept line a
# 2-tuple (client, full path+query) — the query, secret included, rides in
# the full_path arg in both.
ACCESS_MSG = '%s - "%s %s HTTP/%s" %d'
ACCESS_ARGS = ("127.0.0.1:1", "GET", "/api/x?api_key=SECRET", "1.1", 200)
WS_MSG = '%s - "WebSocket %s" [accepted]'
WS_ARGS = ("127.0.0.1:1", "/ws/agent?api_key=SECRET")


def _record(name: str, msg: str, args: object) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestRedactionFilterShapes:
    def test_access_record_redacts_arg_and_keeps_tuple(self):
        record = _record("uvicorn.access", ACCESS_MSG, ACCESS_ARGS)
        assert _ApiKeyRedactionFilter().filter(record) is True
        assert record.args is not None
        assert isinstance(record.args, tuple)
        assert record.args[2] == "/api/x?api_key=[REDACTED]"
        assert "SECRET" not in record.getMessage()

    def test_access_record_formats_through_accessformatter(self):
        """The old args=None collapse made AccessFormatter raise — the exact
        "--- Logging error ---" traceback observed in the log file."""
        record = _record("uvicorn.access", ACCESS_MSG, ACCESS_ARGS)
        _ApiKeyRedactionFilter().filter(record)
        formatted = AccessFormatter("%(client_addr)s - \"%(request_line)s\" %(status_code)s").format(
            record
        )
        assert "SECRET" not in formatted
        assert "api_key=[REDACTED]" in formatted

    def test_websocket_accepted_record_redacts_path_arg(self):
        record = _record("uvicorn.error", WS_MSG, WS_ARGS)
        assert _ApiKeyRedactionFilter().filter(record) is True
        assert record.args is not None
        assert record.args[1] == "/ws/agent?api_key=[REDACTED]"
        formatted = logging.Formatter("%(message)s").format(record)
        assert "SECRET" not in formatted
        assert 'WebSocket /ws/agent?api_key=[REDACTED]" [accepted]' in formatted

    def test_argless_record_rewrites_msg(self):
        record = _record(
            "uvicorn.error", "connect failed: url=/x?api_key=SECRET", None
        )
        assert _ApiKeyRedactionFilter().filter(record) is True
        assert record.args is None
        assert "api_key=[REDACTED]" in str(record.msg)
        assert "SECRET" not in str(record.msg)

    def test_leak_in_msg_template_with_args_collapses(self):
        # The template itself carries the secret: arg redaction alone cannot
        # clean it, so the record must collapse to a redacted msg + args=None.
        record = _record("uvicorn.error", "GET /y?api_key=SECRET %s", ("extra",))
        assert _ApiKeyRedactionFilter().filter(record) is True
        assert record.args is None
        assert "SECRET" not in str(record.msg)
        assert "api_key=[REDACTED]" in str(record.msg)

    def test_record_without_api_key_passes_untouched(self):
        record = _record("uvicorn.access", ACCESS_MSG, ("127.0.0.1:1", "GET", "/api/x", "1.1", 200))
        snapshot = (record.msg, record.args)
        assert _ApiKeyRedactionFilter().filter(record) is True
        assert (record.msg, record.args) == snapshot

    def test_non_string_args_preserved(self):
        record = _record("uvicorn.access", ACCESS_MSG, ("127.0.0.1:1", "GET", "/x?api_key=S", "1.1", 200))
        _ApiKeyRedactionFilter().filter(record)
        assert record.args[0] == "127.0.0.1:1"
        assert record.args[4] == 200

    def test_percent_encoded_param_name_redacts(self):
        """``?api%5Fkey=`` passes auth (starlette percent-decodes param names)
        and must be redacted in its wire form — a literal ``api_key=`` regex
        would leave this spelling in the clear."""
        record = _record("uvicorn.error", WS_MSG, ("127.0.0.1:1", "/ws/agent?api%5Fkey=SECRET"))
        assert _ApiKeyRedactionFilter().filter(record) is True
        assert record.args[1] == "/ws/agent?api%5Fkey=[REDACTED]"
        assert "SECRET" not in record.getMessage()

    def test_any_encoded_spelling_of_api_key_redacts(self):
        """Codex r2 P1: starlette decodes param NAMES, so EVERY encoding that
        decodes to api_key passes auth — ``%61pi_key=``, ``api_%6Bey=``, the
        fully-encoded form — and enumeration-based regexes can never be
        complete. The filter decodes each param name and compares it to the
        exact auth name; the wire spelling is preserved."""
        for wire in (
            "%61pi_key",  # 'a' encoded
            "api_%6Bey",  # 'k' encoded
            "%61%70%69_%6B%65%79",  # every letter encoded
            "%41PI_KEY",  # decodes to API_KEY — auth does NOT read this name
        ):
            record = _record("uvicorn.error", WS_MSG, ("127.0.0.1:1", f"/ws/agent?{wire}=SECRET"))
            assert _ApiKeyRedactionFilter().filter(record) is True
            if wire == "%41PI_KEY":
                # Not the auth param name — nothing to leak past auth; the
                # line passes through untouched.
                assert record.args[1] == f"/ws/agent?{wire}=SECRET"
            else:
                assert record.args[1] == f"/ws/agent?{wire}=[REDACTED]", wire
                assert "SECRET" not in record.getMessage()


class TestInstallFunction:
    # Dependency note (issue #84): handle() installs these filters BEFORE
    # uvicorn.run, and they survive uvicorn's dictConfig because its
    # LOGGING_CONFIG keeps disable_existing_loggers=False and configures no
    # "filters" key for these loggers — the dictConfig reset rebinds
    # handlers but leaves pre-existing logger-level filters attached. If a
    # future uvicorn release changes that, the WS-accepted leak resurfaces
    # even with this filter installed; that coupling is why both loggers are
    # asserted here.

    @pytest.fixture
    def uvicorn_loggers(self):
        access = logging.getLogger("uvicorn.access")
        error = logging.getLogger("uvicorn.error")
        before = (list(access.filters), list(error.filters))
        yield access, error
        # Restore so the installed filter cannot pollute other tests'
        # logging state (the filter mutates records in place).
        access.filters[:] = before[0]
        error.filters[:] = before[1]

    def test_install_attaches_filter_to_both_loggers(self, uvicorn_loggers):
        access, error = uvicorn_loggers
        _install_api_key_redaction()
        assert any(isinstance(f, _ApiKeyRedactionFilter) for f in access.filters)
        assert any(isinstance(f, _ApiKeyRedactionFilter) for f in error.filters)

    def test_install_is_idempotent(self, uvicorn_loggers):
        """A second handle() in the same process must not stack filter
        instances (each install would re-scan every record)."""
        access, error = uvicorn_loggers
        _install_api_key_redaction()
        _install_api_key_redaction()
        assert sum(isinstance(f, _ApiKeyRedactionFilter) for f in access.filters) == 1
        assert sum(isinstance(f, _ApiKeyRedactionFilter) for f in error.filters) == 1

    def test_filters_survive_uvicorn_dictconfig(self, uvicorn_loggers):
        """Pin the upgrade coupling asserted above: uvicorn.run applies its
        LOGGING_CONFIG via dictConfig AFTER handle() installed the filters —
        if a future uvicorn starts clearing pre-existing filters, this is the
        test that says so instead of a silent log leak."""
        import logging.config

        import uvicorn.config

        access, error = uvicorn_loggers
        _install_api_key_redaction()
        names = ("uvicorn", "uvicorn.access", "uvicorn.error")
        snapshot = {
            name: (
                list(logging.getLogger(name).filters),
                list(logging.getLogger(name).handlers),
                logging.getLogger(name).level,
                logging.getLogger(name).propagate,
            )
            for name in names
        }
        try:
            logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
            assert any(isinstance(f, _ApiKeyRedactionFilter) for f in access.filters)
            assert any(isinstance(f, _ApiKeyRedactionFilter) for f in error.filters)
        finally:
            for name, (filters, handlers, level, propagate) in snapshot.items():
                logger = logging.getLogger(name)
                logger.filters[:] = filters
                logger.handlers[:] = handlers
                logger.setLevel(level)
                logger.propagate = propagate

    def test_installed_pipeline_redacts_websocket_line(self, uvicorn_loggers):
        """End-to-end through the uvicorn.error logger (filter installed the
        way handle() installs it): the emitted line carries no secret."""
        access, error = uvicorn_loggers
        _install_api_key_redaction()
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        handler.setLevel(logging.INFO)
        original_level = error.level
        error.setLevel(logging.INFO)
        error.addHandler(handler)
        try:
            error.propagate = False
            error.info(WS_MSG, *WS_ARGS)
        finally:
            error.removeHandler(handler)
            error.propagate = True
            error.setLevel(original_level)
        assert records, "logger-level filter suppressed the record entirely"
        text = logging.Formatter("%(message)s").format(records[0])
        assert "SECRET" not in text
        assert "api_key=[REDACTED]" in text
