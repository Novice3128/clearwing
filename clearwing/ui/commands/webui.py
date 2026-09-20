"""Web UI subcommand."""

import logging
import re


class _ApiKeyRedactionFilter(logging.Filter):
    """Scrub ``api_key=<secret>`` from uvicorn request-log records.

    The browser-compatible auth path is ``?api_key=`` (query param), and
    uvicorn logs the full path+query verbatim — without this filter every
    authenticated request writes the operator key to the log file.
    """

    # The param name itself may arrive percent-encoded (``api%5Fkey=``):
    # starlette's parse_qs percent-decodes param NAMES, so that spelling
    # passes auth and must be redacted on the wire form too.
    _QUERY_RE = re.compile(r"(api(?:%5[Ff]|_)key=)[^&\s]+")
    # ``api_key=`` followed by anything OTHER than the redaction marker —
    # detects a live secret. A plain substring test cannot do this: the
    # redacted form ``api_key=[REDACTED]`` still contains ``api_key=``, so a
    # literal "still contains api_key=" check would collapse EVERY matching
    # record (the original bug: it re-rendered the already-clean message and
    # set args=None unconditionally).
    _LEAK_RE = re.compile(r"api(?:%5[Ff]|_)key=(?!\[REDACTED\])[^&\s]")

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._QUERY_RE.search(record.getMessage()):
            return True
        if isinstance(record.args, tuple | list) and record.args:
            # Main path: the secret sits inside one of the args — uvicorn
            # access records use a 5-tuple (client, method, full_path,
            # http_version, status) and the WebSocket accept/reject lines on
            # uvicorn.error use a 2-tuple whose second element is the full
            # path+query. Redact the string args IN PLACE and keep
            # record.args in tuple shape: uvicorn's AccessFormatter unpacks
            # record.args, so flattening to a pre-rendered msg with
            # args=None raises TypeError in every handler and spams
            # "--- Logging error ---" tracebacks instead of the line.
            record.args = tuple(
                self._QUERY_RE.sub(r"\1[REDACTED]", arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
            if not self._LEAK_RE.search(record.getMessage()):
                return True
        # No args to redact (argless record), or the leak lives in the msg
        # template itself rather than in any arg: collapse to a fully
        # rendered, redacted message.
        record.msg = self._QUERY_RE.sub(r"\1[REDACTED]", record.getMessage())
        record.args = None
        return True


def _install_api_key_redaction() -> None:
    """Attach the redaction filter to both loggers uvicorn uses for URLs.

    ``uvicorn.access`` carries the HTTP request lines, but the WebSocket
    accept/reject lines (``... "WebSocket /ws/agent?api_key=..." [accepted]``)
    are emitted on ``uvicorn.error`` (all three WS protocol impls route
    through it) — installing on one logger only leaves the key in the clear
    on the other.

    NB: this must run BEFORE ``uvicorn.run``, and the filters survive
    uvicorn's dictConfig by design — its LOGGING_CONFIG keeps
    ``disable_existing_loggers=False`` and configures no ``filters`` key for
    these loggers, so the dictConfig reset rebinds handlers but leaves
    pre-existing logger-level filters attached.

    Defense-in-depth note: a trace-level run (``uvicorn.error`` effective
    level <= TRACE) wraps the app in MessageLoggerMiddleware, whose
    ``uvicorn.asgi`` dump includes the raw scope AND headers — the API key
    would appear outside the ``api_key=`` pattern this filter redacts. Do
    not operate the webui at trace log level.
    """
    redaction = _ApiKeyRedactionFilter()
    for logger_name in ("uvicorn.access", "uvicorn.error"):
        target = logging.getLogger(logger_name)
        # Idempotent: a second handle() in the same process (e.g. tests)
        # must not stack filter instances on the loggers.
        if not any(isinstance(existing, _ApiKeyRedactionFilter) for existing in target.filters):
            target.addFilter(redaction)


def add_parser(subparsers):
    parser = subparsers.add_parser("webui", help="Start the web UI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8899, help="Port to bind (default: 8899)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    return parser


def handle(cli, args):
    """Start the FastAPI web UI server."""
    try:
        import uvicorn
    except ImportError:
        cli.console.print(
            "[red]uvicorn is required for the web UI. "
            "Install with: pip install 'clearwing[web]'[/red]"
        )
        return

    from ..web import create_app

    _install_api_key_redaction()

    app = create_app()

    cli.console.print(
        f"[bold cyan]Clearwing Web UI[/bold cyan]\n"
        f"Starting server at http://{args.host}:{args.port}\n"
        f"API docs at http://{args.host}:{args.port}/docs"
    )

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
