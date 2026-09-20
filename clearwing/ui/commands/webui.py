"""Web UI subcommand."""

import logging
import re
from urllib.parse import unquote


class _ApiKeyRedactionFilter(logging.Filter):
    """Scrub ``api_key=<secret>`` from uvicorn request-log records.

    The browser-compatible auth path is ``?api_key=`` (query param), and
    uvicorn logs the full path+query verbatim — without this filter every
    authenticated request writes the operator key to the log file.
    """

    # A query parameter is `name=value`. The name charset (unreserved chars
    # + percent escapes) anchors the match INSIDE the parameter — a greedy
    # "anything but =" would swallow the path prefix ("/api/x?api_key") as
    # part of the name. The value stops at '&', whitespace, and at a '?'
    # ONLY when what follows is param-shaped (`name=`): a rendered request
    # target normally carries '?' as the path/query separator (so
    # "url=/x?api_key=…" in a message template must not hide the api_key
    # token), but a VALUE containing a literal '?' is legal on the wire —
    # parse_qs does not split values at '?' — and truncating there leaked
    # `api_key=?secret` whole and `api_key=abc?def` as a tail (red-team
    # round). Residual, disclosed: a key containing a literal "?name="
    # shape is indistinguishable from abutting params and splits at it —
    # the conservative (over-redaction) direction. Matching is
    # DECODE-AWARE, not spelling-enumeration: starlette's parse_qs
    # percent-decodes param NAMES, so ANY encoded spelling that decodes to
    # ``api_key`` passes auth (``api%5Fkey=``, ``%61pi_key=``,
    # ``api_%6Bey=``, …) — enumerating spellings in the regex can never be
    # complete. Instead every param-shaped token is decoded and compared to
    # the exact name the auth reads (case-sensitive, matching parse_qs
    # semantics); the WIRE spelling of the name is preserved in the
    # redacted output. The value pattern requires at least one character:
    # an empty value carries no secret.
    _PARAM_RE = re.compile(r"([A-Za-z0-9%_.~\-]+)=((?:[^?&\s]|\?(?![A-Za-z0-9%_.~\-]+=))+)")
    _REDACTED = "[REDACTED]"
    _AUTH_PARAM = "api_key"

    def _redact_text(self, text: str) -> str:
        """Replace every api_key param's value with the redaction marker."""

        def _redact_param(match: re.Match[str]) -> str:
            if unquote(match.group(1)) == self._AUTH_PARAM:
                return f"{match.group(1)}={self._REDACTED}"
            return match.group(0)

        return self._PARAM_RE.sub(_redact_param, text)

    def _has_live_api_key(self, text: str) -> bool:
        """True when an api_key param carries something not yet redacted.

        ``startswith`` (not equality): the collapsed form can abut log
        punctuation — ``api_key=[REDACTED]"`` from a quoted template — and
        that is not a live secret.
        """

        for match in self._PARAM_RE.finditer(text):
            if (
                unquote(match.group(1)) == self._AUTH_PARAM
                and not match.group(2).startswith(self._REDACTED)
            ):
                return True
        return False

    def _mentions_api_key(self, text: str) -> bool:
        """True when any api_key param appears, already redacted or not."""

        return any(
            unquote(match.group(1)) == self._AUTH_PARAM
            for match in self._PARAM_RE.finditer(text)
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._mentions_api_key(record.getMessage()):
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
                self._redact_text(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
            if not self._has_live_api_key(record.getMessage()):
                return True
        # No args to redact (argless record), or the leak lives in the msg
        # template itself rather than in any arg: collapse to a fully
        # rendered, redacted message.
        record.msg = self._redact_text(record.getMessage())
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
