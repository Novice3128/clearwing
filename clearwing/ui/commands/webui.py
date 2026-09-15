"""Web UI subcommand."""

import logging
import re


class _ApiKeyRedactionFilter(logging.Filter):
    """Scrub ``api_key=<secret>`` from uvicorn access-log records.

    The browser-compatible auth path is ``?api_key=`` (query param), and
    uvicorn logs the full path+query verbatim — without this filter every
    authenticated request writes the operator key to the log file.
    """

    _QUERY_RE = re.compile(r"(api_key=)[^&\s]+")

    def filter(self, record: logging.LogRecord) -> bool:
        if "api_key=" not in record.getMessage():
            return True
        if record.args:
            record.args = tuple(
                self._QUERY_RE.sub(r"\1[REDACTED]", arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        # Re-render in case the secret sits in msg itself rather than args.
        if "api_key=" in record.getMessage():
            record.msg = self._QUERY_RE.sub(r"\1[REDACTED]", record.getMessage())
            record.args = None
        return True


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

    logging.getLogger("uvicorn.access").addFilter(_ApiKeyRedactionFilter())

    app = create_app()

    cli.console.print(
        f"[bold cyan]Clearwing Web UI[/bold cyan]\n"
        f"Starting server at http://{args.host}:{args.port}\n"
        f"API docs at http://{args.host}:{args.port}/docs"
    )

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
