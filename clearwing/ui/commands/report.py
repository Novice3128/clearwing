"""Report subcommand."""

from pathlib import Path

from ...core.config import default_results_dir
from ...data.database import Database

_SESSION_REPORT_DIRS = ("sessions", "sourcehunt")


def add_parser(subparsers):
    parser = subparsers.add_parser("report", help="Show a session/target report")
    parser.add_argument("target", nargs="?", help="Target IP address (legacy DB view)")
    parser.add_argument(
        "-s", "--session", help="Session id whose report to show (e.g. sh-1234abcd)"
    )
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json", "html", "markdown"],
        default="text",
        help="Report format",
    )
    return parser


def _find_session_report(session_id: str) -> Path | None:
    for subdir in _SESSION_REPORT_DIRS:
        base = Path(default_results_dir(subdir))
        direct = base / session_id / "report.md"
        if direct.is_file():
            return direct
        matches = sorted(base.glob(f"{session_id}*/report.md"))
        if matches:
            return matches[0]
    return None


def handle(cli, args):
    """Show a session report, or the legacy per-target DB summary."""
    if args.session:
        if args.format not in ("markdown", "text"):
            cli.console.print(
                f"[red]Session reports are stored as markdown; -f {args.format} "
                "is not supported for --session[/red]"
            )
            return
        path = _find_session_report(args.session)
        if path is None:
            cli.console.print(f"[red]No report found for session {args.session}[/red]")
            cli.console.print(
                f"Looked under {default_results_dir('sessions')} and "
                f"{default_results_dir('sourcehunt')}"
            )
            return
        content = path.read_text(encoding="utf-8")
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
            cli.console.print(f"[green]Report copied to {out}[/green]")
        else:
            cli.console.print(f"[blue]Session report: {path}[/blue]\n")
            print(content)
        return

    if not args.target:
        cli.console.print("Provide a session id (--session <id>) or a target IP")
        return

    db = Database()
    target = db.get_target(args.target)

    if not target:
        cli.console.print(f"[red]Target {args.target} not found in database[/red]")
        return

    cli.console.print(f"[blue]Report for {args.target}[/blue]")
    cli.console.print(f"OS: {target.get('os', 'Unknown')}")
    cli.console.print(f"Last Scan: {target.get('last_scan', 'Unknown')}")
