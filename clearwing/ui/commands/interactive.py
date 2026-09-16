"""Interactive agent subcommand."""

import asyncio
import logging
import os
import socket
from datetime import datetime

from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from clearwing.agent.graph import create_agent
from clearwing.agent.runtime import Command
from clearwing.agent.tools.ops.dynamic_tool_creator import get_custom_tools
from clearwing.agent.tools.ops.kali_docker_tool import kali_cleanup
from clearwing.data.memory import SessionStore
from clearwing.observability.telemetry import CostTracker
from clearwing.providers.env import DEFAULT_ANTHROPIC_MODEL
from clearwing.ui.tui import ClearwingApp

logger = logging.getLogger(__name__)


def add_parser(subparsers):
    parser = subparsers.add_parser("interactive", help="Start interactive AI agent")
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name; when omitted (or with --base-url, where the "
        "endpoint's own model is guessed) the model is resolved from "
        "config.yaml / env, falling back to claude-sonnet-4-6",
    )
    parser.add_argument("--target", help="Initial target IP address")
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a previous session by ID; the stored model is pinned "
        "for the resumed run (pass --model to override it)",
    )
    parser.add_argument(
        "--no-tui", action="store_true", help="Use legacy Rich-based loop instead of Textual TUI"
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        metavar="DOLLARS",
        help="Stop agent when estimated cost exceeds this amount (USD)",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="OpenAI-compatible API base URL (for vLLM, Ollama, MLX, OpenRouter, etc.)",
    )
    parser.add_argument(
        "--api-key", metavar="KEY", help="API key for the endpoint (overrides env vars)"
    )
    return parser


def handle(cli, args):
    """Run the interactive AI agent loop."""
    # `--model` unset means "defer to config.yaml / env" (#8/#22): the old
    # argparse default was indistinguishable from an explicit choice and
    # short-circuited the config provider section.
    args.model_explicit = args.model is not None
    if not _preflight_check(cli, args):
        return

    # Set up cost limit
    cost_limit = getattr(args, "cost_limit", None)
    if cost_limit:
        tracker = CostTracker()
        tracker.cost_limit = cost_limit

    # Session management
    session = None
    store = SessionStore()
    resume_id = getattr(args, "resume", None)
    if resume_id:
        session = store.load(resume_id)
        if session:
            cli.console.print(f"[green]Resuming session {session.session_id}[/green]")
            args.target = session.target
            # A stored value equal to the legacy argparse default is
            # ambiguous on its own: sessions saved before that default was
            # removed stored the placeholder even when the user never chose
            # a model. New sessions persist the choice intent alongside
            # (model_explicit); legacy rows default to "defer". An explicit
            # --model on this invocation still wins over the stored value.
            if session.model and args.model is None:
                args.model = session.model
                if session.model != DEFAULT_ANTHROPIC_MODEL:
                    args.model_explicit = True
                else:
                    args.model_explicit = bool(getattr(session, "model_explicit", False))
        else:
            cli.console.print(f"[red]Session {resume_id} not found.[/red]")
            return
    else:
        session = store.create(
            target=args.target or "",
            model=args.model or "",
            model_explicit=getattr(args, "model_explicit", args.model is not None),
        )

    # Launch TUI or legacy mode
    use_tui = not getattr(args, "no_tui", False)
    if use_tui:
        _run_tui(cli, args, session)
        return

    _run_interactive_legacy(cli, args, session)


def _preflight_check(cli, args) -> bool:
    """Fast checks before agent starts."""
    errors = []

    # Any of these sources is enough to build an LLM:
    #   CLI flags (--base-url / --api-key)
    #   CLEARWING_BASE_URL / CLEARWING_API_KEY env vars
    #   ANTHROPIC_API_KEY env var (the pre-multi-provider default)
    has_cli_endpoint = bool(getattr(args, "base_url", None) or getattr(args, "api_key", None))
    has_clearwing_env = bool(
        os.environ.get("CLEARWING_BASE_URL") or os.environ.get("CLEARWING_API_KEY")
    )
    has_anthropic_env = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not (has_cli_endpoint or has_clearwing_env or has_anthropic_env):
        errors.append(
            "No LLM credentials found. Set ANTHROPIC_API_KEY for Anthropic direct, "
            "or CLEARWING_BASE_URL + CLEARWING_API_KEY for an OpenAI-compatible endpoint "
            "(OpenRouter, Ollama, LM Studio, vLLM, etc.), "
            "or pass --base-url and --api-key on the command line. "
            "See docs/providers.md for the full list of supported backends."
        )

    if args.target:
        try:
            socket.getaddrinfo(args.target, None)
        except socket.gaierror:
            errors.append(f"Target '{args.target}' is not a valid IP or resolvable hostname.")

    # Skip the "valid model" warning when the user is pointing at a
    # custom endpoint — we have no way to know what models OpenRouter /
    # Ollama / LM Studio / vLLM serve — or when no model was chosen (the
    # configured provider decides).
    if not has_cli_endpoint and not has_clearwing_env and args.model is not None:
        valid_models = [
            "claude-sonnet-4-6",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-haiku-4-5",
        ]
        if args.model not in valid_models:
            cli.console.print(
                f"[yellow]Warning: Model '{args.model}' is not a recognized Anthropic "
                f"model. Known models: {', '.join(valid_models)}[/yellow]"
            )

    if errors:
        for err in errors:
            cli.console.print(f"[red]Preflight error: {err}[/red]")
        return False
    return True


def _sync_session_model(session, graph) -> None:
    """#28: a deferred model leaves the session row as model="" at creation
    time; write the graph's resolved model back so /api/sessions and
    --resume see what actually ran. Mutates only — callers persist the row."""
    if session is None:
        return
    resolved = getattr(getattr(graph, "llm", None), "model_name", None)
    if resolved and session.model != resolved:
        session.model = resolved


def _run_tui(cli, args, session=None):
    """Launch the Textual TUI.

    The session row write-back (resolved model + final status) lives in a
    finally: a crashing `app.run()` must still persist what actually ran —
    the graph is created inside the TUI's on_mount, so skipping the
    write-back on crash would leave a deferred-model session row as ""
    forever. A crash re-raises after the save (the CLI still exits loud).
    """
    session_id = session.session_id if session else None
    app = ClearwingApp(
        target=args.target,
        model=args.model,
        session_id=session_id,
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        model_explicit=getattr(args, "model_explicit", args.model is not None),
    )
    exit_status = "completed"
    try:
        app.run()
    except Exception:
        exit_status = "error"
        logger.exception("TUI crashed")
        raise
    finally:
        if session:
            try:
                session.status = exit_status
                # The graph is created inside the TUI's on_mount; mirror its
                # resolved model into the session row before the final save.
                _sync_session_model(session, getattr(app, "_agent_graph", None))
                SessionStore().save(session)
            except Exception:
                logger.debug("Failed to save session on TUI exit", exc_info=True)


def _run_interactive_legacy(cli, args, session=None):
    """Run the legacy Rich-based interactive agent loop."""
    cli.console.print(
        Panel.fit(
            "[bold cyan]Clearwing Interactive Agent[/bold cyan]\n"
            f"Model: {args.model or '(from config)'}\n"
            "Type 'quit' or 'exit' to end session."
        )
    )

    thread_id = session.thread_id if session else "interactive-session"
    graph = create_agent(
        model_name=args.model,
        session_id=session.session_id if session else None,
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        model_explicit=getattr(args, "model_explicit", args.model is not None),
    )
    # #28: persist the resolved model (deferred sessions are created with
    # model=""); the autosave below and the final save carry it onward.
    _sync_session_model(session, graph)
    if session and session.model:
        try:
            SessionStore().save(session)
        except Exception:
            logger.debug("Failed to persist resolved session model", exc_info=True)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "target": args.target,
        "open_ports": session.open_ports if session else [],
        "services": session.services if session else [],
        "vulnerabilities": session.vulnerabilities if session else [],
        "exploit_results": session.exploit_results if session else [],
        "os_info": session.os_info if session else None,
        "kali_container_id": session.kali_container_id if session else None,
        "custom_tool_names": session.custom_tool_names if session else [],
        "session_id": session.session_id if session else None,
        "flags_found": session.flags_found if session else [],
        "loaded_skills": [],
        "paused": False,
        "total_cost_usd": session.cost_usd if session else 0.0,
        "total_tokens": session.token_count if session else 0,
    }

    if args.target:
        cli.console.print(f"[blue]Target set to: {args.target}[/blue]")

    class _StreamPrinter:
        """Print streaming text deltas, emitting the ``Agent:`` label once."""

        def __init__(self, console):
            self._console = console
            self._started = False

        def reset(self):
            self._started = False

        def __call__(self, delta: str):
            if not delta:
                return
            if not self._started:
                self._console.print("\n[bold cyan]Agent:[/bold cyan] ", end="")
                self._started = True
            self._console.print(delta, end="", highlight=False)

        def finish(self):
            if self._started:
                self._console.print()

    stream_printer = _StreamPrinter(cli.console)
    graph.on_text_delta = stream_printer

    known_custom_tools = set()

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit"):
            break

        if not user_input.strip():
            continue

        stream_printer.reset()

        input_msg = {"messages": [{"role": "user", "content": user_input}]}
        input_msg.update(initial_state)
        initial_state = {}

        async def _collect_events(g, msg) -> list:
            return [ev async for ev in g.astream(msg, config, stream_mode="values")]

        try:
            while True:
                events = asyncio.run(_collect_events(graph, input_msg))
                interrupted = False

                stream_printer.finish()
                for _event in events:
                    pass  # streaming callback already printed text

                # Check for interrupt
                state = graph.get_state(config)
                if state.next:
                    tasks = state.tasks
                    if tasks:
                        for task in tasks:
                            if hasattr(task, "interrupts") and task.interrupts:
                                for intr in task.interrupts:
                                    prompt = str(intr.value)
                                    approved = Confirm.ask(
                                        f"[bold yellow]APPROVAL REQUIRED:[/bold yellow] {prompt}"
                                    )
                                    input_msg = Command(resume=approved)
                                    interrupted = True
                                    break
                            if interrupted:
                                break

                if not interrupted:
                    break

            # Check for new custom tools
            current_state = graph.get_state(config)
            if hasattr(current_state, "values"):
                new_custom = set(current_state.values.get("custom_tool_names", []))
                if new_custom - known_custom_tools:
                    known_custom_tools = new_custom
                    custom_tools = get_custom_tools()
                    graph = create_agent(
                        model_name=args.model,
                        custom_tools=custom_tools,
                        session_id=session.session_id if session else None,
                        base_url=getattr(args, "base_url", None),
                        api_key=getattr(args, "api_key", None),
                        model_explicit=getattr(
                            args, "model_explicit", args.model is not None
                        ),
                    )
                    cli.console.print("[dim]Graph recompiled with new custom tools.[/dim]")

            # Auto-save session
            if session:
                try:
                    sv = current_state.values if hasattr(current_state, "values") else {}
                    session.open_ports = sv.get("open_ports", session.open_ports)
                    session.services = sv.get("services", session.services)
                    session.vulnerabilities = sv.get("vulnerabilities", session.vulnerabilities)
                    session.exploit_results = sv.get("exploit_results", session.exploit_results)
                    session.os_info = sv.get("os_info", session.os_info)
                    session.kali_container_id = sv.get(
                        "kali_container_id", session.kali_container_id
                    )
                    session.custom_tool_names = sv.get(
                        "custom_tool_names", session.custom_tool_names
                    )
                    session.flags_found = sv.get("flags_found", session.flags_found)
                    session.cost_usd = sv.get("total_cost_usd", session.cost_usd)
                    session.token_count = sv.get("total_tokens", session.token_count)
                    SessionStore().save(session)
                except Exception:
                    logger.debug("Failed to save session mid-loop", exc_info=True)

        except KeyboardInterrupt:
            cli.console.print("\n[yellow]Interrupted.[/yellow]")
            continue
        except Exception as e:
            cli.console.print(f"[red]Error: {e}[/red]")
            continue

    # Cleanup Kali container if active
    try:
        final_state = graph.get_state(config)
        container_id = None
        if hasattr(final_state, "values"):
            container_id = final_state.values.get("kali_container_id")
        if container_id:
            cli.console.print("[dim]Cleaning up Kali container...[/dim]")
            kali_cleanup.invoke({"container_id": container_id})
    except Exception:
        logger.debug("Kali container cleanup failed", exc_info=True)

    # Final session save
    if session:
        try:
            session.status = "completed"
            session.end_time = datetime.now()
            SessionStore().save(session)
        except Exception:
            logger.debug("Failed to save final session state", exc_info=True)

    cli.console.print("[bold cyan]Session ended.[/bold cyan]")
