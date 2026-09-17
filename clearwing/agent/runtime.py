from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

import networkx as nx

from clearwing.capabilities import capabilities
from clearwing.core.events import EventBus, EventType
from clearwing.data.knowledge import KnowledgeGraph
from clearwing.data.memory import ContextSummarizer, EpisodicMemory
from clearwing.llm.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    _coerce_chat_messages,
)
from clearwing.llm.native import NativeToolSpec, response_text
from clearwing.observability.bookkeeping import book_llm_call, init_session_audit_logger
from clearwing.observability.otel import get_oi_tracer
from clearwing.observability.telemetry import CostTracker
from clearwing.providers.binding import AgentLimits
from clearwing.safety.guardrails import InputGuardrail, OutputGuardrail

from .protocols import KnowledgeGraphPopulator, LLMInvokable, StateUpdater, SystemPromptFactory
from .tooling import AgentTool, InterruptRequest, tool_execution_context

logger = logging.getLogger(__name__)
tracer = get_oi_tracer(__name__)

# Consecutive-identical-failure guard: without it a model can loop on the
# same failing tool call forever (session 932ff8ec repeated one failing
# call 293 times, ~$249). Nudge at N, halt the turn at 2N. 0 disables.
_DEFAULT_FAILURE_STREAK_NUDGE = 6

_FAILURE_MARKERS = (
    '"error"',
    "error:",
    "traceback (most recent call last)",
    "unauthorized",
    "permission denied",
    "operation not permitted",
)


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.environ.get(name) or "").strip() or default)
    except ValueError:
        return default


def _tool_result_failed(content: str) -> bool:
    if "denied by user" in content[:500].lower():
        return False  # a human decision, not a malfunction
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        if isinstance(error, str):
            return bool(error.strip())
        if error is not None:
            # Structured failure payloads: {"error": {"code": ..., ...}}.
            return True
        if parsed.get("status") == "error" or parsed.get("ok") is False:
            # Repo conventions: callback_listener {"status": "error", ...},
            # potentials {"ok": False, "error": {...}}.
            return True
        return False
    head = content[:2000].lower()
    return any(marker in head for marker in _FAILURE_MARKERS)


def _synthesize_skipped_tool_results(
    tool_calls: list[Any], reason: str
) -> list[ToolMessage]:
    """Placeholder ToolMessages for tool calls that will never run.

    Providers reject a history where an assistant ``tool_use`` has no
    matching ``tool_result`` (Anthropic/OpenAI both 400), so any path that
    abandons a tool batch must answer every call before the turn ends.
    """
    return [
        ToolMessage(
            content=json.dumps({"error": f"skipped: {reason}"}),
            name=str(getattr(tool_call, "fn_name", "") or ""),
            tool_call_id=getattr(tool_call, "call_id", None),
        )
        for tool_call in tool_calls
    ]


def _streak_key(tool_name: str, tool_args: dict[str, Any]) -> str:
    try:
        args_json = json.dumps(tool_args, sort_keys=True, default=str)
    except Exception:
        args_json = str(tool_args)
    return f"{tool_name}:{args_json}"


FLAG_PATTERNS = [
    re.compile(r"flag\{[^}]+\}", re.IGNORECASE),
    re.compile(r"FLAG\{[^}]+\}"),
    re.compile(r"HTB\{[^}]+\}"),
    re.compile(r"CTF\{[^}]+\}"),
    # Hex-boundary lookaround (issue #35): a bare 32-hex class matched any
    # window of longer hex strings — a 64-hex container id scanned as TWO
    # "flags". Every window of a longer hex run has a hex neighbour, so the
    # lookarounds reject them all while a standalone 32-hex (MD5-style
    # flag) still matches.
    re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{32}(?![A-Fa-f0-9])"),
]

# Identifier fields embedded in structured tool results — bookkeeping, not
# CTF loot. Masked before flag scanning so long hex ids (container ids,
# image digests, ...) can never register as flags (issue #35).
_FLAG_SCAN_EXEMPT_KEYS = frozenset(
    {
        "container_id",
        "kali_container_id",
        "image_id",
        "sandbox_id",
        "session_id",
        "checkpoint_id",
        "run_id",
        "request_id",
        "trace_id",
    }
)


def _strip_id_fields(data: Any) -> Any:
    """Recursively replace known identifier values with a placeholder."""
    if isinstance(data, dict):
        return {
            key: ("<id>" if key in _FLAG_SCAN_EXEMPT_KEYS else _strip_id_fields(value))
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_strip_id_fields(item) for item in data]
    return data


def detect_flags(text: str) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    seen: set[str] = set()
    for pattern in FLAG_PATTERNS:
        for match in pattern.finditer(text):
            flag = match.group()
            # Cross-pattern dedup, order-preserving (issue #35): the
            # case-insensitive flag{} pattern and the exact-case FLAG{}
            # pattern both fire on the same capture.
            if flag in seen:
                continue
            seen.add(flag)
            flags.append({"flag": flag, "pattern": pattern.pattern})
    return flags


def _parse_tool_output(content: str) -> Any:
    try:
        return json.loads(content)
    except Exception:
        try:
            return ast.literal_eval(content)
        except Exception:
            return content


class ToolCallDict(TypedDict, total=False):
    """Legacy LangChain-style tool-call shape.

    Retained only for the ``_default_pentest_state_updater`` docs and any
    dict-shaped callers. The native runtime now consumes genai ``ToolCall``
    objects (``.call_id`` / ``.fn_name`` / ``.fn_arguments``) directly.
    """

    id: str
    name: str
    args: dict[str, Any]


@dataclass(slots=True)
class Command:
    resume: bool


@dataclass(slots=True)
class GraphInterrupt:
    value: str


@dataclass(slots=True)
class GraphTask:
    interrupts: list[GraphInterrupt] = field(default_factory=list)


@dataclass(slots=True)
class GraphStateSnapshot:
    values: dict[str, Any]
    next: tuple[str, ...] = ()
    tasks: list[GraphTask] = field(default_factory=list)


@dataclass(slots=True)
class _PendingToolResume:
    tool_calls: list[Any]
    prompt: str


class NativeAgentGraph:
    def __init__(
        self,
        *,
        llm: LLMInvokable,
        native_tools: list[NativeToolSpec],
        tools: list[AgentTool],
        system_prompt_fn: SystemPromptFactory,
        model_name: str,
        session_id: str | None,
        state_updater_fn: StateUpdater,
        knowledge_graph_populator_fn: KnowledgeGraphPopulator | None,
        input_guardrail_tool_names: set[str] | frozenset[str],
        output_guardrail_tool_names: set[str] | frozenset[str],
        enable_cost_tracker: bool,
        enable_episodic_memory: bool,
        enable_audit: bool,
        enable_knowledge_graph: bool,
        enable_input_guardrail: bool,
        enable_output_guardrail: bool,
        enable_event_bus: bool,
        enable_context_summarizer: bool,
        agent_limits: AgentLimits | None = None,
        dynamic_context_fn: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        self.llm = llm
        self.agent_limits = agent_limits
        self.native_tools = native_tools
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt_fn = system_prompt_fn
        self.model_name = model_name
        # Stable per-graph label: prompt-cache routing key and session
        # attribution (issue #10 live evidence: the graph never kept it).
        self.session_id = session_id
        # Renders the per-step dynamic context (scan state, loaded skills,
        # episodic recall, flags) that rides AFTER the cache breakpoint —
        # keeping the system prompt byte-stable for prefix caching (#36).
        self.dynamic_context_fn = dynamic_context_fn
        self.state_updater_fn = state_updater_fn
        self.knowledge_graph_populator_fn = knowledge_graph_populator_fn
        self.input_guardrail_tool_names = set(input_guardrail_tool_names)
        self.output_guardrail_tool_names = set(output_guardrail_tool_names)
        self.on_text_delta: Callable[[str], None] | None = None
        self._state: dict[str, dict[str, Any]] = {}
        self._pending: dict[str, _PendingToolResume | None] = {}
        # thread_id -> (streak_key, consecutive identical failure count)
        self._failure_runs: dict[str, tuple[str, int]] = {}
        # thread_id -> {"steps": n, "tool_calls_total": m} for the logical
        # turn's budget guards. Must survive approval resumes (_aresume
        # restarts _arun_loop, which re-reads these) and reset only when a
        # new user turn starts. Kept out of graph state on purpose:
        # _merge_input copies arbitrary input keys into state, so a state
        # key could be clobbered to re-grant the budget (issue #23).
        self._loop_counters: dict[str, dict[str, int]] = {}
        # Per-graph, per-THREAD cost/token accumulation (issues #37/#52).
        # The CostTracker is a process-wide singleton, so its running totals
        # pool EVERY session and operator job in the process — writing them
        # into state made each job report the cross-job total. Accumulate on
        # the instance instead (like _loop_counters, deliberately NOT state:
        # _merge_input copies arbitrary input keys into state). #52: one
        # graph CAN serve several thread_ids, so the totals key by the
        # owning thread (see _thread_for_state) — thread A's spend
        # must never write into thread B's state, in either direction. The
        # global tracker stays for process-level observation only.
        self._cost_totals: dict[str, dict[str, float]] = {}

        self.cost_tracker = (
            CostTracker() if enable_cost_tracker and capabilities.has("telemetry") else None
        )
        self.episodic_memory = (
            # session_id flows into every recorded episode; without it all
            # rows landed with session_id='' and could not be attributed
            # (issue #19).
            EpisodicMemory(session_id=session_id or "")
            if enable_episodic_memory and capabilities.has("memory")
            else None
        )
        self.context_summarizer = (
            ContextSummarizer()
            if enable_context_summarizer and capabilities.has("memory")
            else None
        )
        self.event_bus = EventBus() if enable_event_bus and capabilities.has("events") else None
        if self.event_bus is not None:
            # Retry/fallback visibility (issue #17): the LLM layer (and the
            # FallbackChain) report waits and provider switches as they
            # happen instead of surfacing them only in the final error's
            # attempts count. The bus is thread-safe, so notices fired from
            # worker threads land safely on the webui pump.
            notice_setter = getattr(self.llm, "set_retry_notice", None)
            if notice_setter is not None:
                bus = self.event_bus

                def _llm_notice(text: str) -> None:
                    # session_id rides the payload so the webui pump can
                    # scope retry/fallback chatter to this session's socket
                    # instead of broadcasting every session's notices to
                    # every connected client.
                    bus.emit(
                        EventType.MESSAGE,
                        {"content": text, "type": "system", "session_id": self.session_id},
                    )

                notice_setter(_llm_notice)
        self.input_guardrail = (
            InputGuardrail() if enable_input_guardrail and capabilities.has("guardrails") else None
        )
        self.output_guardrail = (
            OutputGuardrail()
            if enable_output_guardrail and capabilities.has("guardrails")
            else None
        )
        self.audit_logger = init_session_audit_logger(session_id) if enable_audit else None

        self.knowledge_graph = None
        if enable_knowledge_graph and capabilities.has("knowledge"):
            try:
                from clearwing.core.config import clearwing_home

                self.knowledge_graph = KnowledgeGraph(
                    persist_path=str(clearwing_home() / "knowledge_graph.json"),
                )
            except Exception:
                logger.warning("Failed to initialize KnowledgeGraph", exc_info=True)

    async def astream(
        self, input_data: dict[str, Any] | Command, config: dict, stream_mode: str = "values"
    ):
        del stream_mode
        thread_id = self._thread_id(config)
        try:
            if isinstance(input_data, Command):
                async for event in self._aresume(thread_id, input_data.resume):
                    yield event
                return

            state = self._get_or_create_state(thread_id)
            if self._pending.get(thread_id) is not None:
                # A new user turn on top of a suspended approval batch would
                # orphan its tool_use (provider 400). Interactive clients gate
                # this earlier; this is the last-line defense for direct
                # astream callers.
                logger.warning(
                    "Discarding pending approval batch before new input on %s",
                    thread_id,
                )
                self._stop_cleanup(thread_id)
            # A new user turn starts a fresh logical turn: re-grant the
            # budget guards. The Command(resume) path above must NOT reset —
            # repeated approve would otherwise bypass max_steps/max_tool_calls
            # (issue #23). Operator stop/cancel also deliberately preserve
            # spent budget: only new input resets.
            self._loop_counters.pop(thread_id, None)
            self._merge_input(state, input_data)
            async for event in self._arun_loop(thread_id):
                yield event
        except asyncio.CancelledError:
            # Operator stop: an abandoned tool batch must still be answered
            # (providers 400 a history with orphaned tool_use) before the
            # cancellation propagates to the consumer. NB: this covers
            # cancellations landing on an await INSIDE the generator; if a
            # consumer adds awaits to its for-body, the stop branch in the
            # webui still answers via discard_interrupt.
            self._stop_cleanup(thread_id)
            raise

    async def ainvoke(
        self, input_data: dict[str, Any] | Command, config: dict
    ) -> GraphStateSnapshot:
        async for _ in self.astream(input_data, config):
            pass
        return self.get_state(config)

    def get_state(self, config: dict) -> GraphStateSnapshot:
        thread_id = self._thread_id(config)
        state = self._get_or_create_state(thread_id)
        pending = self._pending.get(thread_id)
        if pending is None:
            return GraphStateSnapshot(values=state, next=(), tasks=[])
        return GraphStateSnapshot(
            values=state,
            next=("tools",),
            tasks=[GraphTask(interrupts=[GraphInterrupt(value=pending.prompt)])],
        )

    def discard_interrupt(self, config: dict) -> bool:
        """Answer a pending approval interrupt (and any dangling tool batch)
        as skipped.

        Used when an operator stops a session that is suspended waiting for
        an ``approve`` frame: without this, the pending batch's tool_calls
        stay unanswered and the next turn 400s on orphaned tool_use. Returns
        True when anything was actually cleaned up.
        """
        return self._stop_cleanup(self._thread_id(config))

    def _stop_cleanup(self, thread_id: str) -> bool:
        """Answer every unanswered tool call on *thread_id* as skipped."""
        cleaned = False
        pending = self._pending.get(thread_id)
        if pending is not None:
            self._pending[thread_id] = None
            state = self._get_or_create_state(thread_id)
            state.setdefault("messages", []).extend(
                _synthesize_skipped_tool_results(
                    pending.tool_calls, "session stopped by operator"
                )
            )
            cleaned = True

        # Mid-batch abandonment: the assistant requested tools but the batch
        # never finished. Scan back past the batch's trailing ToolMessages —
        # a pause/cancel at index > 0 leaves completed results between the
        # request and the unanswered tail, so messages[-1] is not the request.
        state = self._get_or_create_state(thread_id)
        messages = state.get("messages", [])
        idx = len(messages) - 1
        while idx >= 0 and getattr(messages[idx], "type", "") == "tool":
            idx -= 1
        if idx >= 0:
            last = messages[idx]
            calls = getattr(last, "tool_calls", None) or []
            if getattr(last, "type", "") == "ai" and calls:
                answered = {
                    getattr(m, "tool_call_id", None)
                    for m in messages
                    if getattr(m, "type", "") == "tool"
                }
                unanswered = [
                    c for c in calls if getattr(c, "call_id", None) not in answered
                ]
                if unanswered:
                    messages.extend(
                        _synthesize_skipped_tool_results(
                            unanswered, "session stopped by operator"
                        )
                    )
                    cleaned = True
        return cleaned

    async def _aresume(self, thread_id: str, approved: bool):
        pending = self._pending.get(thread_id)
        if pending is None:
            return
        self._pending[thread_id] = None
        state = self._get_or_create_state(thread_id)
        tool_events, paused, halted = await self._arun_tool_calls(
            state, pending.tool_calls, resume_decision=approved
        )
        for event in tool_events:
            yield event
        if paused or halted:
            return
        async for event in self._arun_loop(thread_id):
            yield event

    async def _arun_loop(self, thread_id: str):
        state = self._get_or_create_state(thread_id)
        limits = self.agent_limits
        max_steps = limits.max_steps if limits else None
        max_tool_calls = limits.max_tool_calls if limits else None
        # Counters live per-thread on the instance (not locals): _aresume
        # restarts this loop after an approval pause and must continue the
        # logical turn's budget instead of re-granting it (issue #23).
        counters = self._loop_counters.setdefault(
            thread_id, {"steps": 0, "tool_calls_total": 0}
        )
        while True:
            if max_steps is not None and counters["steps"] >= max_steps:
                logger.info("agent loop stopped: reached max_steps=%d", max_steps)
                break
            assistant_event = await self._aassistant_step(state)
            counters["steps"] += 1
            yield assistant_event
            last = state["messages"][-1]
            tool_calls = getattr(last, "tool_calls", []) or []
            if not tool_calls:
                break
            if max_tool_calls is not None:
                budget_left = max_tool_calls - counters["tool_calls_total"]
                if budget_left <= 0:
                    logger.info(
                        "agent loop stopped: reached max_tool_calls=%d", max_tool_calls
                    )
                    # The last AIMessage requested tools that will never run;
                    # answer them so the next turn doesn't send orphaned tool_use.
                    state.setdefault("messages", []).extend(
                        _synthesize_skipped_tool_results(
                            tool_calls, "tool-call budget (max_tool_calls) reached"
                        )
                    )
                    state["messages"].append(
                        HumanMessage(
                            content=(
                                "Automatic stop: the tool-call budget (max_tool_calls) has "
                                "been reached. Summarize the progress and results collected "
                                "so far."
                            )
                        )
                    )
                    if self.event_bus:
                        self.event_bus.emit_message(
                            "agent loop stopped: max_tool_calls reached", "warning"
                        )
                    break
                if budget_left < len(tool_calls):
                    # A parallel batch can exceed the remaining budget; run the
                    # head of the batch and answer the tail as skipped.
                    state.setdefault("messages", []).extend(
                        _synthesize_skipped_tool_results(
                            tool_calls[budget_left:],
                            "tool-call budget (max_tool_calls) reached mid-batch",
                        )
                    )
                    tool_calls = tool_calls[:budget_left]
            counters["tool_calls_total"] += len(tool_calls)
            tool_events, paused, halted = await self._arun_tool_calls(
                state, tool_calls, resume_decision=Ellipsis
            )
            for event in tool_events:
                yield event
            if paused or halted:
                break

    def _thread_for_state(self, state: dict[str, Any]) -> str:
        """The thread_id owning *state* (issues #37/#52).

        Identity lookup against the per-thread state registry: one graph can
        serve several thread_ids, so a shared bucket made thread A's spend
        land in thread B's state (and vice versa). Deriving the thread from
        the state dict in hand is immune to the context leaks and
        interleavings a ContextVar would introduce (a `set()` inside an
        async generator leaks into the driving task, and interleaved
        `astream`s on one graph would cross-book). States not registered
        here (direct step invocations) share the legacy "default" bucket.
        """
        for thread_id, known in self._state.items():
            if known is state:
                return thread_id
        return "default"

    def _cost_totals_for(self, thread_id: str | None) -> dict[str, float]:
        """Per-thread cost/token bucket (issues #37/#52)."""
        return self._cost_totals.setdefault(
            thread_id or "default",
            {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0},
        )

    @tracer.chain(name="agent.assistant_step")
    async def _aassistant_step(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = list(state.get("messages", []))
        if self.context_summarizer:
            # The role binding's context_budget_tokens (carried on the client)
            # drives when older turns are summarized; None keeps the
            # summarizer's built-in threshold.
            budget = getattr(self.llm, "context_budget_tokens", None)
            should = (
                self.context_summarizer.should_summarize(messages, max_tokens=budget)
                if budget
                else self.context_summarizer.should_summarize(messages)
            )
            if should:
                try:
                    # Compaction is committed to state (issue #38): the
                    # covered messages leave the history and the summary
                    # text/coverage persist, so the next step lands under
                    # the threshold again instead of re-running (and
                    # re-billing) a fresh LLM summary on every step.
                    pre_compaction = len(messages)
                    result = await self.context_summarizer.summarize(
                        messages,
                        self.llm,
                        prior=state.get("context_summary"),
                        prompt_cache_key=self.session_id or None,
                    )
                    if result["view"] is not messages:
                        state["messages"] = result["view"]
                        messages = list(result["view"])
                    state["context_summary"] = {
                        "text": result["text"],
                        "covered_count": result["covered_count"],
                    }
                    # The summary LLM call is real spend: book its usage
                    # like a main-loop call (tracker + instance totals +
                    # state) so cost limits and result totals see it — the
                    # usage used to ride on a discarded response object.
                    summary_usage = result.get("usage")
                    if summary_usage and (
                        summary_usage.get("input_tokens") or summary_usage.get("output_tokens")
                    ):
                        s_input = int(summary_usage.get("input_tokens") or 0)
                        s_output = int(summary_usage.get("output_tokens") or 0)
                        s_cached = int(summary_usage.get("cached_tokens") or 0)
                        # Same pricing attribution as the main loop: the
                        # member that actually served the call (a
                        # FallbackChain records it), else the client's
                        # resolved model, else the graph's label (Codex
                        # PR-55 r3 — summary tokens used to be priced as the
                        # primary provider even when a fallback served them).
                        # Bookkeeping is single-entry (#61): the tracker
                        # record and the audit row move together — the
                        # summarizer call used to be priced but never
                        # audited (session 61279304's 0.16% reconciliation
                        # gap was exactly one such call).
                        summary_model = (
                            getattr(self.llm, "served_model_name", None)
                            or getattr(self.llm, "model_name", None)
                            or self.model_name
                        )
                        summary_cost = book_llm_call(
                            s_input,
                            s_output,
                            tracker=self.cost_tracker,
                            model=summary_model,
                            cached_tokens=s_cached,
                            provider=getattr(self.llm, "served_provider_name", None)
                            or getattr(self.llm, "provider_name", None),
                            session_id=self.session_id,
                            audit_logger=self.audit_logger,
                            agent="summarizer",
                        )
                        totals = self._cost_totals_for(self._thread_for_state(state))
                        totals["cost_usd"] += summary_cost
                        totals["input_tokens"] += s_input
                        totals["output_tokens"] += s_output
                        state["total_cost_usd"] = totals["cost_usd"]
                        state["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
                    if self.event_bus and len(result["view"]) < pre_compaction:
                        # Only announce ACTUAL compaction: when nothing was
                        # newly coverable the view is unchanged and a
                        # "context summarized" event would mislead operators
                        # (and spam the transcript on every step past the
                        # threshold).
                        self.event_bus.emit_message(
                            (
                                f"context summarized: history compacted "
                                f"({result['covered_count']} messages covered by summary)"
                            ),
                            "system",
                        )
                except Exception:
                    logger.debug("Context summarization failed", exc_info=True)

        sys_prompt = self.system_prompt_fn(state)
        # Coerce the runtime's internal message model (AIMessage/ToolMessage/
        # HumanMessage/dict) into genai ChatMessage objects, pulling the system
        # prompt out into the `system=` string. Assistant turns carry their
        # `tool_calls` and tool-result turns carry `tool_response_call_id`, so
        # the provider can pair function_call/function_call_output correctly.
        system, chat_messages = _coerce_chat_messages(messages)
        system = "\n\n".join(part for part in (sys_prompt, system) if part) or sys_prompt

        provider_name = getattr(self.llm, "provider_name", None)
        configured_provider = provider_name
        # Prompt-cache wiring (issue #36): the growing history is the
        # cacheable prefix (single ephemeral breakpoint on the last stable
        # message, applied by the client), routed per-session so
        # OpenAI-style providers keep hitting the same prefix cache. The
        # per-step context note is appended after the breakpoint and never
        # cached. Both are inert on providers without caching.
        context_note = self.dynamic_context_fn(state) if self.dynamic_context_fn else None
        # The session summary rides in the same post-breakpoint tail
        # (issue #38): injecting it into the message history — or worse,
        # the system prompt — would mutate the cacheable prefix on every
        # re-summarization. As a tail note it is byte-stable between
        # re-summarizations, so the prefix stays a cache hit.
        if self.context_summarizer:
            summary_state = state.get("context_summary") or {}
            summary_text = summary_state.get("text") or ""
            if summary_text:
                summary_block = self.context_summarizer.summary_note(summary_text)
                context_note = (
                    "\n\n".join(part for part in (summary_block, context_note) if part)
                    or None
                )
        response = await self.llm.achat_stream(
            messages=chat_messages,
            system=system,
            tools=self.native_tools or None,
            on_text_delta=self.on_text_delta,
            cache_prefix=True,
            prompt_cache_key=self.session_id or None,
            context_note=context_note,
        )
        # A FallbackChain records which member actually served the call;
        # a bare client has no such attribute and the pre-call value stays.
        served_provider = getattr(self.llm, "served_provider_name", None)
        if served_provider:
            provider_name = served_provider
        usage = response.usage
        input_tokens = (usage.prompt_tokens or 0) if usage else 0
        output_tokens = (usage.completion_tokens or 0) if usage else 0
        # Prompt-cache hits: providers report them in prompt_tokens_details;
        # older genai builds and test doubles may not expose the field at all
        # (same getattr discipline as the hunter path).
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        cached_tokens = (getattr(details, "cached_tokens", None) or 0) if details else 0
        assistant_text = response_text(response)
        # tool_calls are raw genai ToolCall objects (.call_id/.fn_name/
        # .fn_arguments). Store them on the AIMessage so the next turn's
        # ChatMessage assistant round-trips them, and so the tool loop can
        # pair each result by call_id.
        tool_calls = list(response.tool_calls)
        # Cost pricing and audit must attribute the call to the model that
        # actually served it. `model_name` is the caller-supplied label
        # (often the webui placeholder); the resolved endpoint model lives
        # on the client and is echoed back on every response.
        served_model = response.provider_model_name
        configured_model = getattr(self.llm, "model_name", None)
        effective_model = served_model or configured_model or self.model_name
        # Pricing key: providers echo canonical/versioned names (e.g.
        # claude-opus-4-7-20260901) that miss the pricing table — charging
        # those at the Sonnet fallback would understate spend whenever the
        # configured name is priced. Metadata and audit keep the served
        # name; only the pricing lookup normalizes.
        pricing_model = effective_model
        if (
            served_model
            and configured_model
            and served_model != configured_model
            and not CostTracker.has_pricing(served_model)
            and CostTracker.has_pricing(configured_model)
            # The configured-name fallback pricing rule exists for versioned
            # echoes of the SAME provider's model (claude-opus-4-7-20260901
            # vs claude-opus-4-7). When a FallbackChain served the call the
            # configured name belongs to a DIFFERENT provider's model —
            # billing it at the primary's rates would silently misstate
            # spend on exactly the failover scenario the chain enables.
            # served_by_primary (not provider-name comparison — two
            # openai_compat members share the same label) is the failover
            # signal; bare clients have no such attribute and default True.
            and getattr(self.llm, "served_by_primary", True)
            and served_provider in (None, configured_provider)
        ):
            pricing_model = configured_model
        ai_message = AIMessage(
            content=assistant_text,
            tool_calls=tool_calls,
            response_metadata={
                "usage": {
                    "input_tokens": (usage.prompt_tokens or 0) if usage else 0,
                    "output_tokens": (usage.completion_tokens or 0) if usage else 0,
                    "total_tokens": (usage.total_tokens or 0) if usage else 0,
                },
                "model": effective_model,
            },
        )
        state.setdefault("messages", []).append(ai_message)

        if input_tokens or output_tokens:
            # Single-entry bookkeeping (#61): the tracker record and the
            # audit row are written together — per-call cost, never the
            # graph's running total (the cumulative value double-counts
            # when audit rows are summed per session, issue #10 live
            # evidence). Pricing normalizes to pricing_model while the
            # audit row keeps the served/configured effective_model.
            call_cost = book_llm_call(
                input_tokens,
                output_tokens,
                tracker=self.cost_tracker,
                model=pricing_model,
                audit_model=effective_model,
                cached_tokens=cached_tokens,
                provider=provider_name,
                session_id=self.session_id,
                audit_logger=self.audit_logger,
            )
            # Instance totals (issues #37/#52): state must report THIS
            # thread's spend on this graph — not the tracker's cross-session
            # running total, and not the other threads sharing the graph.
            totals = self._cost_totals_for(self._thread_for_state(state))
            totals["cost_usd"] += call_cost
            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            state["total_cost_usd"] = totals["cost_usd"]
            state["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]

        if self.event_bus:
            # A 200-char PREVIEW tag: the webui dedups its turn-end inline
            # agent_message against this echo by BYTE-EQUALITY (see
            # _note_delivered_frame in ui/web/app.py). Never change this to
            # full text without revisiting that gate, and never switch the
            # gate to prefix matching — a full-text echo would then swallow
            # the authoritative full-text send.
            self.event_bus.emit_message(assistant_text[:200], "agent")

        if assistant_text:
            found_flags = detect_flags(assistant_text)
            if found_flags:
                existing_flags = list(state.get("flags_found", []))
                state["flags_found"] = existing_flags + found_flags
                if self.event_bus:
                    for found in found_flags:
                        self.event_bus.emit_flag(found["flag"], "LLM response")

        return dict(state)

    def _failure_streak_bound(self) -> int | None:
        raw = None
        if self.agent_limits is not None:
            raw = getattr(self.agent_limits, "identical_failure_streak", None)
        if raw is None:
            raw = _env_int("CLEARWING_IDENTICAL_FAILURE_STREAK", _DEFAULT_FAILURE_STREAK_NUDGE)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            return None
        return raw

    async def _arun_tool_calls(
        self,
        state: dict[str, Any],
        tool_calls: list[Any],
        *,
        resume_decision: object,
    ) -> tuple[list[dict[str, Any]], bool, bool]:
        events: list[dict[str, Any]] = []
        result_messages: list[BaseMessage] = []
        new_flags: list[dict[str, str]] = []
        thread_id = self._find_thread_id_for_state(state)
        streak_bound = self._failure_streak_bound()
        nudge_count = 0
        nudge_tool = ""
        halted = False
        halt_reason = ""

        for index, tool_call in enumerate(tool_calls):
            # tool_call is a genai ToolCall: .call_id / .fn_name / .fn_arguments
            tool_name = str(getattr(tool_call, "fn_name", "") or "")
            tool_call_id = getattr(tool_call, "call_id", None)
            tool_args = getattr(tool_call, "fn_arguments", None)
            if not isinstance(tool_args, dict):
                tool_args = {}
            if self.event_bus:
                self.event_bus.emit(EventType.TOOL_START, {"tool": tool_name, "args": tool_args})

            if self.output_guardrail and tool_name in self.output_guardrail_tool_names:
                command = tool_args.get("command", "")
                result = self.output_guardrail.check_command(command)
                if not result.passed and self.event_bus:
                    self.event_bus.emit_message(f"Guardrail blocked: {result.reason}", "warning")

            tool = self.tools.get(tool_name)
            if tool is None:
                content = json.dumps({"error": f"unknown tool: {tool_name}"})
            else:
                try:
                    content = await self._ainvoke_tool(tool, tool_args, resume_decision)
                except InterruptRequest as exc:
                    if result_messages:  # preserve earlier parallel tool_results so their tool_use blocks aren't orphaned in history
                        state.setdefault("messages", []).extend(result_messages)
                        events.append(dict(state))
                    self._pending[self._find_thread_id_for_state(state)] = _PendingToolResume(
                        tool_calls=tool_calls[index:],
                        prompt=exc.prompt,
                    )
                    # Notify listeners (webui WS/TUI) that a human approval is pending.
                    # Without this emit the graph suspends silently and interactive
                    # clients can never learn to send {"type": "approve", ...}.
                    if self.event_bus:
                        self.event_bus.emit(
                            EventType.APPROVAL_NEEDED,
                            {"prompt": exc.prompt, "tool": tool_name},
                        )
                    return events, True, False
                except Exception as exc:
                    content = json.dumps({"error": str(exc)})

            if not isinstance(content, str):
                content = json.dumps(content)

            # Identical-failure streak guard: nudge at N consecutive identical
            # failing calls, halt the turn at 2N (default N=6, env-overridable).
            if streak_bound:
                key = _streak_key(tool_name, tool_args)
                if _tool_result_failed(content):
                    last_key, count = self._failure_runs.get(thread_id, ("", 0))
                    count = count + 1 if last_key == key else 1
                    if len(self._failure_runs) > 256:
                        self._failure_runs.pop(next(iter(self._failure_runs)))
                    self._failure_runs[thread_id] = (key, count)
                    if count == streak_bound:
                        nudge_count = count
                        nudge_tool = tool_name
                    elif count >= 2 * streak_bound:
                        halted = True
                        halt_reason = (
                            f"Tool '{tool_name}' failed with identical arguments "
                            f"{count} times in a row (last result: {content[:200]})"
                        )
                        logger.warning("agent turn halted: %s", halt_reason)
                        if self.event_bus:
                            self.event_bus.emit(EventType.ERROR, {"message": halt_reason})
                        # Answer the halting call and every remaining call in
                        # the batch, or the next turn 400s on orphaned tool_use.
                        result_messages.append(
                            ToolMessage(
                                content=content,
                                name=tool_name,
                                tool_call_id=tool_call_id,
                            )
                        )
                        result_messages.extend(
                            _synthesize_skipped_tool_results(
                                tool_calls[index + 1 :],
                                "turn halted by identical-failure guard",
                            )
                        )
                        break
                else:
                    self._failure_runs.pop(thread_id, None)

            message = ToolMessage(
                content=content,
                name=tool_name,
                tool_call_id=tool_call_id,
            )
            result_messages.append(message)

            if self.input_guardrail and tool_name in self.input_guardrail_tool_names:
                gr = self.input_guardrail.check(content)
                if not gr.passed and self.event_bus:
                    self.event_bus.emit_message(f"Input guardrail warning: {gr.reason}", "warning")

            if self.episodic_memory:
                target = state.get("target") or "unknown"
                self.episodic_memory.record(
                    target=target,
                    event_type=f"tool:{tool_name}",
                    content=content[:500],
                )

            if self.cost_tracker:
                self.cost_tracker.record_tool_call(tool_name, 0)

            if self.audit_logger:
                self.audit_logger.log_tool_call(
                    tool_name=tool_name, args=tool_args, result=content[:2000]
                )

            if self.knowledge_graph and self.knowledge_graph_populator_fn:
                graph_data = self.knowledge_graph_populator_fn(
                    self.knowledge_graph,
                    tool_name,
                    content,
                    state,
                )
                if graph_data:
                    state["graph_data"] = graph_data

            data = _parse_tool_output(content)
            extra_updates = self.state_updater_fn(tool_name, data, state) or {}
            for key, value in extra_updates.items():
                state[key] = value

            # Flag scan (issue #35): structured tool results embed long hex
            # identifiers (container ids, ...) that must not count as loot.
            # The hex-boundary pattern stops windows inside longer hex runs,
            # and known id fields are masked before scanning the serialized
            # form so a bare 32-hex id value can't pose as an MD5 flag.
            if isinstance(data, (dict, list)):
                scan_text = json.dumps(_strip_id_fields(data), default=str)
            else:
                scan_text = content
            found_flags = detect_flags(scan_text)
            if found_flags:
                new_flags.extend(found_flags)

            if self.event_bus:
                self.event_bus.emit(
                    EventType.TOOL_RESULT,
                    {
                        "tool": tool_name,
                        "content_length": len(content),
                        "flags_found": len(found_flags),
                    },
                )

        state.setdefault("messages", []).extend(result_messages)
        if nudge_count and not halted:
            nudge = HumanMessage(
                content=(
                    f"Automatic guard: tool '{nudge_tool}' has now failed with identical "
                    f"arguments {nudge_count} times in a row. Do not repeat the same call "
                    "unchanged — adjust the approach or give your final answer."
                )
            )
            state["messages"].append(nudge)
            if self.event_bus:
                self.event_bus.emit_message(nudge.content, "warning")
        if halted:
            state["messages"].append(
                HumanMessage(
                    content=(
                        f"Automatic stop: {halt_reason} The turn has been halted to protect "
                        "the budget. Summarize the progress and results collected so far."
                    )
                )
            )
            if self.audit_logger:
                self.audit_logger.log_tool_call(
                    tool_name=tool_name, args=tool_args, result=halt_reason
                )
        if new_flags:
            existing_flags = list(state.get("flags_found", []))
            state["flags_found"] = existing_flags + new_flags
            if self.event_bus:
                self.event_bus.emit(EventType.FLAG_FOUND, {"flags": new_flags})

        events.append(dict(state))
        return events, False, halted

    async def _ainvoke_tool(
        self, tool: AgentTool, arguments: dict[str, Any], resume_decision: object
    ) -> Any:
        # Reject calls missing required inputs up front, naming the fields
        # so the model can retry with them (instead of an opaque TypeError).
        missing = tool.missing_required_arguments(arguments)
        if missing:
            raise ValueError(f"missing required argument(s): {', '.join(missing)}")
        with tool_execution_context(resume_decision=resume_decision):
            if asyncio.iscoroutinefunction(tool.func):
                return await tool.func(**arguments)
            return await asyncio.to_thread(tool.func, **arguments)

    def _merge_input(self, state: dict[str, Any], input_data: dict[str, Any]) -> None:
        for key, value in input_data.items():
            if key == "messages":
                state.setdefault("messages", []).extend(value)
            elif key == "context_summary":
                # Runtime-owned key: this is the committed compaction state
                # (issue #38). An input frame carrying it — e.g. a replayed
                # start frame — would overwrite the live summary state and
                # make the runtime believe already-dropped history is still
                # covered, silently losing context. Input may not set it.
                continue
            else:
                state[key] = value

    def _get_or_create_state(self, thread_id: str) -> dict[str, Any]:
        return self._state.setdefault(thread_id, {"messages": []})

    def _thread_id(self, config: dict) -> str:
        return config.get("configurable", {}).get("thread_id", "default")

    def _find_thread_id_for_state(self, state: dict[str, Any]) -> str:
        for thread_id, existing_state in self._state.items():
            if existing_state is state:
                return thread_id
        raise KeyError("state not registered")


def populate_knowledge_graph(
    kg: Any, tool_name: str, content: str, state: dict[str, Any]
) -> dict[str, Any]:
    target = state.get("target", "")
    if not target:
        return {}

    try:
        kg.add_target(target)
        data = _parse_tool_output(content)

        if tool_name == "scan_ports" and isinstance(data, list):
            for port_info in data:
                port = port_info.get("port")
                proto = port_info.get("protocol", "tcp")
                if port:
                    kg.add_port(target, port, proto)

        elif tool_name == "detect_services" and isinstance(data, list):
            for svc_info in data:
                port = svc_info.get("port")
                proto = svc_info.get("protocol", "tcp")
                service = svc_info.get("service", "unknown")
                version = svc_info.get("version", "")
                if port and service:
                    port_id = f"{target}:{port}/{proto}"
                    kg.add_port(target, port, proto)
                    kg.add_service(port_id, service, version)

        elif tool_name == "scan_vulnerabilities" and isinstance(data, list):
            for vuln in data:
                cve = vuln.get("cve", "")
                cvss = vuln.get("cvss", 0.0)
                port = vuln.get("port")
                service = vuln.get("service", "unknown")
                if cve:
                    service_id = f"{target}:{port}/tcp:{service}" if port else service
                    kg.add_vulnerability(service_id, cve, cvss)

        elif tool_name == "detect_os" and isinstance(data, str):
            kg.add_target(target, os=data)

        elif tool_name == "exploit_vulnerability" and isinstance(data, dict):
            cve = data.get("cve", "unknown")
            success = data.get("success", False)
            exploit = data.get("exploit", "unknown")
            kg.add_exploit_result(cve, exploit, success=success)

        # v0.4: SRP tools
        elif tool_name == "srp_handshake" and isinstance(data, dict):
            kg.add_protocol("SRP-6a")
            server_params = data.get("server_params", {})
            algo = server_params.get("algorithm", "")
            iterations = server_params.get("iterations", 0)
            if algo:
                kg.add_algorithm(algo)
                kg.add_relationship("protocol:SRP-6a", f"algorithm:{algo}", "USES_ALGORITHM")
            if iterations and target:
                kg.add_kdf_config(algo or "PBKDF2-HMAC-SHA256", iterations, target)
            skd = data.get("2skd")
            if isinstance(skd, dict):
                kg.add_key_material("auk", target)
                kg.add_key_material("srp_x", target)
                kg.add_relationship("protocol:SRP-6a", "key:srp_x:" + target, "AUTHENTICATES_WITH")
                if iterations:
                    kdf_id = f"kdf:{algo or 'PBKDF2-HMAC-SHA256'}:{iterations}:{target}"
                    kg.add_relationship(kdf_id, "key:auk:" + target, "DERIVES_KEY")
                    kg.add_relationship(kdf_id, "key:srp_x:" + target, "DERIVES_KEY")

        elif tool_name == "srp_extract_verifier_info" and isinstance(data, dict):
            kg.add_protocol("SRP-6a")
            valid_user = data.get("valid_user", {})
            algo = valid_user.get("algorithm", "")
            iterations = valid_user.get("iterations", 0)
            if algo:
                kg.add_algorithm(algo)
                kg.add_relationship("protocol:SRP-6a", f"algorithm:{algo}", "USES_ALGORITHM")
            if iterations and algo and target:
                kg.add_kdf_config(algo, iterations, target)

        elif tool_name == "srp_fuzz_parameters" and isinstance(data, dict):
            kg.add_protocol("SRP-6a")
            for vuln in data.get("vulnerabilities", []):
                desc = vuln.get("description", "SRP parameter validation bypass")
                eid = f"vuln:srp_fuzz:{vuln.get('vector', 'unknown')}"
                kg.add_entity("exploit", eid, description=desc)
                kg.add_relationship("protocol:SRP-6a", eid, "VULNERABLE_TO")

        elif tool_name == "srp_timing_attack" and isinstance(data, dict):
            kg.add_protocol("SRP-6a")
            if data.get("significant"):
                desc = data.get("conclusion", "Timing side-channel in SRP authentication")
                eid = f"vuln:srp_timing:{data.get('test_type', 'unknown')}"
                kg.add_entity("exploit", eid, description=desc)
                kg.add_relationship("protocol:SRP-6a", eid, "VULNERABLE_TO")

        # v0.4: KDF tools
        elif tool_name == "analyze_kdf_parameters" and isinstance(data, dict):
            algo = data.get("algorithm", "")
            iterations = data.get("iterations", 0)
            if algo:
                kg.add_algorithm(algo)
            if algo and iterations and target:
                kg.add_kdf_config(
                    algo, iterations, target,
                    risk_level=data.get("risk_level", ""),
                    iterations_compliant=data.get("iterations_compliant"),
                )

        elif tool_name == "benchmark_kdf_cracking" and isinstance(data, dict):
            algo = data.get("algorithm", "")
            iterations = data.get("iterations", 0)
            if algo and iterations and target:
                entity = kg.add_kdf_config(algo, iterations, target)
                assessment = data.get("assessment", "")
                if assessment:
                    entity.properties["cracking_assessment"] = assessment

        elif tool_name == "test_2skd_implementation" and isinstance(data, dict):
            server_params = data.get("server_params", {})
            algo = server_params.get("algorithm", "PBKDF2-HMAC-SHA256")
            iterations = server_params.get("iterations", 0)
            if target:
                kg.add_key_material("auk", target)
                kg.add_key_material("srp_x", target)
            if algo and iterations and target:
                kdf_id = f"kdf:{algo}:{iterations}:{target}"
                kg.add_kdf_config(algo, iterations, target)
                kg.add_relationship(kdf_id, "key:auk:" + target, "DERIVES_KEY")
                kg.add_relationship(kdf_id, "key:srp_x:" + target, "DERIVES_KEY")

        elif tool_name == "kdf_oracle_test" and isinstance(data, dict):
            if data.get("oracle_detected"):
                oracle_types = data.get("oracle_type", [])
                desc = data.get("conclusion", "KDF oracle detected")
                eid = f"vuln:kdf_oracle:{':'.join(oracle_types)}"
                kg.add_entity("exploit", eid, description=desc)
                if target:
                    kg.add_relationship(target, eid, "VULNERABLE_TO")

        # v0.4: Vault tools
        elif tool_name == "parse_vault_blob" and isinstance(data, dict):
            algo = data.get("algorithm", "")
            enc = data.get("encryption", "")
            km = data.get("key_management", "")
            if enc:
                kg.add_algorithm(enc)
            if km:
                kg.add_algorithm(km)
            if algo and algo != "unknown":
                kg.add_algorithm(algo)

        elif tool_name == "analyze_key_hierarchy" and isinstance(data, dict):
            for step in data.get("key_chain", []):
                algo = step.get("algorithm", "")
                if algo:
                    kg.add_algorithm(algo)
            for algo in data.get("wrapping_algorithms", []):
                if algo:
                    kg.add_algorithm(algo)
            for algo in data.get("derivation_algorithms", []):
                if algo:
                    kg.add_algorithm(algo)
            if data.get("extractable_keys") and target:
                for ek in data["extractable_keys"]:
                    algo = ek.get("algorithm", "unknown")
                    km = kg.add_key_material(f"extractable_{ek.get('step', 0)}", target, extractable=True)
                    if algo:
                        kg.add_algorithm(algo)

        elif tool_name == "test_aead_integrity" and isinstance(data, dict):
            for vuln in data.get("vulnerabilities", []):
                mod = vuln.get("modification", "unknown")
                desc = vuln.get("description", f"AEAD {mod} bypass")
                eid = f"vuln:aead:{mod}"
                kg.add_entity("exploit", eid, description=desc)
                enc_algo = data.get("original_blob_format", "")
                if enc_algo:
                    kg.add_relationship(f"algorithm:{enc_algo}", eid, "VULNERABLE_TO")

        elif tool_name == "key_wrap_analysis" and isinstance(data, dict):
            for algo_info in data.get("algorithm_analysis", []):
                algo = algo_info.get("algorithm", "")
                if algo:
                    kg.add_algorithm(algo)

        # v0.4: Credential tools
        elif tool_name == "analyze_2skd_entropy" and isinstance(data, dict):
            algo = data.get("algorithm", "")
            iterations = data.get("iterations", 0)
            if algo:
                kg.add_algorithm(algo)
            if algo and iterations and target:
                entity = kg.add_kdf_config(algo, iterations, target)
                assessment = data.get("assessment", "")
                if assessment:
                    entity.properties["2skd_assessment"] = assessment
                entity.properties["combined_entropy_bits"] = data.get("combined_entropy_bits", 0)

        elif tool_name == "test_secret_key_validation" and isinstance(data, dict):
            if data.get("factor_separation"):
                signals = data.get("separation_signals", [])
                desc = data.get("conclusion", "2SKD factor separation detected")
                eid = f"vuln:2skd_factor_separation:{':'.join(signals)}"
                kg.add_entity("exploit", eid, description=desc)
                if target:
                    kg.add_relationship(target, eid, "VULNERABLE_TO")

        elif tool_name == "enumerate_secret_key_format" and isinstance(data, dict):
            fmt = data.get("format_analysis", {})
            entropy = fmt.get("total_entropy_bits", 0)
            if target and entropy:
                km = kg.add_key_material("secret_key", target, entropy_bits=entropy)
                risks = data.get("predictability_risks", [])
                if risks:
                    km.properties["predictability_risks"] = risks

        elif tool_name == "offline_crack_setup" and isinstance(data, dict):
            algo = data.get("algorithm", "")
            iterations = data.get("iterations", 0)
            if algo and iterations and target:
                entity = kg.add_kdf_config(algo, iterations, target)
                feasibility = data.get("feasibility", "")
                if feasibility:
                    entity.properties["cracking_feasibility"] = feasibility

        # v0.4: Mycelium tools
        elif tool_name == "mycelium_create_channel" and isinstance(data, dict):
            kg.add_protocol("Mycelium")
            ch_type = data.get("channel_type", "u")
            ch_uuid = data.get("channel_uuid", "")
            if ch_uuid and target:
                eid = f"channel:{ch_type}:{ch_uuid[:8]}"
                kg.add_entity("channel", eid, channel_type=ch_type, target=target)
                kg.add_relationship(target, eid, "HAS_CHANNEL")

        elif tool_name == "mycelium_fuzz_auth" and isinstance(data, dict):
            kg.add_protocol("Mycelium")
            for bypass in data.get("bypasses", []):
                desc = bypass.get("description", "Mycelium auth bypass")
                eid = f"vuln:mycelium_auth:{bypass.get('vector', 'unknown')}"
                kg.add_entity("exploit", eid, description=desc)
                if target:
                    kg.add_relationship(target, eid, "VULNERABLE_TO")

        elif tool_name == "mycelium_test_race" and isinstance(data, dict):
            kg.add_protocol("Mycelium")
            if data.get("successful_writes", 0) > 1 or data.get("successful_reads", 0) > 0:
                eid = "vuln:mycelium_race_condition"
                desc = "; ".join(data.get("findings", []))
                kg.add_entity("exploit", eid, description=desc)
                if target:
                    kg.add_relationship(target, eid, "VULNERABLE_TO")

        # v0.4: Recovery tools
        elif tool_name == "test_recovery_acceptance" and isinstance(data, dict):
            if data.get("accepted_count", 0) > 0:
                eid = "vuln:recovery_code_acceptance"
                desc = f"{data['accepted_count']} recovery code(s) accepted"
                kg.add_entity("exploit", eid, description=desc)
                if target:
                    kg.add_relationship(target, eid, "VULNERABLE_TO")

        elif tool_name == "analyze_recovery_entropy" and isinstance(data, dict):
            bits = data.get("total_entropy_bits", 0)
            if bits and target:
                km = kg.add_key_material("recovery_code", target, entropy_bits=bits)
                km.properties["assessment"] = data.get("assessment", "")

        # v0.4: Session tools
        elif tool_name == "replay_with_mutations" and isinstance(data, dict):
            for finding in data.get("findings", []):
                if "WARNING" in finding:
                    eid = "vuln:weak_token_validation"
                    kg.add_entity("exploit", eid, description=finding)
                    if target:
                        kg.add_relationship(target, eid, "VULNERABLE_TO")

        elif tool_name == "test_session_fixation" and isinstance(data, dict):
            if data.get("fixation_risk"):
                eid = "vuln:session_fixation"
                cookies = data.get("session_like_unchanged", [])
                desc = f"Session fixation risk: cookies unchanged after auth: {', '.join(cookies)}"
                kg.add_entity("exploit", eid, description=desc)
                if target:
                    kg.add_relationship(target, eid, "VULNERABLE_TO")

        # v0.4: Bundle tools
        elif tool_name == "search_bundle_patterns" and isinstance(data, dict):
            for match in data.get("matches", []):
                pat = match.get("pattern", "")
                if pat in ("hardcoded_secret", "private_key", "aws_key", "flag_format"):
                    eid = f"vuln:bundle_leak:{pat}"
                    kg.add_entity("exploit", eid, description=f"JS bundle contains {pat}: {match.get('match', '')[:100]}")
                    if target:
                        kg.add_relationship(target, eid, "VULNERABLE_TO")

        elif tool_name == "extract_api_routes" and isinstance(data, dict):
            for route in data.get("routes", []):
                path = route.get("path", "")
                if path and target:
                    eid = f"endpoint:{target}:{path}"
                    kg.add_entity("endpoint", eid, path=path, methods=route.get("methods", []))

        # v0.4: CC tools
        elif tool_name == "cc_discover_schema" and isinstance(data, dict):
            if data.get("schema_complete"):
                eid = f"endpoint:{target}:{data.get('endpoint', '/cc')}"
                kg.add_entity("endpoint", eid, schema=data.get("discovered_fields", {}))
            for field_info in data.get("discovered_fields", {}).values():
                if field_info.get("type") == "uuid":
                    kg.add_entity("parameter", f"param:{field_info.get('value', '')}", type="uuid")

        elif tool_name == "cc_fuzz_fields" and isinstance(data, dict):
            for finding in data.get("interesting_findings", []):
                if finding.get("severity") == "HIGH":
                    eid = f"vuln:cc_field:{finding.get('field', 'unknown')}"
                    kg.add_entity("exploit", eid, description=finding.get("description", ""))
                    if target:
                        kg.add_relationship(target, eid, "VULNERABLE_TO")

        kg.save()
        return nx.node_link_data(kg._graph)
    except Exception:
        logger.debug("Knowledge graph population failed", exc_info=True)
        return {}
