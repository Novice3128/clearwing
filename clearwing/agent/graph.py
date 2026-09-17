from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

from clearwing.agent.runtime import NativeAgentGraph, populate_knowledge_graph
from clearwing.agent.state import AgentState
from clearwing.agent.tooling import ensure_agent_tool
from clearwing.capabilities import capabilities
from clearwing.llm.fallback import FallbackChain
from clearwing.llm.native import AsyncLLMClient
from clearwing.providers import ProviderManager, resolve_llm_endpoint
from clearwing.providers.binding import AgentLimits
from clearwing.providers.env import DEFAULT_ANTHROPIC_MODEL, resolve_fallback_endpoints

from .prompts import build_dynamic_context, build_system_prompt
from .tools import get_all_tools, get_custom_tools

logger = logging.getLogger(__name__)


def _default_agent_limits() -> AgentLimits:
    """Env-overridable loop bounds for entry points without a provider profile.

    Without these a WebUI/CLI session can loop unbounded (session 932ff8ec
    repeated one failing tool call 293 times, ~$249). The defaults are
    generous for legitimate tasks; set CLEARWING_MAX_STEPS /
    CLEARWING_MAX_TOOL_CALLS to override (values <= 0 mean unbounded).
    """

    def _env_int(name: str, default: int) -> int | None:
        raw = (os.environ.get(name) or "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return value if value > 0 else None

    return AgentLimits(
        max_steps=_env_int("CLEARWING_MAX_STEPS", 100),
        max_tool_calls=_env_int("CLEARWING_MAX_TOOL_CALLS", 400),
    )


def _default_pentest_state_updater(tool_name: str, data: Any, state: dict) -> dict:
    if tool_name == "scan_ports" and isinstance(data, list):
        return {"open_ports": state.get("open_ports", []) + data}
    if tool_name == "detect_services" and isinstance(data, list):
        return {"services": state.get("services", []) + data}
    if tool_name == "scan_vulnerabilities" and isinstance(data, list):
        return {"vulnerabilities": state.get("vulnerabilities", []) + data}
    if tool_name == "detect_os" and isinstance(data, str):
        return {"os_info": data}
    if tool_name == "exploit_vulnerability" and isinstance(data, dict):
        return {"exploit_results": state.get("exploit_results", []) + [data]}
    if tool_name == "kali_setup" and isinstance(data, str):
        return {"kali_container_id": data}
    return {}


_DEFAULT_PENTEST_GUARDRAIL_TOOLS = frozenset(
    {
        "scan_ports",
        "detect_services",
        "scan_vulnerabilities",
        "detect_os",
    }
)

_DEFAULT_OUTPUT_GUARDRAIL_TOOLS = frozenset({"kali_execute"})


def build_react_graph(
    llm_with_tools: AsyncLLMClient,
    tools: list,
    system_prompt_fn,
    *,
    state_schema=AgentState,
    model_name: str = "claude-sonnet-4-6",
    session_id: str = None,
    state_updater_fn=None,
    knowledge_graph_populator_fn=None,
    input_guardrail_tool_names=None,
    output_guardrail_tool_names=None,
    enable_cost_tracker: bool = True,
    enable_episodic_memory: bool = True,
    enable_audit: bool = True,
    enable_knowledge_graph: bool = True,
    enable_input_guardrail: bool = True,
    enable_output_guardrail: bool = True,
    enable_event_bus: bool = True,
    enable_context_summarizer: bool = True,
    agent_limits=None,
    dynamic_context_fn=None,
):
    del state_schema
    if state_updater_fn is None:
        state_updater_fn = _default_pentest_state_updater
    if knowledge_graph_populator_fn is None:
        knowledge_graph_populator_fn = populate_knowledge_graph
    if input_guardrail_tool_names is None:
        input_guardrail_tool_names = _DEFAULT_PENTEST_GUARDRAIL_TOOLS
    if output_guardrail_tool_names is None:
        output_guardrail_tool_names = _DEFAULT_OUTPUT_GUARDRAIL_TOOLS

    # `llm_with_tools` is now a bare AsyncLLMClient (no ChatModel `bind_tools`
    # facade). The native client threads the tool list through each `achat`
    # call rather than binding it, so build the NativeToolSpec list here once
    # and hand both the client and the specs to the runtime.
    native_tools = [ensure_agent_tool(t) for t in tools]

    return NativeAgentGraph(
        llm=llm_with_tools,
        native_tools=native_tools,
        tools=tools,
        system_prompt_fn=system_prompt_fn,
        model_name=model_name,
        session_id=session_id,
        state_updater_fn=state_updater_fn,
        knowledge_graph_populator_fn=knowledge_graph_populator_fn,
        input_guardrail_tool_names=input_guardrail_tool_names,
        output_guardrail_tool_names=output_guardrail_tool_names,
        enable_cost_tracker=enable_cost_tracker,
        enable_episodic_memory=enable_episodic_memory,
        enable_audit=enable_audit,
        enable_knowledge_graph=enable_knowledge_graph and capabilities.has("knowledge"),
        enable_input_guardrail=enable_input_guardrail,
        enable_output_guardrail=enable_output_guardrail,
        enable_event_bus=enable_event_bus,
        enable_context_summarizer=enable_context_summarizer,
        agent_limits=agent_limits,
        dynamic_context_fn=dynamic_context_fn,
    )


def _maybe_wrap_fallback_chain(
    primary: AsyncLLMClient,
    *,
    cli_base_url: str | None,
    cli_api_key: str | None,
) -> Any:
    """Wrap *primary* in a FallbackChain when config.yaml declares one.

    Gated on per-request credentials (issue #17): an explicit base_url or
    api_key from a webui start frame / CLI flag means the operator chose
    THIS endpoint for the session — the conversation must never be silently
    re-routed to a different provider. Env/config-tier deployments (the
    compose stack) keep the chain.

    Failures building an individual fallback client only drop that entry;
    the primary alone is still a valid (chain-less) result.
    """
    if cli_base_url or cli_api_key:
        return primary
    endpoints = resolve_fallback_endpoints()
    if not endpoints:
        return primary
    fallbacks: list[AsyncLLMClient] = []
    for endpoint in endpoints:
        try:
            fallbacks.append(
                ProviderManager.for_endpoint(endpoint).get_native_client("default")
            )
        except Exception:
            logger.warning(
                "Failed to build fallback client for %s; skipping it",
                endpoint.describe(),
                exc_info=True,
            )
    if not fallbacks:
        return primary
    logger.info(
        "LLM fallback chain active: %s -> %s",
        primary.model_name,
        " -> ".join(client.model_name for client in fallbacks),
    )
    return FallbackChain(primary, fallbacks)


def _create_llm(
    model_name: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
    provider_manager: ProviderManager | None = None,
    task: str = "default",
    model_explicit: bool = False,
) -> AsyncLLMClient:
    if provider_manager is not None:
        # model_explicit does not apply here: the manager owns per-task
        # endpoint resolution and its client's own model_name is what the
        # runtime attributes cost/audit against.
        return provider_manager.get_native_client(task)
    if base_url or api_key:
        # Explicit per-request credentials win for THEIR fields only
        # (issue #16): an unset or placeholder model still defers to
        # env/config instead of being guessed from the base_url hostname.
        # config_provider is deliberately not forced to {} — config.yaml
        # fills whatever the frame left blank (per-field merge in
        # resolve_llm_endpoint).
        effective_model = (
            model_name
            if model_name
            and (model_explicit or model_name != DEFAULT_ANTHROPIC_MODEL)
            else None
        )
        endpoint = resolve_llm_endpoint(
            cli_model=effective_model,
            cli_base_url=base_url,
            cli_api_key=api_key,
        )
        return ProviderManager.for_endpoint(endpoint).get_native_client("default")
    # No per-request credentials: resolve from env / config.yaml instead.
    # The webui start frame always carries a model name; passing it as a
    # bare cli_model used to take the "CLI flags win" branch, which never
    # consulted config.yaml / env — every chat session then died with
    # "no API key or base URL configured".
    #
    # model_explicit disambiguates "the user typed this model" from "the
    # entry point's placeholder": an explicit choice must win even when it
    # happens to equal DEFAULT_ANTHROPIC_MODEL (issue #22), while a
    # placeholder/None defers to the configured provider model.
    endpoint = resolve_llm_endpoint()
    if model_name and (
        model_explicit
        or endpoint.source == "default"
        or model_name != DEFAULT_ANTHROPIC_MODEL
    ):
        endpoint = dataclasses.replace(endpoint, model=model_name)
    primary = ProviderManager.for_endpoint(endpoint).get_native_client("default")
    return _maybe_wrap_fallback_chain(
        primary, cli_base_url=base_url, cli_api_key=api_key
    )


def create_agent(
    model_name: str | None = "claude-sonnet-4-6",
    custom_tools: list = None,
    session_id: str = None,
    base_url: str = None,
    api_key: str = None,
    provider_manager: ProviderManager | None = None,
    model_explicit: bool = False,
):
    all_tools = get_all_tools()
    if custom_tools:
        all_tools.extend(custom_tools)

    runtime_tools = get_custom_tools()
    for rt in runtime_tools:
        if rt not in all_tools:
            all_tools.append(rt)

    # No `bind_tools` step: the native client takes the tool list per-call.
    # `build_react_graph` builds the NativeToolSpec list from `all_tools`.
    agent_limits = None
    if provider_manager is None:
        llm = _create_llm(
            model_name, base_url=base_url, api_key=api_key, model_explicit=model_explicit
        )
        agent_limits = _default_agent_limits()
    else:
        llm = _create_llm(
            model_name,
            base_url=base_url,
            api_key=api_key,
            provider_manager=provider_manager,
            model_explicit=model_explicit,
        )
        agent_limits = provider_manager.get_agent_limits("default")

    # The graph's model_name is a display/audit fallback label; cost pricing
    # and audit use the client's resolved model (see _aassistant_step).
    return build_react_graph(
        llm_with_tools=llm,
        tools=all_tools,
        system_prompt_fn=build_system_prompt,
        dynamic_context_fn=build_dynamic_context,
        state_schema=AgentState,
        model_name=model_name or DEFAULT_ANTHROPIC_MODEL,
        session_id=session_id,
        agent_limits=agent_limits,
    )
