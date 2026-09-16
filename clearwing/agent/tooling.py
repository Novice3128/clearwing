from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from clearwing.llm.native import NativeToolSpec, ToolInputModel

_TOOL_ACTIVE: ContextVar[bool] = ContextVar("_TOOL_ACTIVE", default=False)
_TOOL_RESUME_DECISION: ContextVar[object] = ContextVar("_TOOL_RESUME_DECISION", default=Ellipsis)
# Ambient session attribution (issue #41): the id of the interactive session
# whose turn is currently executing in this context. Tools spawned by the
# agent (e.g. sourcehunt hunts) read it to attribute their LLM spend to the
# invoking session instead of leaking into whatever consumer happens to be
# listening on the process-wide EventBus. Propagates through asyncio tasks
# and asyncio.to_thread (both copy the context).
_SESSION_ID: ContextVar[str | None] = ContextVar("_SESSION_ID", default=None)


def current_session_id() -> str | None:
    """The ambient session id for cost/event attribution, or None.

    Set by the webui around each agent turn and by operator jobs around
    ``arun``; nested consumers (hunt tool calls run via ``asyncio.to_thread``)
    inherit it automatically.
    """
    return _SESSION_ID.get()


@contextmanager
def session_scope(session_id: str | None):
    """Bind *session_id* as the ambient attribution id within the block."""
    token = _SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _SESSION_ID.reset(token)


class InterruptRequest(RuntimeError):
    def __init__(self, prompt: str) -> None:
        super().__init__(prompt)
        self.prompt = prompt


def interrupt(prompt: str) -> bool:
    if not _TOOL_ACTIVE.get():
        return False
    decision = _TOOL_RESUME_DECISION.get()
    if decision is Ellipsis:
        raise InterruptRequest(prompt)
    return bool(decision)


@contextmanager
def tool_execution_context(*, resume_decision: object = Ellipsis):
    token_active = _TOOL_ACTIVE.set(True)
    token_decision = _TOOL_RESUME_DECISION.set(resume_decision)
    try:
        yield
    finally:
        _TOOL_RESUME_DECISION.reset(token_decision)
        _TOOL_ACTIVE.reset(token_active)


@dataclass
class AgentTool:
    func: Callable[..., Any]
    name: str
    description: str
    input_model: type[BaseModel]
    input_schema: dict[str, Any] = field(init=False)
    args_schema: type[BaseModel] = field(init=False)

    def __post_init__(self) -> None:
        self.args_schema = self.input_model
        self.input_schema = self.input_model.model_json_schema()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    async def ainvoke(self, arguments: dict[str, Any] | None = None, /, **kwargs: Any) -> Any:
        params = _normalize_arguments(arguments, kwargs)
        # Inherit the enclosing resume decision: a nested tool invoked by an
        # already-approved parent must not raise a fresh InterruptRequest
        # (which would reset the approval and loop forever on resume).
        with tool_execution_context(resume_decision=_TOOL_RESUME_DECISION.get()):
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**params)
            return await asyncio.to_thread(self.func, **params)

    def invoke(self, arguments: dict[str, Any] | None = None, /, **kwargs: Any) -> Any:
        params = _normalize_arguments(arguments, kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(params))
        return self.ainvoke(params)

    def missing_required_arguments(self, arguments: dict[str, Any] | None) -> list[str]:
        """Names of required parameters absent from *arguments*.

        Extra keys are not flagged: tools may tolerate superfluous LLM
        arguments, but a call missing required inputs fails with an opaque
        ``TypeError`` (issue #9: ``save_report() missing 3 required
        positional arguments``) — naming the fields lets the model self-
        correct on the next attempt.
        """
        params = dict(arguments or {})
        return [
            name
            for name, field in self.input_model.model_fields.items()
            if field.is_required() and name not in params
        ]

    def with_resume_decision(self, decision: bool) -> Callable[[dict[str, Any]], Any]:
        async def runner(arguments: dict[str, Any]) -> Any:
            with tool_execution_context(resume_decision=decision):
                if inspect.iscoroutinefunction(self.func):
                    return await self.func(**arguments)
                return await asyncio.to_thread(self.func, **arguments)

        return runner


def tool(func: Callable[..., Any] | None = None, **decorator_kwargs: Any):
    def wrap(fn: Callable[..., Any]) -> AgentTool:
        name = decorator_kwargs.get("name") or fn.__name__
        description = decorator_kwargs.get("description") or inspect.getdoc(fn) or ""
        return AgentTool(
            func=fn,
            name=name,
            description=description,
            input_model=_build_input_model(fn),
        )

    if func is None:
        return wrap
    return wrap(func)


def as_native_tool_spec(tool_obj: Any) -> NativeToolSpec:
    if isinstance(tool_obj, NativeToolSpec):
        return tool_obj
    if isinstance(tool_obj, AgentTool):
        return NativeToolSpec(
            name=tool_obj.name,
            description=tool_obj.description,
            schema=tool_obj.input_schema,
            handler=tool_obj.func,
        )
    if all(hasattr(tool_obj, attr) for attr in ("name", "description")):
        schema = getattr(tool_obj, "input_schema", None)
        if not isinstance(schema, dict):
            args_schema = getattr(tool_obj, "args_schema", None)
            if args_schema is not None and hasattr(args_schema, "model_json_schema"):
                schema = args_schema.model_json_schema()
            elif args_schema is not None and hasattr(args_schema, "schema"):
                schema = args_schema.schema()
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        handler = getattr(tool_obj, "func", None) or getattr(tool_obj, "invoke", None) or tool_obj
        return NativeToolSpec(
            name=str(tool_obj.name),
            description=str(getattr(tool_obj, "description", "") or ""),
            schema=schema,
            handler=handler,
        )
    raise TypeError(f"Unsupported tool object: {tool_obj!r}")


def ensure_agent_tool(tool_obj: Any) -> NativeToolSpec:
    return as_native_tool_spec(tool_obj)


def _normalize_arguments(
    arguments: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    params = dict(arguments or {})
    params.update(kwargs)
    return params


def _build_input_model(fn: Callable[..., Any]) -> type[BaseModel]:
    signature = inspect.signature(fn)
    annotations = get_type_hints(fn)
    fields: dict[str, tuple[Any, Any]] = {}

    for name, parameter in signature.parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        annotation = annotations.get(name, Any)
        default = ... if parameter.default is inspect._empty else parameter.default
        fields[name] = (annotation, default)

    model_name = "".join(part.capitalize() for part in fn.__name__.split("_")) + "Input"
    return create_model(model_name, __base__=ToolInputModel, **fields)
