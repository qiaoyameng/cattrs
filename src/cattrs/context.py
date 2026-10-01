from __future__ import annotations

import inspect
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

__all__ = [
    "INHERIT_CONTEXT",
    "adapt_structure_hook",
    "adapt_unstructure_hook",
    "current_context",
]


_CONTEXT_MARKER = "_cattrs_context_hook"

current_context: ContextVar[Any] = ContextVar("cattrs_context", default=None)
INHERIT_CONTEXT: Any = object()


def _context_parameter(
    hook: Callable[..., Any], positional_args: int
) -> inspect.Parameter | None:
    if getattr(hook, _CONTEXT_MARKER, False):
        return inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    try:
        sig = inspect.signature(hook)
    except (TypeError, ValueError):
        return None

    positional = [
        parameter
        for parameter in sig.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if (
        len(positional) > positional_args
        and positional[positional_args].name == "context"
    ):
        return positional[positional_args]
    context_parameter = sig.parameters.get("context")
    if (
        context_parameter is not None
        and context_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ):
        return context_parameter
    return None


def adapt_unstructure_hook(hook: Callable[..., Any]) -> Callable[..., Any]:
    """Return a hook which resolves the conversion context when called."""
    if getattr(hook, _CONTEXT_MARKER, False):
        return hook
    context_parameter = _context_parameter(hook, 1)
    if context_parameter is None:
        return hook
    pass_as_keyword = context_parameter.kind is inspect.Parameter.KEYWORD_ONLY

    @wraps(hook)
    def unstructure_with_context(value: Any, context: Any = INHERIT_CONTEXT) -> Any:
        if context is INHERIT_CONTEXT:
            context = current_context.get()
        if pass_as_keyword:
            return hook(value, context=context)
        return hook(value, context)

    setattr(unstructure_with_context, _CONTEXT_MARKER, True)
    return unstructure_with_context


def adapt_structure_hook(hook: Callable[..., Any]) -> Callable[..., Any]:
    """Return a hook which resolves the conversion context when called."""
    if getattr(hook, _CONTEXT_MARKER, False):
        return hook
    context_parameter = _context_parameter(hook, 2)
    if context_parameter is None:
        return hook
    pass_as_keyword = context_parameter.kind is inspect.Parameter.KEYWORD_ONLY

    @wraps(hook)
    def structure_with_context(
        value: Any, type_: Any, context: Any = INHERIT_CONTEXT
    ) -> Any:
        if context is INHERIT_CONTEXT:
            context = current_context.get()
        if pass_as_keyword:
            return hook(value, type_, context=context)
        return hook(value, type_, context)

    setattr(structure_with_context, _CONTEXT_MARKER, True)
    return structure_with_context
