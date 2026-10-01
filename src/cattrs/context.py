from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from inspect import Parameter, signature
from typing import Any

from .fns import identity

__all__ = [
    "Context",
    "context_aware_structure_hook",
    "context_aware_unstructure_hook",
]

#: Per-conversion contextual data. Its shape is intentionally user-defined.
Context = Any

_UNSTRUCTURE_PASSTHROUGHS = (dict,)


def _accepts_context(func: Callable[..., Any], required_positional: int) -> bool:
    try:
        params = signature(func).parameters
    except (TypeError, ValueError, RuntimeError):
        return False

    if "context" in params:
        return True
    return False


def context_aware_unstructure_hook(
    hook: Callable[..., Any],
) -> Callable[..., Any]:
    if getattr(hook, "__cattrs_context_aware__", False) or _accepts_context(hook, 1):
        return hook

    @wraps(hook)
    def unstructure_with_context(value: Any, *, context: Context = None) -> Any:
        return hook(value)

    unstructure_with_context.__cattrs_context_aware__ = True  # type: ignore[attr-defined]
    return unstructure_with_context


def is_unstructure_passthrough(hook: Callable[..., Any]) -> bool:
    """Whether the hook can be called without a context argument."""
    return hook is identity or hook in _UNSTRUCTURE_PASSTHROUGHS


def context_aware_structure_hook(
    hook: Callable[..., Any],
) -> Callable[..., Any]:
    if getattr(hook, "__cattrs_context_aware__", False) or _accepts_context(hook, 2):
        return hook

    try:
        params = signature(hook).parameters
    except (TypeError, ValueError, RuntimeError):
        params = {}

    positional_count = sum(
        p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        for p in params.values()
    )

    @wraps(hook)
    def structure_with_context(
        value: Any, type_: Any = None, *, context: Context = None
    ) -> Any:
        return hook(value) if positional_count < 2 else hook(value, type_)

    structure_with_context.__cattrs_context_aware__ = True  # type: ignore[attr-defined]
    return structure_with_context
