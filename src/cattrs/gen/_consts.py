from __future__ import annotations

from threading import local
from typing import Any, Callable

from attrs import frozen

from ..dispatch import ContextualStructureHook, ContextualUnstructureHook


@frozen
class AttributeOverride:
    omit_if_default: bool | None = None
    rename: str | None = None
    omit: bool | None = None  # Omit the field completely.
    struct_hook: (
        Callable[[Any, Any], Any] | ContextualStructureHook | None
    ) = None
    unstruct_hook: (
        Callable[[Any], Any] | ContextualUnstructureHook | None
    ) = None


neutral = AttributeOverride()
already_generating = local()
