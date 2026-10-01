from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .converters import BaseConverter


def enum_unstructure_factory(
    type: type[Enum], converter: "BaseConverter"
) -> Callable[[Enum], Any]:
    """A factory for generating enum unstructure hooks.

    If the enum is a typed enum (has `_value_`), we use the underlying value's hook.
    Otherwise, we use the value directly.
    """
    if "_value_" in type.__annotations__:
        return lambda e, *, context=None: converter.unstructure(
            e.value, context=context
        )

    return lambda e: e.value


def enum_structure_factory(
    type: type[Enum], converter: "BaseConverter"
) -> Callable[[Any, type[Enum]], Enum]:
    """A factory for generating enum structure hooks.

    If the enum is a typed enum (has `_value_`), we structure the value first.
    Otherwise, we use the value directly.
    """
    if "_value_" in type.__annotations__:
        val_type = type.__annotations__["_value_"]
        val_hook = converter.get_structure_hook(val_type)
        return lambda v, _, *, context=None: type(
            val_hook(v, val_type, context=context)
        )

    return lambda v, _: type(v)
