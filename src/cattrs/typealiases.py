"""Utilities for type aliases."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from ._compat import get_origin
from ._generics import deep_copy_with
from .dispatch import StructureHook, UnstructureHook
from .gen._generics import generate_mapping

if TYPE_CHECKING:
    from .converters import BaseConverter

__all__ = [
    "get_type_alias_base",
    "is_type_alias",
    "type_alias_structure_factory",
    "type_alias_unstructure_factory",
]

if sys.version_info >= (3, 12):
    from types import GenericAlias
    from typing import TypeAliasType

    def is_type_alias(type: Any) -> bool:
        """Is this a PEP 695 type alias?"""
        return isinstance(
            type.__origin__ if type.__class__ is GenericAlias else type, TypeAliasType
        )

else:

    def is_type_alias(type: Any) -> bool:
        """Is this a PEP 695 type alias?"""
        return False


def get_type_alias_base(type: Any) -> Any:
    """
    What is this a type alias of?

    Works only on 3.12+.
    """
    base = type.__value__
    if type.__class__ is GenericAlias:
        origin = get_origin(type)
        if isinstance(origin, TypeAliasType) and getattr(origin, "__type_params__", ()):
            mapping = generate_mapping(type)
            return _substitute_type_alias_base(base, mapping)
    return base


def _substitute_type_alias_base(base: Any, mapping: dict[str, Any]) -> Any:
    if hasattr(base, "__name__") and base.__name__ in mapping:
        return mapping[base.__name__]
    return deep_copy_with(base, mapping)


def _make_type_alias_hook(
    type: Any, converter: BaseConverter, structure: bool
) -> StructureHook | UnstructureHook:
    base = get_type_alias_base(type)
    dispatch = (
        converter._structure_func if structure else converter._unstructure_func
    )
    get_hook = (
        converter.get_structure_hook if structure else converter.get_unstructure_hook
    )

    def placeholder(value: Any, type_: Any = type) -> Any:
        hook = dispatch.dispatch(type_)
        return hook(value, type_) if structure else hook(value)

    dispatch._direct_dispatch[type] = placeholder
    try:
        hook = get_hook(base)

        if structure:

            def type_alias_hook(value: Any, _, __hook=hook, __base=base) -> Any:
                return __hook(value, __base)

        else:

            def type_alias_hook(value: Any, __hook=hook) -> Any:
                return __hook(value)

        dispatch._direct_dispatch[type] = type_alias_hook
        dispatch.dispatch.cache_clear()
        return type_alias_hook
    except Exception:
        dispatch._direct_dispatch.pop(type, None)
        dispatch.dispatch.cache_clear()
        raise


def type_alias_structure_factory(type: Any, converter: BaseConverter) -> StructureHook:
    return _make_type_alias_hook(type, converter, structure=True)


def type_alias_unstructure_factory(
    type: Any, converter: BaseConverter
) -> UnstructureHook:
    return _make_type_alias_hook(type, converter, structure=False)
