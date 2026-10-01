"""Utilities for type aliases."""

from __future__ import annotations

import sys
from collections.abc import Mapping as AbcMapping
from typing import TYPE_CHECKING, Any

from ._compat import (
    NoneType,
    get_origin,
    is_annotated,
    is_generic,
    is_union_type,
)
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
    if is_generic(type):
        mapping = generate_mapping(type)
        if getattr(base := type.__value__, "__name__", None) in mapping:
            return mapping[base.__name__]
        return deep_copy_with(base, mapping)
    return type.__value__


def type_alias_structure_factory(type: Any, converter: BaseConverter) -> StructureHook:
    base = get_type_alias_base(type)

    def structure_type_alias(v: Any, _: Any, *, __base=base) -> Any:
        return converter.structure(v, __base)

    structure_type_alias._cattrs_generated_type_alias = True
    return structure_type_alias


def type_alias_unstructure_factory(
    type: Any, converter: BaseConverter
) -> UnstructureHook:
    base = get_type_alias_base(type)

    if is_union_type(base):
        base_hook = converter.get_unstructure_hook(base)
        if base_hook is not converter._unstructure_union:
            return lambda v, __hook=base_hook: __hook(v)

        def unstructure_type_alias(value: Any, __base=base) -> Any:
            for member in __base.__args__:
                if _matches_union_member(value, member):
                    return converter.unstructure(value, unstructure_as=member)
            return converter.unstructure(value)

        return unstructure_type_alias

    return lambda v, __base=base: converter.unstructure(v, unstructure_as=__base)


def _matches_union_member(value: Any, member: Any) -> bool:
    if member is NoneType:
        return value is None
    if is_annotated(member):
        return _matches_union_member(value, member.__origin__)
    if is_union_type(member):
        return any(_matches_union_member(value, arg) for arg in member.__args__)
    if is_type_alias(member):
        return _matches_union_member(value, get_type_alias_base(member))

    origin = get_origin(member)
    if origin is not None:
        if origin is list:
            return isinstance(value, list)
        if origin is dict:
            return isinstance(value, AbcMapping)
        return False
    return isinstance(value, member) if isinstance(member, type) else False
