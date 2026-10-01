import copy
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import MISSING
from typing import Any, Callable, Union

from attrs import NOTHING, NothingType

from .. import BaseConverter
from .._compat import (
    NoneType,
    adapted_fields,
    get_newtype_base,
    get_origin,
    has,
    is_literal,
    is_subclass,
    is_union_type,
)
from ..errors import AmbiguousUnionError, UnionStructureError
from ..typealiases import get_type_alias_base, is_type_alias

__all__ = [
    "configure_tagged_union",
    "configure_union_passthrough",
    "configure_union_trial",
    "default_tag_generator",
]


def default_tag_generator(typ: type) -> str:
    """Return the class name."""
    return typ.__name__


def configure_tagged_union(
    union: Any,
    converter: BaseConverter,
    tag_generator: Callable[[type], str] = default_tag_generator,
    tag_name: str = "_type",
    default: Union[type, NothingType] = NOTHING,
) -> None:
    """
    Configure the converter so that `union` (which should be a union, or a type alias
    of one) is un/structured with the help of an additional piece of data in the
    unstructured payload, the tag.

    :param converter: The converter to apply the strategy to.
    :param tag_generator: A `tag_generator` function is used to map each
        member of the union to a tag, which is then included in the
        unstructured payload. The default tag generator returns the name of
        the class.
    :param tag_name: The key under which the tag will be set in the
        unstructured payload. By default, `'_type'`.
    :param default: An optional class to be used if the tag information
        is not present when structuring.

    The tagged union strategy currently only works with the dict
    un/structuring base strategy.

    .. versionadded:: 23.1.0

    ..  versionchanged:: 25.1
        Type aliases of unions are now also supported.
    """
    if is_type_alias(union):
        union = union.__value__
    args = union.__args__

    tag_to_hook = {}
    exact_cl_unstruct_hooks = {}
    cl_to_tag = {}

    if default is not NOTHING:
        default_handler = converter.get_structure_hook(default)

        def structure_default(val: dict, _cl=default, _h=default_handler):
            return _h(val, _cl)

        tag_to_hook = defaultdict(lambda: structure_default)
        cl_to_tag = defaultdict(lambda: default)

        if getattr(converter, "forbid_extra_keys", False):

            def structure_tagged_union(
                val: dict,
                _,
                _tag_to_hook=tag_to_hook,
                _tag_name=tag_name,
                _dh=default_handler,
                _default=default,
            ) -> union:
                if _tag_name in val:
                    val = val.copy()
                    return _tag_to_hook[val.pop(_tag_name)](val)
                return _dh(val, _default)

        else:

            def structure_tagged_union(
                val: dict,
                _,
                _tag_to_hook=tag_to_hook,
                _tag_name=tag_name,
                _dh=default_handler,
                _default=default,
            ) -> union:
                if _tag_name in val:
                    return _tag_to_hook[val[_tag_name]](val)
                return _dh(val, _default)

    else:
        if getattr(converter, "forbid_extra_keys", False):

            def structure_tagged_union(
                val: dict, _, _tag_to_cl=tag_to_hook, _tag_name=tag_name
            ) -> union:
                val = val.copy()
                return _tag_to_cl[val.pop(_tag_name)](val)

        else:

            def structure_tagged_union(
                val: dict, _, _tag_to_cl=tag_to_hook, _tag_name=tag_name
            ) -> union:
                return _tag_to_cl[val[_tag_name]](val)

    def unstructure_tagged_union(
        val: union,
        _exact_cl_unstruct_hooks=exact_cl_unstruct_hooks,
        _cl_to_tag=cl_to_tag,
        _tag_name=tag_name,
    ) -> dict:
        res = _exact_cl_unstruct_hooks[val.__class__](val)
        res[_tag_name] = _cl_to_tag[val.__class__]
        return res

    converter.register_unstructure_hook(union, unstructure_tagged_union)
    converter.register_structure_hook(union, structure_tagged_union)

    for cl in args:
        tag = tag_generator(cl)
        struct_handler = converter.get_structure_hook(cl)
        unstruct_handler = converter.get_unstructure_hook(cl)

        def structure_union_member(val: dict, _cl=cl, _h=struct_handler) -> cl:
            return _h(val, _cl)

        def unstructure_union_member(val: union, _h=unstruct_handler) -> dict:
            return _h(val)

        tag_to_hook[tag] = structure_union_member
        exact_cl_unstruct_hooks[cl] = unstructure_union_member
        cl_to_tag[cl] = tag


def configure_union_passthrough(
    union: Any, converter: BaseConverter, accept_ints_as_floats: bool = True
) -> None:
    """
    Configure the converter to support validating and passing through unions of the
    provided types and their subsets.

    For example, all mature JSON libraries natively support producing unions of ints,
    floats, Nones, and strings. Using this strategy, a converter can be configured
    to efficiently validate and pass through unions containing these types.

    The most important point is that another library (in this example the JSON
    library) handles producing the union, and the converter is configured to just
    validate it.

    Literals of provided types are also supported, and are checked by value.

    NewTypes of provided types are also supported.

    The strategy is designed to be O(1) in execution time, and independent of the
    ordering of types in the union.

    If the union contains a class and one or more of its subclasses, the subclasses
    will also be included when validating the superclass.

    :param accept_ints_as_floats: When set (the default), if the provided union
        contains both ints and floats, actual unions containing only floats will also accept
        ints. See https://typing.python.org/en/latest/spec/special-types.html#special-cases-for-float-and-complex
        for more information.

    .. versionadded:: 23.2.0
    .. versionchanged:: 25.2.0
        Introduced the `accept_ints_as_floats` parameter.
    .. versionchanged:: 26.2.0
        PEP 695 type aliases are now supported as union members.
    """

    def get_passthrough_base(type: Any) -> Any:
        return get_type_alias_base(type) if is_type_alias(type) else type

    args = {get_passthrough_base(type) for type in union.__args__}

    def make_structure_native_union(exact_type: Any) -> Callable:
        # `exact_type` is likely to be a subset of the entire configured union (`args`).
        literal_values = {
            v for t in exact_type.__args__ if is_literal(t) for v in t.__args__
        }

        # We have no idea what the actual type of `val` will be, so we can't
        # use it blindly with an `in` check since it might not be hashable.
        # So we do an additional check when handling literals.
        # Note: do no use `literal_values` here, since {0, False} gets reduced to {0}
        literal_classes = {
            v.__class__
            for t in exact_type.__args__
            if is_literal(t)
            for v in t.__args__
        }

        non_literal_classes = {
            get_newtype_base(t) or get_passthrough_base(t)
            for t in exact_type.__args__
            if not is_literal(t)
            and ((get_newtype_base(t) or get_passthrough_base(t)) in args)
        }

        # We augment the set of allowed classes with any configured subclasses of
        # the exact subclasses.
        non_literal_classes |= {
            a for a in args if any(is_subclass(a, c) for c in non_literal_classes)
        }

        # We check for spillover - union types not handled by the strategy.
        # If spillover exists and we fail to validate our types, we call
        # further into the converter with the rest.
        spillover = {
            a
            for a in exact_type.__args__
            if (get_newtype_base(a) or get_passthrough_base(a))
            not in non_literal_classes
            and not is_literal(a)
        }

        # By default, when floats are part of the union, accept ints too.
        if (
            accept_ints_as_floats
            and int in args
            and float in args
            and float in non_literal_classes
            and int not in non_literal_classes
        ):
            non_literal_classes.add(int)

        if spillover:
            spillover_type = (
                Union[tuple(spillover)] if len(spillover) > 1 else next(iter(spillover))
            )

            def structure_native_union(
                val: Any,
                _: Any,
                classes=non_literal_classes,
                vals=literal_values,
                converter=converter,
                spillover=spillover_type,
            ) -> exact_type:
                if val.__class__ in literal_classes and val in vals:
                    return val
                if val.__class__ in classes:
                    return val
                return converter.structure(val, spillover)

        else:

            def structure_native_union(
                val: Any, _: Any, classes=non_literal_classes, vals=literal_values
            ) -> exact_type:
                if val.__class__ in literal_classes and val in vals:
                    return val
                if val.__class__ in classes:
                    return val
                raise TypeError(f"{val} ({val.__class__}) not part of {_}")

        return structure_native_union

    def contains_native_union(exact_type: Any) -> bool:
        """Can we handle this type?"""
        if is_union_type(exact_type):
            type_args = set(exact_type.__args__)
            # We special case optionals, since they are very common
            # and are handled a little more efficiently by default.
            if len(type_args) == 2 and type(None) in type_args:
                return False

            literal_classes = {
                lit_arg.__class__
                for t in type_args
                if is_literal(t)
                for lit_arg in t.__args__
            }
            non_literal_types = {
                get_newtype_base(t) or get_passthrough_base(t)
                for t in type_args
                if not is_literal(t)
            }

            return (literal_classes | non_literal_types) & args
        return False

    converter.register_structure_hook_factory(
        contains_native_union, make_structure_native_union
    )


def configure_union_trial(
    union: Any,
    converter: BaseConverter,
) -> None:
    """
    Configure the converter to structure `union` (which should be a union, or a
    type alias of one) by independently trying every potentially-applicable
    member.

    This strategy is useful when the union members have no common discriminator
    (and so :func:`configure_tagged_union` cannot be used) and no member has a
    unique required field (and so the default union strategy cannot be used).

    For each value to structure, the strategy works as follows:

    1. Members which *obviously* do not apply are filtered out by convention:
       an _attrs_ class or dataclass whose required (non-default) fields are
       not all present in the input mapping cannot apply. Members which are not
       _attrs_ classes or dataclasses (such as non-union primitives, or nested
       unions) are never filtered out on this basis.
    2. Every remaining candidate is structured independently on its own
       :func:`copy.deepcopy` of the input. The first member of the union is
       *not* tried first; the outcome is independent of the order of the
       members in the union definition.
    3. If exactly one member succeeds, its result is returned.
    4. If several members succeed, :class:`cattrs.AmbiguousUnionError` is
       raised.
    5. If every member fails, :class:`cattrs.UnionStructureError` is raised,
       carrying the exception produced while trying each member.

    .. warning::
       Each candidate receives an independent deep copy of the *input*, so a
       custom hook mutating its input cannot affect another candidate. Hooks
       may still have external side effects (files, databases, global state...);
       those cannot be isolated or rolled back.

    Nested unions are supported: they are flattened into the outer union and
    their members are tried as candidates as well.

    .. versionadded:: 26.2.0
    """
    if is_type_alias(union):
        union = get_type_alias_base(union)

    def is_supported_trial_union(exact_type: Any) -> bool:
        if not is_union_type(exact_type):
            return False
        # Let the specialized, more efficient optional hook handle plain
        # `Optional[T]`; only unions overlapping with the configured member
        # types are claimed.
        if len(exact_type.__args__) == 2 and NoneType in exact_type.__args__:
            return False
        return any(
            (get_origin(m) or _get_passthrough_base(m)) in member_types
            for m in exact_type.__args__
            if m is not NoneType
        )

    member_types = {
        (get_origin(a) or _get_passthrough_base(a))
        for a in _flatten_union(union)
        if a is not NoneType
    }

    converter.register_structure_hook_factory(
        is_supported_trial_union,
        lambda exact_type: _make_trial_hook(exact_type, converter),
    )


def _get_passthrough_base(type: Any) -> Any:
    return get_type_alias_base(type) if is_type_alias(type) else type


def _flatten_union(union: Any) -> tuple:
    """Recursively flatten (possibly nested) union members."""
    res = []
    for arg in union.__args__:
        base = _get_passthrough_base(arg) if is_type_alias(arg) else arg
        if is_union_type(base):
            res.extend(_flatten_union(base))
        else:
            res.append(arg)
    return tuple(res)


def _required_keys(typ: Any, converter: BaseConverter) -> frozenset[str] | None:
    """Required (non-default) input keys for an attrs/dataclass member.

    Returns `None` when the member is not an introspectable _attrs_ class or
    dataclass, meaning it cannot be filtered out by the key convention.
    """
    base = get_origin(typ) or _get_passthrough_base(typ)
    if not has(base):
        return None

    overrides = getattr(
        converter.get_structure_hook(typ), "overrides", {}
    )

    keys = set()
    for at in adapted_fields(base):
        if not at.init:
            continue
        if at.default is not NOTHING and at.default is not MISSING:
            continue
        override = overrides.get(at.name)
        name = override.rename if override is not None and override.rename else at.name
        keys.add(name)
    return frozenset(keys)


def _prune_candidates(
    members: tuple, val: Any, converter: BaseConverter
) -> tuple[list, dict[Any, BaseException]]:
    """Filter out members which obviously cannot structure `val`.

    Returns the retained candidates and, for every filtered-out member, the
    reason why it was considered inapplicable.
    """
    candidates = []
    reasons: dict[Any, BaseException] = {}

    seen = set()
    for member in members:
        base = _get_passthrough_base(member) if is_type_alias(member) else member
        if base in seen:
            continue
        seen.add(base)

        if member is NoneType:
            if val is None:
                candidates.append(member)
            else:
                reasons[member] = TypeError("value is not None")
            continue

        if isinstance(val, Mapping):
            required = _required_keys(member, converter)
            if required is not None:
                missing = required - val.keys()
                if missing:
                    reasons[member] = TypeError(
                        "missing required key(s): " + ", ".join(sorted(missing))
                    )
                    continue

        candidates.append(member)

    return candidates, reasons


def _make_trial_hook(union: Any, converter: BaseConverter) -> Callable:
    members = _flatten_union(union)

    def structure_union_by_trial(val: Any, _) -> union:
        if val is None and NoneType in members:
            return None

        candidates, prune_reasons = _prune_candidates(members, val, converter)

        # Sort candidates by qualified name so the outcome never depends on
        # the order in which members were written in the union definition.
        candidates = sorted(candidates, key=_member_sort_key)

        succeeded = []
        failures = dict(prune_reasons)

        for member in candidates:
            # Each branch gets an independent view of the input; mutating it
            # inside a custom hook cannot leak into another branch.
            try:
                branch_val = copy.deepcopy(val)
            except Exception:
                branch_val = val
            try:
                succeeded.append((member, converter.structure(branch_val, member)))
            except Exception as exc:
                failures[member] = exc

        if len(succeeded) == 1:
            return succeeded[0][1]
        if len(succeeded) > 1:
            ambiguous_members = [m for m, _ in succeeded]
            ambiguous_results = [r for _, r in succeeded]
            raise AmbiguousUnionError(
                "Value can be structured as more than one member of "
                f"{union!r}: {', '.join(repr(m) for m in ambiguous_members)}",
                union,
                ambiguous_members,
                ambiguous_results,
            )

        ordered_members = sorted(members, key=_member_sort_key)
        raise UnionStructureError(
            None,
            union,
            ordered_members,
            [
                failures.get(m, TypeError("member was not tried"))
                for m in ordered_members
            ],
        )

    return structure_union_by_trial


def _member_sort_key(member: Any) -> tuple:
    base = get_origin(member) or _get_passthrough_base(member)
    module = getattr(base, "__module__", "") or ""
    qualname = getattr(base, "__qualname__", None) or repr(member)
    return (0, module, qualname) if base is not NoneType else (1, "", "None")
