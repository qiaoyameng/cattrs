"""A strategy for un/structuring object graphs with shared references and cycles."""

from __future__ import annotations

from typing import Any, Iterable

from attrs import NOTHING, Factory

from .._compat import NoneType, adapted_fields, get_args, get_origin, has, is_union_type
from ..converters import BaseConverter
from ..errors import (
    DanglingReferenceError,
    DuplicateReferenceError,
    GraphConversionError,
    GraphCycleError,
    GraphTypeMismatchError,
    UnknownGraphTypeError,
)

__all__ = ["GraphSupport"]

_REF = "$ref"
_TYPE = "$type"
_ID = "$id"
_FIELDS = "fields"
_ITEMS = "items"
_ENTRIES = "entries"


def _type_name(cl: type) -> str:
    return f"{cl.__module__}.{cl.__qualname__}"


def _is_frozen(cl: type) -> bool:
    params = getattr(cl, "__dataclass_params__", None)
    if params is not None:
        return bool(params.frozen)
    return getattr(getattr(cl, "__setattr__", None), "__name__", "").startswith(
        "_frozen_setattr"
    )


def _requires_custom_constructor(cl: type) -> bool:
    """Whether `cl` routes initialization through a user-defined constructor."""
    init = cl.__init__
    if init is object.__init__:
        return False
    if hasattr(cl, "__attrs_attrs__"):
        # attrs moves the generated init to `__attrs_init__` when the user
        # defines their own `__init__` (or disables init generation).
        return hasattr(cl, "__attrs_init__")
    # Dataclasses keep a user-defined `__init__`; generated ones are exec'd.
    code = getattr(init, "__code__", None)
    return code is None or code.co_filename != "<string>"


def _default_of(field: Any, path: str) -> Any:
    """Produce the default for an attrs-style field, or raise if there is none."""
    default = field.default
    if default is NOTHING:
        raise GraphConversionError(
            f"Missing required field {field.name!r} for pre-allocated instance", path
        )
    if isinstance(default, Factory):
        return default.factory()
    return default


class _Record:
    """Bookkeeping for one defined node while structuring."""

    __slots__ = ("cls", "done", "kind", "obj")

    def __init__(self, kind: str, cls: type | None, obj: Any) -> None:
        self.kind = kind
        self.cls = cls
        self.obj = obj
        self.done = False


class GraphSupport:
    """Explicit opt-in graph (shared reference and cycle) conversion for a converter.

    The strategy wraps a converter without modifying it; the converter's own
    tree format is unchanged. Only declared attrs/dataclass types and the
    ``list``/``dict`` containers within them are tracked as graph nodes;
    everything else (scalars and any type with its own hooks) is converted by
    the wrapped converter's existing hooks.

    The unstructured document uses an inline reference representation:

    - first occurrence of a class:
      ``{"$type": "module.QualName", "$id": n, "fields": {...}}``
    - first occurrence of a list: ``{"$id": n, "items": [...]}``
    - first occurrence of a dict: ``{"$id": n, "entries": {...}}``
    - every further occurrence: ``{"$ref": n}``

    Reference ids are allocated per conversion and are meaningless outside the
    document they appear in; no state is shared between two conversions.

    :param converter: The converter whose hooks are used for scalar values.
    :param allowed_types: Additional attrs/dataclass types that may appear in
        ``$type`` names, beyond those reachable from the target type's
        annotations. Type names from the payload are never imported; they must
        be declared here or be reachable from the structured type.

    .. versionadded:: 26.3.0
    """

    def __init__(
        self,
        converter: BaseConverter,
        allowed_types: Iterable[type] = (),
    ) -> None:
        self.converter = converter
        self.allowed_types = tuple(allowed_types)
        self._fields_cache: dict[type, Any] = {}

    def unstructure(self, obj: Any) -> Any:
        """Unstructure `obj` into a graph document with reference nodes."""
        return _UnstructureState(self).run(obj)

    def structure(self, data: Any, cl: Any) -> Any:
        """Structure a graph document back into objects, restoring identity."""
        registry = self._build_registry(cl)
        return _StructureState(self, registry).run(data, cl)

    def fields_of(self, cl: type) -> Any:
        if cl not in self._fields_cache:
            self._fields_cache[cl] = adapted_fields(cl)
        return self._fields_cache[cl]

    def _build_registry(self, root: Any) -> dict[str, type]:
        registry: dict[str, type] = {}

        def visit(t: Any) -> None:
            if t is None or t is Any:
                return
            if is_union_type(t):
                for arg in get_args(t):
                    if arg is not NoneType:
                        visit(arg)
                return
            if get_origin(t) is not None:
                for arg in get_args(t):
                    visit(arg)
                return
            if isinstance(t, type) and has(t):
                name = _type_name(t)
                if name not in registry:
                    registry[name] = t
                    for f in self.fields_of(t):
                        if f.type is not None:
                            visit(f.type)

        visit(root)
        for t in self.allowed_types:
            visit(t)
        return registry


class _UnstructureState:
    """Per-conversion unstructuring state; discarded after each conversion."""

    def __init__(self, support: GraphSupport) -> None:
        self.support = support
        self.ref_ids: dict[int, int] = {}
        self.objects: list[Any] = []  # keeps ids alive for this conversion
        self.fields_cache: dict[type, Any] = {}

    def run(self, obj: Any) -> Any:
        return self._unstructure(obj)

    def _next_id(self, obj: Any) -> int:
        ref_id = len(self.objects)
        self.objects.append(obj)
        self.ref_ids[id(obj)] = ref_id
        return ref_id

    def _unstructure(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, list):
            existing = self.ref_ids.get(id(obj))
            if existing is not None:
                return {_REF: existing}
            return {
                _ID: self._next_id(obj),
                _ITEMS: [self._unstructure(item) for item in obj],
            }
        if isinstance(obj, dict):
            existing = self.ref_ids.get(id(obj))
            if existing is not None:
                return {_REF: existing}
            entries = {}
            for key, value in obj.items():
                out_key = self.support.converter.unstructure(key)
                if isinstance(out_key, (dict, list)):
                    raise GraphConversionError(
                        f"Unsupported compound dict key {key!r}; "
                        "dict keys must unstructure to scalars"
                    )
                entries[out_key] = self._unstructure(value)
            return {_ID: self._next_id(obj), _ENTRIES: entries}
        cl = type(obj)
        if has(cl):
            existing = self.ref_ids.get(id(obj))
            if existing is not None:
                return {_REF: existing}
            ref_id = self._next_id(obj)
            fields = self.fields_cache.get(cl)
            if fields is None:
                fields = self.fields_cache[cl] = self.support.fields_of(cl)
            fields_payload = {
                f.name: self._unstructure(getattr(obj, f.name)) for f in fields
            }
            return {_TYPE: _type_name(cl), _ID: ref_id, _FIELDS: fields_payload}
        return self.support.converter.unstructure(obj)


class _StructureState:
    """Per-conversion structuring state; discarded after each conversion."""

    def __init__(self, support: GraphSupport, registry: dict[str, type]) -> None:
        self.support = support
        self.registry = registry
        self.records: dict[int, _Record] = {}

    def run(self, data: Any, cl: Any) -> Any:
        return self._structure(data, cl, "$")

    def _structure(self, data: Any, expected: Any, path: str) -> Any:
        if expected is None:
            expected = Any
        if data is None:
            if expected is Any:
                return None
            try:
                return self.support.converter.structure(None, expected)
            except Exception as exc:
                exc.add_note(f"while structuring graph position {path}")
                raise
        if isinstance(data, dict):
            if _REF in data:
                return self._structure_ref(data, expected, path)
            if _TYPE in data:
                return self._structure_class(data, expected, path)
            if _ITEMS in data:
                return self._structure_list(data, expected, path)
            if _ENTRIES in data:
                return self._structure_dict(data, expected, path)
        if self._expects_node(expected):
            raise GraphTypeMismatchError(
                f"Expected a graph node for {expected!r}, got {data!r}", path
            )
        try:
            return self.support.converter.structure(data, expected)
        except Exception as exc:
            exc.add_note(f"while structuring graph position {path}")
            raise

    def _expects_node(self, expected: Any) -> bool:
        if expected is None or expected is Any:
            return False
        if is_union_type(expected):
            return any(
                self._expects_node(arg)
                for arg in get_args(expected)
                if arg is not NoneType
            )
        origin = get_origin(expected)
        if origin in (list, dict) or expected in (list, dict):
            return True
        return isinstance(expected, type) and has(expected)

    def _resolve_expected(
        self, expected: Any, kind: str, cls: type | None, path: str
    ) -> Any:
        """Validate a node against the declared type, returning the matching member."""
        if expected is None or expected is Any:
            return Any
        if is_union_type(expected):
            for arg in get_args(expected):
                if arg is NoneType:
                    continue
                try:
                    return self._resolve_expected(arg, kind, cls, path)
                except GraphTypeMismatchError:
                    continue
            raise GraphTypeMismatchError(
                f"Graph {kind} node does not match any member of {expected!r}", path
            )
        origin = get_origin(expected)
        if kind == "list" and (origin is list or expected is list):
            return expected
        if kind == "dict" and (origin is dict or expected is dict):
            return expected
        if kind == "class":
            target = origin if origin is not None else expected
            if isinstance(target, type) and issubclass(cls, target):
                return expected
        raise GraphTypeMismatchError(
            f"Graph {kind} node"
            + (f" of type {_type_name(cls)!r}" if cls is not None else "")
            + f" does not match declared type {expected!r}",
            path,
        )

    @staticmethod
    def _check_ref_id(data: dict, path: str) -> int:
        ref_id = data[_ID]
        if not isinstance(ref_id, int) or isinstance(ref_id, bool):
            raise GraphConversionError(f"Invalid reference id {ref_id!r}", path)
        return ref_id

    def _define(self, ref_id: int, record: _Record, path: str) -> None:
        if ref_id in self.records:
            raise DuplicateReferenceError(ref_id, path)
        self.records[ref_id] = record

    def _structure_ref(self, data: dict, expected: Any, path: str) -> Any:
        if set(data) != {_REF}:
            raise GraphConversionError(f"Malformed reference node {data!r}", path)
        ref_id = data[_REF]
        if not isinstance(ref_id, int) or isinstance(ref_id, bool):
            raise GraphConversionError(f"Invalid reference id {ref_id!r}", path)
        record = self.records.get(ref_id)
        if record is None:
            raise DanglingReferenceError(ref_id, path)
        self._resolve_expected(expected, record.kind, record.cls, path)
        if record.kind == "class" and record.obj is None:
            record.obj = self._allocate(record.cls, path)
        return record.obj

    def _structure_class(self, data: dict, expected: Any, path: str) -> Any:
        if set(data) != {_TYPE, _ID, _FIELDS}:
            raise GraphConversionError(f"Malformed class node {data!r}", path)
        type_name = data[_TYPE]
        if not isinstance(type_name, str):
            raise GraphConversionError(f"Invalid type name {type_name!r}", path)
        cls = self.registry.get(type_name)
        if cls is None:
            raise UnknownGraphTypeError(type_name, path)
        self._resolve_expected(expected, "class", cls, path)
        ref_id = self._check_ref_id(data, path)
        payload = data[_FIELDS]
        if not isinstance(payload, dict):
            raise GraphConversionError(f"Invalid fields payload {payload!r}", path)
        record = _Record("class", cls, None)
        self._define(ref_id, record, path)
        field_defs = {f.name: f for f in self.support.fields_of(cls)}
        extra = set(payload) - set(field_defs)
        if extra:
            raise GraphConversionError(
                f"Unexpected fields for {_type_name(cls)}: {sorted(extra)}", path
            )
        init_values: dict[str, Any] = {}
        deferred_values: dict[str, Any] = {}
        for name, value in payload.items():
            field = field_defs[name]
            target = init_values if field.init else deferred_values
            target[name] = self._structure(
                value, field.type if field.type is not None else Any, f"{path}.{name}"
            )
        if record.obj is None:
            kwargs = {field_defs[name].alias: v for name, v in init_values.items()}
            try:
                record.obj = cls(**kwargs)
            except Exception as exc:
                raise GraphConversionError(
                    f"Cannot construct {_type_name(cls)}: {exc}", path
                ) from exc
        else:
            # The instance was pre-allocated to satisfy a reference cycle.
            for name, value in init_values.items():
                setattr(record.obj, name, value)
            for name, field in field_defs.items():
                if name not in payload and field.init:
                    setattr(record.obj, name, _default_of(field, path))
        for name, value in deferred_values.items():
            try:
                setattr(record.obj, name, value)
            except Exception as exc:
                raise GraphConversionError(
                    f"Cannot set init=False field {name!r} on {_type_name(cls)}: {exc}",
                    path,
                ) from exc
        record.done = True
        return record.obj

    def _structure_list(self, data: dict, expected: Any, path: str) -> Any:
        if set(data) != {_ID, _ITEMS}:
            raise GraphConversionError(f"Malformed list node {data!r}", path)
        member = self._resolve_expected(expected, "list", None, path)
        ref_id = self._check_ref_id(data, path)
        items = data[_ITEMS]
        if not isinstance(items, list):
            raise GraphConversionError(f"Invalid items payload {items!r}", path)
        args = get_args(member)
        elem_type = args[0] if args else Any
        result: list[Any] = []
        self._define(ref_id, _Record("list", None, result), path)
        for index, item in enumerate(items):
            result.append(self._structure(item, elem_type, f"{path}[{index}]"))
        return result

    def _structure_dict(self, data: dict, expected: Any, path: str) -> Any:
        if set(data) != {_ID, _ENTRIES}:
            raise GraphConversionError(f"Malformed dict node {data!r}", path)
        member = self._resolve_expected(expected, "dict", None, path)
        ref_id = self._check_ref_id(data, path)
        entries = data[_ENTRIES]
        if not isinstance(entries, dict):
            raise GraphConversionError(f"Invalid entries payload {entries!r}", path)
        args = get_args(member)
        key_type = args[0] if len(args) == 2 else Any
        value_type = args[1] if len(args) == 2 else Any
        result: dict[Any, Any] = {}
        self._define(ref_id, _Record("dict", None, result), path)
        for key, value in entries.items():
            if isinstance(key, (dict, list)):
                raise GraphConversionError(
                    f"Unsupported compound dict key {key!r}", path
                )
            out_key = (
                key
                if key_type is Any
                else self.support.converter.structure(key, key_type)
            )
            result[out_key] = self._structure(value, value_type, f"{path}[{key!r}]")
        return result

    def _allocate(self, cls: type, path: str) -> Any:
        name = _type_name(cls)
        if _is_frozen(cls):
            raise GraphCycleError(
                f"Reference cycle through frozen class {name}; "
                "it cannot be rebuilt without bypassing its initialization",
                path,
            )
        if _requires_custom_constructor(cls):
            raise GraphCycleError(
                f"Reference cycle through {name}, which requires a custom "
                "constructor; it cannot be safely rebuilt in two phases",
                path,
            )
        try:
            return object.__new__(cls)
        except TypeError as exc:
            raise GraphCycleError(
                f"Cannot pre-allocate {name} to resolve a reference cycle: {exc}",
                path,
            ) from exc
