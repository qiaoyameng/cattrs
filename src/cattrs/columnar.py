"""Columnar batch structuring support."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, overload

from attrs import NOTHING, Factory, frozen
from attrs import has as attrs_has
from attrs import resolve_types

from ._compat import (
    adapted_fields,
    get_origin,
    has_with_generic,
    is_annotated,
    is_bare,
    is_generic,
)
from ._generics import deep_copy_with
from .errors import (
    ColumnLengthError,
    ColumnValidationError,
    ColumnValidationNote,
    ForbiddenExtraKeysError,
    MissingColumnError,
)
from .gen import AttributeOverride, neutral
from .gen._generics import generate_mapping
from .gen._shared import _annotated_override_or_default, find_structure_handler

__all__ = [
    "ColumnBatch",
    "ColumnStructurePlan",
    "make_column_structure_plan",
]

T = TypeVar("T")
Columns = Mapping[str, Sequence[Any]]


@frozen
class _ColumnField:
    name: str
    column_name: str
    attribute_name: str
    type: Any
    init: bool
    kw_only: bool
    handler: Any
    has_default: bool
    default: Any


@frozen
class ColumnBatch(Generic[T]):
    """A batch of structured rows and the row numbers that produced them."""

    objects: list[T]
    rows: list[int]
    errors: list[Exception] = Factory(list)


@frozen
class ColumnStructurePlan(Generic[T]):
    """A reusable plan for reading a class from named data columns."""

    cl: type[T]
    fields: tuple[_ColumnField, ...]
    columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    forbid_extra_keys: bool

    def _validate(self, columns: Mapping[str, Any]) -> tuple[int, list[Exception]]:
        errors: list[Exception] = []
        for field in self.fields:
            if (
                field.init
                and not field.has_default
                and field.column_name not in columns
            ):
                errors.append(
                    MissingColumnError(self.cl, field.name, field.column_name)
                )

        known_columns = set(self.columns)
        lengths: dict[str, int] = {}
        invalid: list[str] = []
        for column, data in columns.items():
            if column not in known_columns:
                continue
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                lengths[column] = len(data)
            else:
                invalid.append(column)

        if invalid:
            errors.append(
                TypeError(
                    "Columns must be non-string sequences; invalid columns: "
                    f"{sorted(invalid)}"
                )
            )
        elif len({*lengths.values()}) > 1:
            errors.append(ColumnLengthError(self.cl, dict(sorted(lengths.items()))))

        if self.forbid_extra_keys:
            unknown = set(columns) - set(self.columns)
            if unknown:
                errors.append(ForbiddenExtraKeysError("", self.cl, unknown))

        return max(lengths.values(), default=0), errors

    def iter_batches(
        self,
        columns: Columns,
        batch_size: int | None = None,
        *,
        on_error: Literal["raise", "collect"] = "raise",
    ) -> Iterator[ColumnBatch[T]]:
        """Yield structured objects in batches, without making per-row dicts."""
        if on_error not in ("raise", "collect"):
            raise ValueError("on_error must be 'raise' or 'collect'")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        row_count, errors = self._validate(columns)
        if errors:
            if on_error == "raise":
                raise ColumnValidationError(
                    f"While structuring columns for {self.cl!r}", errors, self.cl
                )
            yield ColumnBatch([], [], errors)
            return

        if row_count == 0:
            return
        size = batch_size or row_count
        for start in range(0, row_count, size):
            stop = min(start + size, row_count)
            objects: list[T] = []
            rows: list[int] = []
            batch_errors: list[Exception] = []

            for row_index in range(start, stop):
                args: list[Any] = []
                kwargs: dict[str, Any] | None = None
                post_init: list[tuple[str, Any]] = []
                row_errors: list[Exception] = []
                use_keywords = False

                for field in self.fields:
                    if field.column_name not in columns:
                        if not field.init:
                            continue
                        if field.kw_only:
                            continue
                        if (
                            isinstance(field.default, Factory)
                            and field.default.takes_self
                        ):
                            use_keywords = True
                            continue
                        default = field.default
                        if isinstance(default, Factory):
                            default = default.factory()
                        args.append(default)
                        continue

                    value = columns[field.column_name][row_index]
                    if field.handler is None:
                        converted = value
                    else:
                        try:
                            converted = field.handler(value, field.type)
                        except Exception as exc:
                            note = ColumnValidationNote(
                                f"Structuring {self.cl.__qualname__} "
                                f"@ row {row_index}, column {field.column_name!r}",
                                row_index,
                                field.name,
                                field.type,
                            )
                            exc.__notes__ = [
                                *getattr(exc, "__notes__", []),
                                note,
                            ]
                            row_errors.append(exc)
                            continue

                    if not field.init:
                        post_init.append((field.name, converted))
                    elif field.kw_only or use_keywords:
                        if kwargs is None:
                            kwargs = {}
                        kwargs[field.attribute_name] = converted
                    else:
                        args.append(converted)

                if row_errors:
                    if on_error == "raise":
                        raise ColumnValidationError(
                            f"While structuring columns for {self.cl!r}",
                            row_errors,
                            self.cl,
                        )
                    batch_errors.extend(row_errors)
                    continue

                try:
                    if kwargs is None:
                        obj = self.cl(*args)
                    else:
                        obj = self.cl(*args, **kwargs)
                    for attribute_name, value in post_init:
                        setattr(obj, attribute_name, value)
                except Exception as exc:
                    note = ColumnValidationNote(
                        f"Structuring {self.cl.__qualname__} @ row {row_index}",
                        row_index,
                        "",
                        None,
                    )
                    exc.__notes__ = [*getattr(exc, "__notes__", []), note]
                    if on_error == "raise":
                        raise ColumnValidationError(
                            f"While structuring columns for {self.cl!r}",
                            [exc],
                            self.cl,
                        ) from exc
                    batch_errors.append(exc)
                else:
                    objects.append(obj)
                    rows.append(row_index)

            yield ColumnBatch(objects, rows, batch_errors)

    @overload
    def structure(self, columns: Columns, *, on_error: Literal["raise"]) -> list[T]: ...

    @overload
    def structure(
        self, columns: Columns, *, on_error: Literal["collect"]
    ) -> ColumnBatch[T]: ...

    @overload
    def structure(
        self, columns: Columns, *, on_error: str = "raise"
    ) -> list[T] | ColumnBatch[T]: ...

    def structure(
        self, columns: Columns, *, on_error: Literal["raise", "collect"] = "raise"
    ) -> list[T] | ColumnBatch[T]:
        """Structure all rows, either raising or returning collected errors."""
        if on_error == "raise":
            return [
                obj
                for batch in self.iter_batches(columns, on_error="raise")
                for obj in batch.objects
            ]

        objects: list[T] = []
        rows: list[int] = []
        errors: list[Exception] = []
        for batch in self.iter_batches(columns, on_error="collect"):
            objects.extend(batch.objects)
            rows.extend(batch.rows)
            errors.extend(batch.errors)
        return ColumnBatch(objects, rows, errors)


def make_column_structure_plan(
    cl: type[T],
    converter: Any,
    **overrides: AttributeOverride,
) -> ColumnStructurePlan[T]:
    """Create a reusable column-access plan for an attrs class or dataclass."""
    if not has_with_generic(cl):
        raise TypeError(
            f"Columnar structuring requires an attrs class or dataclass, got {cl!r}"
        )

    hook_overrides = getattr(
        converter.get_structure_hook(cl, cache_result=False),
        "overrides",
        {},
    )

    type_mapping = {}
    if is_generic(cl):
        base = get_origin(cl)
        type_mapping = generate_mapping(cl, type_mapping)
        if base is not None:
            cl = base

    for base in getattr(cl, "__orig_bases__", ()):
        if is_generic(base):
            type_mapping = generate_mapping(base, type_mapping)
            break

    attribs = adapted_fields(cl)
    if attrs_has(cl) and any(isinstance(attribute.type, str) for attribute in attribs):
        resolve_types(cl)
        attribs = adapted_fields(cl)

    use_alias = getattr(converter, "use_alias", False)
    prefer = converter._prefer_attrib_converters
    type_overrides = getattr(converter, "type_overrides", {})
    plan_fields = []

    for attribute in attribs:
        name = attribute.name
        override = overrides.get(name)
        if override is None:
            override = hook_overrides.get(name)
        if override is None:
            override = type_overrides.get(attribute.type)
        if override is None:
            override = _annotated_override_or_default(attribute.type, neutral)

        if override.omit or (override.omit is None and not attribute.init):
            continue

        field_type = attribute.type
        if isinstance(field_type, TypeVar):
            field_type = type_mapping.get(field_type.__name__, field_type)
        elif is_generic(field_type) and not is_bare(field_type) and not is_annotated(
            field_type
        ):
            field_type = deep_copy_with(field_type, type_mapping, cl)

        if override.struct_hook is not None:
            handler = override.struct_hook
        else:
            handler = find_structure_handler(
                attribute, field_type, converter, prefer
            )

        column_name = (
            override.rename
            if override.rename is not None
            else attribute.alias if use_alias else name
        )

        plan_fields.append(
            _ColumnField(
                name=name,
                column_name=column_name,
                attribute_name=attribute.alias,
                type=field_type,
                init=attribute.init,
                kw_only=attribute.kw_only,
                handler=handler,
                has_default=attribute.default is not NOTHING,
                default=attribute.default,
            )
        )

    return ColumnStructurePlan(
        cl=cl,
        fields=tuple(plan_fields),
        columns=tuple(field.column_name for field in plan_fields),
        required_columns=tuple(
            field.column_name for field in plan_fields if not field.has_default
        ),
        forbid_extra_keys=getattr(converter, "forbid_extra_keys", False),
    )
