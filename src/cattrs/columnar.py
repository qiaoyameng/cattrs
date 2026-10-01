from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

from attrs import NOTHING, resolve_types
from attrs import has as attrs_has

from ._compat import (
    adapted_fields,
    get_args,
    get_origin,
    has_with_generic,
    is_annotated,
    is_generic,
    is_optional,
)
from ._generics import deep_copy_with
from .errors import (
    ColumnLengthError,
    ColumnRowValidationNote,
    ColumnValidationNote,
    ColumnarValidationError,
    ForbiddenExtraKeysError,
    InvalidColumnError,
    MissingColumnError,
)
from .fns import raise_error
from .gen._consts import neutral
from .gen._generics import generate_mapping
from .gen._shared import _annotated_override_or_default, find_structure_handler

T = TypeVar("T")


class ErrorMode(Enum):
    """Controls when columnar structuring reports errors."""

    STOP = "stop"
    COLLECT = "collect"
    RAISE = "stop"


@dataclass(frozen=True)
class _ColumnField:
    name: str
    alias: str
    type: Any
    required: bool
    optional: bool
    init: bool
    kw_only: bool
    parameter_name: str
    hook: Any
    has_default: bool


@dataclass(frozen=True)
class ColumnarResult(Generic[T]):
    """The objects and errors produced by a columnar structuring operation.

    ..  versionadded:: 26.2.0
    """

    objects: tuple[T, ...]
    errors: tuple[Exception, ...]
    row_count: int
    row_indices: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether all rows were structured successfully."""
        return not self.errors


class ColumnStructurePlan(Generic[T]):
    """A reusable plan for reading rows from named columns.

    The plan resolves field aliases, defaults and structure hooks once. Running
    the plan reads each cell directly from its column; it never materializes a
    dictionary per row.

    ..  versionadded:: 26.2.0
    """

    def __init__(self, cl: type[T], converter: Any) -> None:
        self.cl, origin, typevar_map = _adapt_target(cl)
        if attrs_has(cl):
            attribs = self.cl.__attrs_attrs__
            if any(isinstance(attribute.type, str) for attribute in attribs):
                resolve_types(origin or self.cl)
        attribs = adapted_fields(self.cl)

        use_alias = getattr(converter, "use_alias", False)
        prefer_attribute_converters = converter._prefer_attrib_converters
        type_overrides = getattr(converter, "type_overrides", {})

        fields: list[_ColumnField] = []
        for attribute in attribs:
            name = attribute.name
            override = _annotated_override_or_default(attribute.type, neutral)
            if attribute.type in type_overrides:
                override = type_overrides[attribute.type]

            if override.omit or (override.omit is None and not attribute.init):
                continue

            field_type = attribute.type
            if isinstance(field_type, TypeVar):
                field_type = typevar_map.get(field_type.__name__, field_type)
            elif is_generic(field_type) and not is_annotated(field_type):
                field_type = deep_copy_with(field_type, typevar_map, self.cl)

            hook = (
                override.struct_hook
                if override.struct_hook is not None
                else find_structure_handler(
                    attribute,
                    field_type,
                    converter,
                    prefer_attribute_converters,
                )
            )
            if hook == raise_error:
                hook = None

            alias = name if not use_alias else attribute.alias
            if override.rename is not None:
                alias = override.rename

            has_default = attribute.default is not NOTHING
            optional = _is_optional_field(field_type)
            required = not has_default and not optional
            fields.append(
                _ColumnField(
                    name=name,
                    alias=alias,
                    type=field_type,
                    required=required,
                    optional=optional,
                    init=attribute.init,
                    kw_only=attribute.kw_only,
                    parameter_name=attribute.alias,
                    hook=hook,
                    has_default=has_default,
                )
            )

        self._fields = tuple(fields)
        self._converter = converter

    @property
    def fields(self) -> tuple[str, ...]:
        """The field names included in this plan."""
        return tuple(field.name for field in self._fields)

    @property
    def columns(self) -> tuple[str, ...]:
        """The input column names used by this plan."""
        return tuple(field.alias for field in self._fields)

    def prepare(
        self,
        columns: Mapping[str, Any],
        *,
        forbid_extra_columns: bool | None = None,
    ) -> _PreparedColumns:
        """Validate column presence and lengths and bind a batch.

        :meta private:
        """
        if not isinstance(columns, Mapping):
            raise TypeError("columns must be a mapping of column names to sequences")

        shape_errors: list[Exception] = []
        present: dict[str, Any] = {}
        first_length: int | None = None
        for column in columns.values():
            if _is_indexable_column(column):
                first_length = len(column)
                break

        for field in self._fields:
            if field.alias in columns:
                column = columns[field.alias]
                if not _is_indexable_column(column):
                    shape_errors.append(InvalidColumnError(field.alias))
                    continue
                present[field.name] = column
                if first_length is None:
                    first_length = len(column)
            elif field.required:
                shape_errors.append(
                    MissingColumnError(self.cl, field.name, field.alias)
                )

        valid_columns = [
            column for column in columns.values() if _is_indexable_column(column)
        ]
        if valid_columns:
            expected_length = first_length if first_length is not None else 0
            for column_name, column in columns.items():
                if _is_indexable_column(column) and (
                    len(column) != expected_length
                ):
                    shape_errors.append(
                        ColumnLengthError(
                            column_name,
                            expected_length,
                            len(column),
                        )
                    )
        else:
            expected_length = first_length if first_length is not None else 0

        if forbid_extra_columns is None:
            forbid_extra_columns = getattr(
                self._converter, "forbid_extra_keys", False
            )
        extra_columns = set(columns) - set(self.columns)
        if forbid_extra_columns and extra_columns:
            shape_errors.append(ForbiddenExtraKeysError("", self.cl, extra_columns))

        return _PreparedColumns(
            plan=self,
            columns=present,
            row_count=expected_length,
            shape_errors=tuple(shape_errors),
        )

    def structure(
        self,
        columns: Mapping[str, Any],
        *,
        on_error: ErrorMode | str = ErrorMode.STOP,
        forbid_extra_columns: bool | None = None,
    ) -> ColumnarResult[T]:
        """Structure all columnar rows into a single result."""
        prepared = self.prepare(
            columns, forbid_extra_columns=forbid_extra_columns
        )
        return prepared.structure(on_error=_normalize_error_mode(on_error))

    def __call__(
        self,
        columns: Mapping[str, Any],
        *,
        on_error: ErrorMode | str = ErrorMode.STOP,
        forbid_extra_columns: bool | None = None,
    ) -> ColumnarResult[T]:
        """Structure a column-oriented batch."""
        return self.structure(
            columns,
            on_error=on_error,
            forbid_extra_columns=forbid_extra_columns,
        )

    def iter_batches(
        self,
        columns: Mapping[str, Any],
        batch_size: int | None = None,
        *,
        on_error: ErrorMode | str = ErrorMode.STOP,
        forbid_extra_columns: bool | None = None,
    ) -> Iterator[ColumnarResult[T]]:
        """Iterate structured rows in batches."""
        prepared = self.prepare(
            columns, forbid_extra_columns=forbid_extra_columns
        )
        yield from prepared.iter_batches(
            batch_size=batch_size,
            on_error=_normalize_error_mode(on_error),
        )

    def iter_rows(
        self,
        columns: Mapping[str, Any],
        *,
        forbid_extra_columns: bool | None = None,
    ) -> Iterator[T]:
        """Iterate structured rows one at a time, stopping at the first error."""
        prepared = self.prepare(
            columns, forbid_extra_columns=forbid_extra_columns
        )
        yield from prepared.iter_rows()


@dataclass(frozen=True)
class _PreparedColumns(Generic[T]):
    plan: ColumnStructurePlan[T]
    columns: Mapping[str, Any]
    row_count: int
    shape_errors: tuple[Exception, ...]

    def _raise_shape_errors(self) -> None:
        if self.shape_errors:
            raise ColumnarValidationError(
                "While structuring columnar data",
                self.shape_errors,
                self.plan.cl,
            )

    def structure(self, *, on_error: ErrorMode) -> ColumnarResult[T]:
        if self.shape_errors:
            if on_error is ErrorMode.STOP:
                self._raise_shape_errors()
            return ColumnarResult(
                (),
                self.shape_errors,
                self.row_count,
                (),
            )

        objects, row_indices, errors = self._process(
            0,
            self.row_count,
            on_error,
        )
        return ColumnarResult(
            tuple(objects),
            tuple(errors),
            self.row_count,
            tuple(row_indices),
        )

    def iter_batches(
        self, *, batch_size: int | None, on_error: ErrorMode
    ) -> Iterator[ColumnarResult[T]]:
        if self.shape_errors:
            if on_error is ErrorMode.STOP:
                self._raise_shape_errors()
            yield ColumnarResult((), self.shape_errors, self.row_count, ())
            return

        if batch_size is None:
            batch_size = self.row_count
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        for start in range(0, self.row_count, batch_size):
            stop = min(start + batch_size, self.row_count)
            objects, row_indices, errors = self._process(
                start,
                stop,
                on_error,
            )
            yield ColumnarResult(
                tuple(objects),
                tuple(errors),
                stop - start,
                tuple(row_indices),
            )

    def _process(
        self, start: int, stop: int, on_error: ErrorMode
    ) -> tuple[list[T], list[int], list[Exception]]:
        objects: list[T] = []
        row_indices: list[int] = []
        errors: list[Exception] = []
        for row_index in range(start, stop):
            try:
                objects.append(self._structure_row(row_index))
                row_indices.append(row_index)
            except ColumnarValidationError as error:
                if on_error is ErrorMode.STOP:
                    raise
                for subexc in error.exceptions:
                    if not hasattr(subexc, "__notes__"):
                        note = ColumnRowValidationNote(
                            f"Structuring class {self.plan.cl.__qualname__} "
                            f"@ row {row_index}",
                            row_index,
                        )
                        subexc.__notes__ = [note]
                    errors.append(subexc)
            except Exception as error:
                note = ColumnRowValidationNote(
                    f"Structuring class {self.plan.cl.__qualname__} @ row {row_index}",
                    row_index,
                )
                error.__notes__ = getattr(error, "__notes__", []) + [note]
                if on_error is ErrorMode.STOP:
                    raise ColumnarValidationError(
                        "While structuring columnar data",
                        [error],
                        self.plan.cl,
                    ) from error
                errors.append(error)
        return objects, row_indices, errors

    def iter_rows(self) -> Iterator[T]:
        if self.shape_errors:
            self._raise_shape_errors()
        for row_index in range(self.row_count):
            yield self._structure_row(row_index)

    def _structure_row(self, row_index: int) -> T:
        positional: list[Any] = []
        keyword: dict[str, Any] = {}
        active_positional = True
        cell_errors: list[Exception] = []

        for field in self.plan._fields:
            column = self.columns.get(field.name)
            value: Any = None

            if column is not None:
                try:
                    value = self._convert_field(field, column[row_index])
                except Exception as error:
                    note = ColumnValidationNote(
                        f"Structuring class {self.plan.cl.__qualname__} "
                        f"@ row {row_index}, column {field.alias!r}",
                        row_index,
                        field.name,
                        field.alias,
                        field.type,
                    )
                    error.__notes__ = getattr(error, "__notes__", []) + [note]
                    cell_errors.append(error)
                    continue
            elif field.has_default:
                active_positional = False
                continue
            elif field.optional:
                value = None
            else:
                error = MissingColumnError(self.plan.cl, field.name, field.alias)
                note = ColumnValidationNote(
                    f"Structuring class {self.plan.cl.__qualname__} "
                    f"@ row {row_index}, column {field.alias!r}",
                    row_index,
                    field.name,
                    field.alias,
                    field.type,
                )
                error.__notes__ = getattr(error, "__notes__", []) + [note]
                cell_errors.append(error)
                continue

            if active_positional and not field.kw_only:
                positional.append(value)
            else:
                keyword[field.parameter_name] = value

        if cell_errors:
            raise ColumnarValidationError(
                "While structuring columnar data",
                cell_errors,
                self.plan.cl,
            )

        try:
            if keyword:
                return self.plan.cl(*positional, **keyword)
            return self.plan.cl(*positional)
        except Exception as error:
            note = ColumnRowValidationNote(
                f"Structuring class {self.plan.cl.__qualname__} @ row {row_index}",
                row_index,
            )
            error.__notes__ = getattr(error, "__notes__", []) + [note]
            raise ColumnarValidationError(
                "While structuring columnar data",
                [error],
                self.plan.cl,
            ) from error

    def _convert_field(self, field: _ColumnField, value: Any) -> Any:
        if field.hook is None:
            return value
        if field.hook == self.plan._converter._structure_call:
            return field.type(value)
        return field.hook(value, field.type)


def _adapt_target(cl: Any) -> tuple[Any, Any, dict[str, Any]]:
    if not has_with_generic(cl):
        raise TypeError("Column structure plans require an attrs class or dataclass")

    origin = get_origin(cl)
    typevar_map: dict[str, Any] = {}
    target = cl
    if is_generic(cl):
        typevar_map = generate_mapping(cl, typevar_map)
        if origin is not None:
            target = origin

    for base in getattr(target, "__orig_bases__", ()):
        if is_generic(base) and not str(base).startswith("typing.Generic"):
            typevar_map = generate_mapping(base, typevar_map)
            break

    return target, origin, typevar_map


def _is_optional_field(field_type: Any) -> bool:
    if is_annotated(field_type):
        return is_optional(get_args(field_type)[0])
    return is_optional(field_type)


def _is_indexable_column(value: Any) -> bool:
    return (
        hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
        and not isinstance(value, (str, bytes, Mapping))
    )


def _normalize_error_mode(mode: ErrorMode | str) -> ErrorMode:
    if isinstance(mode, ErrorMode):
        return mode
    normalized = mode.lower()
    if normalized in ("stop", "raise"):
        return ErrorMode.STOP
    if normalized in ("collect", "continue"):
        return ErrorMode.COLLECT
    raise ValueError("on_error must be 'stop' or 'collect'")


def make_column_structure_plan(
    converter: Any, cl: type[T]
) -> ColumnStructurePlan[T]:
    """Create a column access plan using *converter*."""
    return ColumnStructurePlan(cl, converter)


def structure_columns(
    converter: Any,
    columns: Mapping[str, Any],
    cl: type[T],
    *,
    on_error: ErrorMode | str = ErrorMode.STOP,
    forbid_extra_columns: bool | None = None,
) -> ColumnarResult[T]:
    """Structure column-oriented data using *converter*."""
    return make_column_structure_plan(converter, cl).structure(
        columns,
        on_error=on_error,
        forbid_extra_columns=forbid_extra_columns,
    )
