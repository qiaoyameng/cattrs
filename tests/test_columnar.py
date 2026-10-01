from dataclasses import dataclass, field
from typing import Optional

import attrs
import pytest

from cattrs import (
    ColumnLengthError,
    ColumnValidationError,
    Converter,
    MissingColumnError,
    transform_error,
)
from cattrs.gen import override as override_attr
from cattrs.columnar import make_column_structure_plan


@attrs.define
class AttrsRow:
    name: str
    age: int
    nickname: Optional[str] = None


@dataclass
class DataclassRow:
    name: str
    age: int
    nickname: str | None = None


@attrs.define
class AliasRow:
    name: str = attrs.field(alias="full_name")


@attrs.define
class DefaultFactoryRow:
    tags: list[str] = attrs.Factory(list)


@attrs.define
class RequiredDefaultFactoryRow:
    name: str
    tags: list[str] = attrs.Factory(list)


@pytest.mark.parametrize("cl", [AttrsRow, DataclassRow])
def test_structure_columns(converter, cl):
    columns = {"name": ["Ada", "Grace"], "age": [36, 85]}

    assert converter.structure_columns(columns, cl) == [
        cl("Ada", 36),
        cl("Grace", 85),
    ]


def test_column_alias():
    converter = Converter(use_alias=True)
    result = converter.structure_columns({"full_name": ["Ada"]}, AliasRow)
    assert result == [AliasRow(full_name="Ada")]


def test_optional_and_default_factory(converter):
    result = converter.structure_columns({"name": ["Ada"], "age": [36]}, AttrsRow)
    assert result == [AttrsRow("Ada", 36)]

    assert converter.structure_columns({}, DefaultFactoryRow) == []
    result = converter.structure_columns(
        {"name": ["Ada"]}, RequiredDefaultFactoryRow
    )
    assert result == [RequiredDefaultFactoryRow("Ada", [])]


def test_custom_hook(converter):
    converter.register_structure_hook(int, lambda value, _: int(value) + 1)
    result = converter.structure_columns({"name": ["Ada"], "age": ["36"]}, AttrsRow)
    assert result[0].age == 37


def test_plan_is_reusable(converter):
    plan = converter.make_column_structure_plan(AttrsRow)
    assert plan.required_columns == ("name", "age")
    assert plan.structure({"name": ["Ada"], "age": [36]}) == [AttrsRow("Ada", 36)]
    assert plan.structure({"name": ["Grace"], "age": [85]}) == [AttrsRow("Grace", 85)]


def test_missing_required_column(converter):
    with pytest.raises(ColumnValidationError) as exc:
        converter.structure_columns({"age": [36]}, AttrsRow)

    error = exc.value.exceptions[0]
    assert isinstance(error, MissingColumnError)
    assert error.column_name == "name"
    assert transform_error(exc.value) == [
        "unknown error (Missing column 'name' for field AttrsRow.name) @ $"
    ]


def test_unequal_column_lengths(converter):
    with pytest.raises(ColumnValidationError) as exc:
        converter.structure_columns({"name": ["Ada"], "age": [36, 85]}, AttrsRow)

    assert isinstance(exc.value.exceptions[0], ColumnLengthError)


def test_invalid_cell_reports_row_and_path(converter):
    def reject_none(value, _):
        if value is None:
            raise ValueError("bad name")
        return value

    converter.register_structure_hook(str, reject_none)
    with pytest.raises(ColumnValidationError) as exc:
        converter.structure_columns(
            {"name": [None, "Ada"], "age": [36, 85]}, AttrsRow
        )

    messages = transform_error(exc.value)
    assert any("@ $[0].name" in message for message in messages)

    with pytest.raises(ColumnValidationError) as exc:
        converter.structure_columns(
            {"name": ["Ada", "Grace"], "age": [36, "bad"]}, AttrsRow
        )

    messages = transform_error(exc.value)
    assert messages == [
        "invalid value for type, expected int @ $[1].age"
    ]


def test_collect_mode_reports_all_invalid_cells(converter):
    def reject_none(value, _):
        if value is None:
            raise ValueError("bad name")
        return value

    converter.register_structure_hook(str, reject_none)
    result = converter.structure_columns(
        {"name": [None, "Ada"], "age": [36, "bad"]},
        AttrsRow,
        on_error="collect",
    )

    messages = transform_error(
        ColumnValidationError("column errors", result.errors, AttrsRow)
    )
    assert any("@ $[0].name" in message for message in messages)
    assert any("@ $[1].age" in message for message in messages)


def test_nested_hook_error_keeps_field_path():
    @attrs.define
    class Inner:
        value: int

    @attrs.define
    class Outer:
        inner: Inner

    with pytest.raises(ColumnValidationError) as exc:
        Converter(detailed_validation=True).structure_columns(
            {"inner": [{"value": "bad"}]},
            Outer,
        )

    assert transform_error(exc.value) == [
        "invalid value for type, expected int @ $[0].inner.value"
    ]


def test_collect_mode_keeps_successful_object_order(converter):
    result = converter.structure_columns(
        {"name": ["Ada", "Grace", "Lin"], "age": [36, "bad", 41]},
        AttrsRow,
        on_error="collect",
    )

    assert result.objects == [AttrsRow("Ada", 36), AttrsRow("Lin", 41)]
    assert result.rows == [0, 2]
    assert len(result.errors) == 1
    assert result.errors[0].__notes__[0].index == 1
    assert result.errors[0].__notes__[0].name == "age"


def test_iter_batches(converter):
    plan = converter.make_column_structure_plan(AttrsRow)
    columns = {"name": ["Ada", "Grace", "Lin"], "age": [36, 85, 41]}

    batches = list(plan.iter_batches(columns, batch_size=2))

    assert [batch.rows for batch in batches] == [[0, 1], [2]]
    assert [obj.name for batch in batches for obj in batch.objects] == [
        "Ada",
        "Grace",
        "Lin",
    ]


def test_iter_batches_collects_errors_per_batch(converter):
    plan = converter.make_column_structure_plan(AttrsRow)
    columns = {"name": ["Ada", "Grace"], "age": ["bad", 85]}

    batch = next(plan.iter_batches(columns, batch_size=1, on_error="collect"))

    assert batch.objects == []
    assert batch.rows == []
    assert batch.errors[0].__notes__[0].index == 0


def test_no_per_row_dict_is_required(monkeypatch):
    converter = Converter()
    plan = make_column_structure_plan(AttrsRow, converter)
    real_init = AttrsRow.__init__

    def fail_on_dict(self, *args, **kwargs):
        if not args and kwargs:
            raise AssertionError("rows must not be copied into dicts")
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr(AttrsRow, "__init__", fail_on_dict)
    result = plan.structure({"name": ["Ada"], "age": [36]})
    assert result == [AttrsRow("Ada", 36)]


def test_forbid_extra_keys():
    converter = Converter(forbid_extra_keys=True)
    with pytest.raises(ColumnValidationError):
        converter.structure_columns(
            {"name": ["Ada"], "age": [36], "extra": [True]}, AttrsRow
        )


def test_generic_column_plan(converter):
    from typing import Generic, TypeVar

    type_var = TypeVar("type_var")

    @attrs.define
    class GenericRow(Generic[type_var]):
        value: type_var

    result = converter.structure_columns({"value": ["1", "2"]}, GenericRow[int])
    assert result == [GenericRow(1), GenericRow(2)]


def test_dataclass_default_factory(converter):
    @dataclass
    class Defaulted:
        name: str
        tags: list[str] = field(default_factory=list)

    result = converter.structure_columns({"name": ["Ada"]}, Defaulted)
    assert result == [Defaulted("Ada", [])]


def test_generated_hook_rename_override():
    from cattrs.gen import make_dict_structure_fn

    converter = Converter()
    converter.register_structure_hook(
        AliasRow,
        make_dict_structure_fn(
            AliasRow, converter, name=override_attr(rename="full_name")
        ),
    )

    result = converter.structure_columns({"full_name": ["Ada"]}, AliasRow)
    assert result == [AliasRow(full_name="Ada")]


def test_extra_columns_are_ignored_by_default(converter):
    result = converter.structure_columns(
        {"name": ["Ada"], "age": [36], "extra": [object()]},
        AttrsRow,
    )
    assert result == [AttrsRow("Ada", 36)]


def test_missing_and_length_errors_are_collected_together(converter):
    result = converter.make_column_structure_plan(AttrsRow).structure(
        {
            "name": ["Ada", "Grace"],
            "age": [36],
            "nickname": [None, None],
        },
        on_error="collect",
    )

    assert len(result.errors) == 1
    assert isinstance(result.errors[0], ColumnLengthError)


def test_missing_column_is_separate_from_length_error(converter):
    result = converter.make_column_structure_plan(AttrsRow).structure(
        {"name": ["Ada", "Grace"]},
        on_error="collect",
    )

    assert len(result.errors) == 1
    assert isinstance(result.errors[0], MissingColumnError)
