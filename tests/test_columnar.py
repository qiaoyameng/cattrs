from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Optional

import attr
import pytest

from cattrs.columnar import structure_columns

from cattrs import (
    BaseConverter,
    ColumnLengthError,
    ColumnarValidationError,
    Converter,
    ErrorMode,
    MissingColumnError,
    transform_error,
)
from cattrs.gen import override


@attr.define
class AttrsPerson:
    name: str
    age: int
    nickname: Optional[str] = None
    active: bool = True


@dataclass
class DataclassPerson:
    name: str
    age: int
    nickname: str | None = None
    active: bool = True


@attr.define
class NestedChild:
    value: int


@attr.define
class NestedParent:
    child: NestedChild


@pytest.mark.parametrize("cl", [AttrsPerson, DataclassPerson])
def test_structure_columns(converter: BaseConverter, cl):
    result = converter.structure_columns(
        {"name": ["Ada", "Lin"], "age": [20, "21"]},
        cl,
    )

    assert result.ok
    assert result.objects == (cl("Ada", 20), cl("Lin", 21))
    assert result.row_count == 2


@pytest.mark.parametrize("cl", [AttrsPerson, DataclassPerson])
def test_missing_optional_and_default_columns(converter: BaseConverter, cl):
    result = converter.structure_columns(
        {"name": ["Ada"], "age": [20]},
        cl,
    )

    assert result.objects == (cl("Ada", 20, None, True),)


def test_optional_without_default_becomes_none(genconverter: Converter):
    @attr.define
    class OptionalOnly:
        value: Optional[int]

    result = genconverter.structure_columns(
        {"other": [1, 2]},
        OptionalOnly,
    )

    assert result.objects == (OptionalOnly(None), OptionalOnly(None))


def test_attrs_alias_and_annotated_rename():
    @attr.define
    class Aliased:
        full_name: str = attr.field(alias="fullName")
        age_years: Annotated[int, override(rename="ageYears")]

    converter = Converter(use_alias=True)
    result = converter.structure_columns(
        {"fullName": ["Ada"], "ageYears": ["20"]},
        Aliased,
        on_error=ErrorMode.COLLECT,
    )

    assert result.objects == (Aliased("Ada", 20),)


def test_converter_use_alias():
    @attr.define
    class Aliased:
        full_name: str = attr.field(alias="fullName")

    converter = Converter(use_alias=True)
    result = converter.structure_columns(
        {"fullName": ["Ada"]},
        Aliased,
    )

    assert result.objects == (Aliased("Ada"),)


def test_custom_hook_and_plan_reuse(genconverter: Converter):
    calls = []

    @attr.define
    class Hooked:
        value: int

    def hook(value, _):
        calls.append(value)
        return value + 1

    genconverter.register_structure_hook(int, hook)
    plan = genconverter.make_column_structure_plan(Hooked)

    first = plan({"value": [1]})
    second = plan.structure({"value": [2, 3]})

    assert first.objects == (Hooked(2),)
    assert second.objects == (Hooked(3), Hooked(4),)
    assert calls == [1, 2, 3]


def test_iter_rows_stops_at_first_invalid_cell(converter: BaseConverter):
    rows = converter.make_column_structure_plan(AttrsPerson).iter_rows(
        {"name": ["Ada", "Grace"], "age": ["20", "bad"]}
    )

    assert next(rows) == AttrsPerson("Ada", 20)
    with pytest.raises(ColumnarValidationError) as excinfo:
        next(rows)

    assert transform_error(excinfo.value) == [
        "invalid value for type, expected int @ $[1].age"
    ]


def test_iter_batches_collects_errors_in_row_order(genconverter: Converter):
    batches = list(
        genconverter.iter_column_batches(
            {"name": ["Ada", None, "Lin"], "age": [20, "bad", 22]},
            AttrsPerson,
            1,
            on_error="collect",
        )
    )

    assert [batch.objects for batch in batches] == [
        (AttrsPerson("Ada", 20),),
        (),
        (AttrsPerson("Lin", 22),),
    ]
    assert [len(batch.errors) for batch in batches] == [0, 1, 0]
    assert [batch.row_count for batch in batches] == [1, 1, 1]
    assert [batch.row_indices for batch in batches] == [(0,), (), (2,)]


def test_collect_invalid_cell_reports_row_and_path(converter: BaseConverter):
    result = converter.structure_columns(
        {"name": ["Ada", "Grace"], "age": ["bad", 2]},
        AttrsPerson,
        on_error=ErrorMode.COLLECT,
    )

    assert not result.ok
    assert result.objects == (AttrsPerson("Grace", 2),)
    assert transform_error(
        ColumnarValidationError("", result.errors, AttrsPerson)
    ) == ["invalid value for type, expected int @ $[0].age"]


def test_missing_required_column_collects(genconverter: Converter):
    result = genconverter.structure_columns(
        {"age": [1]},
        AttrsPerson,
        on_error=ErrorMode.COLLECT,
    )

    assert result.objects == ()
    assert isinstance(result.errors[0], MissingColumnError)
    assert transform_error(
        ColumnarValidationError("", result.errors, AttrsPerson)
    ) == ["required field missing @ $.name"]


def test_missing_required_column_stops(genconverter: Converter):
    with pytest.raises(ColumnarValidationError) as excinfo:
        genconverter.structure_columns({"age": [1]}, AttrsPerson)

    assert isinstance(excinfo.value.exceptions[0], MissingColumnError)


def test_length_mismatch_stops(converter: BaseConverter):
    with pytest.raises(ColumnarValidationError) as excinfo:
        converter.structure_columns(
            {"name": ["Ada"], "age": [1, 2]},
            AttrsPerson,
        )

    assert isinstance(excinfo.value.exceptions[0], ColumnLengthError)


def test_length_mismatch_collects(genconverter: Converter):
    result = genconverter.structure_columns(
        {"name": ["Ada"], "age": [1, 2]},
        AttrsPerson,
        on_error=ErrorMode.COLLECT,
    )

    assert result.objects == ()
    assert isinstance(result.errors[0], ColumnLengthError)


def test_invalid_column_type(genconverter: Converter):
    result = genconverter.structure_columns(
        {"name": "Ada", "age": [1]},
        AttrsPerson,
        on_error=ErrorMode.COLLECT,
    )

    assert result.objects == ()
    assert result.errors[0].column_name == "name"


def test_forbid_extra_columns(genconverter: Converter):
    with pytest.raises(ColumnarValidationError):
        genconverter.structure_columns(
            {"name": ["Ada"], "age": [1], "unexpected": [True]},
            AttrsPerson,
            forbid_extra_columns=True,
        )


def test_module_level_function_uses_supplied_converter():
    converter = BaseConverter()
    result = structure_columns(
        converter,
        {"name": ["Ada"], "age": [1]},
        AttrsPerson,
    )

    assert result.objects == (AttrsPerson("Ada", 1),)


def test_plan_rejects_non_classes(genconverter: Converter):
    with pytest.raises(TypeError):
        genconverter.make_column_structure_plan(dict[str, int])


def test_columns_are_read_by_position_without_row_dicts(monkeypatch, genconverter):
    class RecordingDict(dict):
        def __init__(self, *args, **kwargs):
            raise AssertionError("columnar structuring must not build row dicts")

    @attr.define
    class Simple:
        value: int

    monkeypatch.setattr(genconverter, "_dict_factory", RecordingDict)
    result = genconverter.structure_columns({"value": [1, "2"]}, Simple)

    assert [item.value for item in result.objects] == [1, 2]


def test_nested_custom_field_path():
    genconverter = Converter(detailed_validation=True)
    result = genconverter.structure_columns(
        {"child": [{"value": "bad"}]},
        NestedParent,
        on_error=ErrorMode.COLLECT,
    )

    assert transform_error(
        ColumnarValidationError("", result.errors, NestedParent)
    ) == ["invalid value for type, expected int @ $[0].child.value"]


def test_batch_size_must_be_positive(genconverter: Converter):
    plan = genconverter.make_column_structure_plan(AttrsPerson)
    with pytest.raises(ValueError):
        list(
            plan.iter_batches(
                {"name": ["Ada"], "age": [1]},
                0,
                on_error=ErrorMode.COLLECT,
            )
        )


def test_collect_result_keeps_original_row_indices(genconverter: Converter):
    result = genconverter.structure_columns(
        {"name": ["Ada", "Grace", "Lin"], "age": ["bad", 2, 3]},
        AttrsPerson,
        on_error=ErrorMode.COLLECT,
    )

    assert result.row_indices == (1, 2)
    assert result.objects == (AttrsPerson("Grace", 2), AttrsPerson("Lin", 3))


def test_missing_middle_default_column_keeps_later_value(genconverter: Converter):
    @attr.define
    class Defaults:
        first: int
        middle: int = 2
        last: int = 3

    result = genconverter.structure_columns(
        {"first": [1], "last": [4]},
        Defaults,
    )

    assert result.objects == (Defaults(1, 2, 4),)


def test_constructor_error_reports_row(genconverter: Converter):
    def non_negative(instance, attribute, value):
        if value < 0:
            raise ValueError("must be non-negative")

    @attr.define
    class Validated:
        value: int = attr.field(validator=non_negative)

    result = genconverter.structure_columns(
        {"value": [1, -1, 2]},
        Validated,
        on_error=ErrorMode.COLLECT,
    )

    assert result.row_indices == (0, 2)
    assert transform_error(
        ColumnarValidationError("", result.errors, Validated)
    ) == ["invalid value @ $[1]"]


def test_annotated_optional_without_default(genconverter: Converter):
    @attr.define
    class OptionalAnnotated:
        value: Annotated[Optional[int], "metadata"]

    result = genconverter.structure_columns(
        {"other": [1]},
        OptionalAnnotated,
    )

    assert result.objects == (OptionalAnnotated(None),)


def test_raise_is_stop_alias(genconverter: Converter):
    with pytest.raises(ColumnarValidationError):
        genconverter.structure_columns(
            {"name": ["Ada"], "age": ["bad"]},
            AttrsPerson,
            on_error="raise",
        )
