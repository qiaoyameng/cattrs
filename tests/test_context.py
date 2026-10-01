from dataclasses import dataclass
from typing import Union

import pytest
from attrs import define

from cattrs import BaseConverter, Converter, INHERIT_CONTEXT
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override


@define
class Trace:
    name: str


@define
class Inner:
    trace: Trace
    inner_only: str


@define
class Other:
    trace: Trace
    other_only: str


@dataclass
class Wrapper:
    trace: Trace
    items: list[Inner]
    by_name: dict[str, Inner]
    choice: Union[Inner, Other]


@define
class Outer:
    trace: Trace
    wrapper: Wrapper


InnerOrOther = Union[Inner, Other]


def make_outer(prefix: str, choice_cls: type[Inner] | type[Other]) -> Outer:
    if choice_cls is Inner:
        choice: Inner | Other = Inner(Trace(f"{prefix}-choice"), f"{prefix}-choice")
    else:
        choice = Other(Trace(f"{prefix}-choice"), f"{prefix}-choice")
    return Outer(
        Trace(f"{prefix}-outer"),
        Wrapper(
            Trace(f"{prefix}-wrapper"),
            [Inner(Trace(f"{prefix}-item"), f"{prefix}-item")],
            {"x": Inner(Trace(f"{prefix}-mapped"), f"{prefix}-mapped")},
            choice,
        ),
    )


def make_raw_outer(prefix: str, choice: str) -> dict[str, object]:
    choice_data: dict[str, object] = {"trace": f"{prefix}-choice"}
    if choice == "inner":
        choice_data["inner_only"] = f"{prefix}-choice"
    else:
        choice_data["other_only"] = f"{prefix}-choice"
    return {
        "trace": f"{prefix}-outer",
        "wrapper": {
            "trace": f"{prefix}-wrapper",
            "items": [
                {"trace": f"{prefix}-item", "inner_only": f"{prefix}-item"}
            ],
            "by_name": {
                "x": {"trace": f"{prefix}-mapped", "inner_only": f"{prefix}-mapped"}
            },
            "choice": choice_data,
        },
    }


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_context_propagation_and_reentrant_replacement(converter_cls):
    converter = converter_cls()
    events: list[tuple[str, object]] = []
    state = {"reentered": False}

    def unstructure_trace(value: Trace, context: object) -> str:
        events.append((value.name, context))
        return value.name

    def structure_trace(value: str, _: type[Trace], context: object) -> Trace:
        events.append((value, context))
        return Trace(value)

    def unstructure_union(value: Inner | Other, context: object):
        events.append(("union", context))
        selected = Inner if isinstance(value, Inner) else Other
        if not state["reentered"]:
            state["reentered"] = True
            converter.unstructure(make_outer("b", Other), context="tenant-b")
        return converter.unstructure(
            value, selected, context=INHERIT_CONTEXT
        )

    def structure_union(value: dict[str, object], _: object, context: object):
        events.append(("union", context))
        selected = Inner if "inner_only" in value else Other
        if not state["reentered"]:
            state["reentered"] = True
            converter.structure(
                make_raw_outer("b", "other"), Outer, context="tenant-b"
            )
        return converter.structure(value, selected, context=INHERIT_CONTEXT)

    converter.register_unstructure_hook(Trace, unstructure_trace)
    converter.register_structure_hook(Trace, structure_trace)
    converter.register_unstructure_hook(InnerOrOther, unstructure_union)
    converter.register_structure_hook(InnerOrOther, structure_union)

    unstructured = converter.unstructure(
        make_outer("a", Inner), context="tenant-a"
    )
    assert unstructured == make_raw_outer("a", "inner")

    expected_unstructure = [
        ("a-outer", "tenant-a"),
        ("a-wrapper", "tenant-a"),
        ("a-item", "tenant-a"),
        ("a-mapped", "tenant-a"),
        ("union", "tenant-a"),
        ("b-outer", "tenant-b"),
        ("b-wrapper", "tenant-b"),
        ("b-item", "tenant-b"),
        ("b-mapped", "tenant-b"),
        ("union", "tenant-b"),
        ("b-choice", "tenant-b"),
        ("a-choice", "tenant-a"),
    ]
    assert events == expected_unstructure

    events.clear()
    state["reentered"] = False
    structured = converter.structure(
        make_raw_outer("a", "inner"), Outer, context="tenant-a"
    )
    assert structured == make_outer("a", Inner)
    assert events == expected_unstructure


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_legacy_two_argument_structure_hook_remains_supported(converter_cls):
    converter = converter_cls()

    def unstructure_str(value: str) -> str:
        return value.upper()

    def structure_str(value: str, type_: type[str]) -> str:
        assert type_ is str
        return value.upper()

    converter.register_unstructure_hook(str, unstructure_str)
    converter.register_structure_hook(str, structure_str)
    assert converter.unstructure("abc", context="tenant-a") == "ABC"
    assert converter.structure("abc", str, context="tenant-a") == "ABC"
    assert converter.get_unstructure_hook(str) is unstructure_str
    assert converter.get_structure_hook(str) is structure_str


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_keyword_only_context_hooks(converter_cls):
    converter = converter_cls()

    @define
    class Value:
        name: str

    converter.register_unstructure_hook(
        Value, lambda value, *, context: (value.name, context)
    )
    converter.register_structure_hook(
        Value, lambda value, _, *, context: (value, context)
    )

    assert converter.unstructure(Value("name"), context="tenant") == (
        "name",
        "tenant",
    )
    assert converter.structure("name", Value, "tenant") == (
        "name",
        "tenant",
    )


def test_contextual_field_override():
    converter = Converter()

    @define
    class Holder:
        value: str

    converter.register_unstructure_hook_factory(
        lambda t: t is Holder,
        lambda t: make_dict_unstructure_fn(
            t, converter, value=override(unstruct_hook=lambda value, context: (value, context))
        ),
    )
    converter.register_structure_hook_factory(
        lambda t: t is Holder,
        lambda t: make_dict_structure_fn(
            t,
            converter,
            value=override(struct_hook=lambda value, _, context: (value, context)),
        ),
    )

    assert converter.unstructure(Holder("name"), context="tenant") == {
        "value": ("name", "tenant")
    }
    assert converter.structure({"value": "name"}, Holder, context="tenant").value == (
        "name",
        "tenant",
    )
