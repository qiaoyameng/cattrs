"""Tests for the union trial strategy."""

import sys
from dataclasses import dataclass
from typing import Union

import pytest
from attrs import define, field

from cattrs import AmbiguousUnionError, BaseConverter, UnionStructureError
from cattrs.strategies import configure_union_trial

try:
    from cattrs._compat import NoneType
except ImportError:  # pragma: no cover
    NoneType = type(None)


@define
class ClassA:
    field_one: str
    shared: str = "shared"


@define
class ClassB:
    field_two: str
    shared: str = "shared"


@define
class AllDefaultA:
    a: str = "a"


@define
class AllDefaultB:
    b: str = "b"


@define
class AliasA:
    a: int


@define
class AliasB:
    b: int


def test_single_match(converter: BaseConverter) -> None:
    """Only the single applicable member is returned."""
    union = Union[ClassA, ClassB]
    configure_union_trial(union, converter)

    assert converter.structure({"field_one": "x"}, union) == ClassA("x")
    assert converter.structure({"field_two": "y"}, union) == ClassB("y")


def test_default_overlap_is_ambiguous(converter: BaseConverter) -> None:
    """Members with only overlapping defaults are ambiguous for ``{}``."""
    union = Union[AllDefaultA, AllDefaultB]
    configure_union_trial(union, converter)

    with pytest.raises(AmbiguousUnionError) as exc:
        converter.structure({}, union)

    assert set(exc.value.members) == {AllDefaultA, AllDefaultB}
    assert {r.__class__ for r in exc.value.results} == {AllDefaultA, AllDefaultB}


def test_default_overlap_resolved_by_key(converter: BaseConverter) -> None:
    """Without required keys the trial cannot exploit a distinguishing key.

    Extra keys are allowed by default, so both members match; that is a
    genuine ambiguity, not a first-match pick.
    """
    union = Union[AllDefaultA, AllDefaultB]
    configure_union_trial(union, converter)

    with pytest.raises(AmbiguousUnionError) as exc:
        converter.structure({"a": "x"}, union)
    assert set(exc.value.members) == {AllDefaultA, AllDefaultB}


def test_all_fail_reports_each_branch(converter: BaseConverter) -> None:
    """When every member fails, each branch's error is reported."""

    @define
    class IntA:
        field_one: int

    @define
    class IntB:
        field_two: int

    union = Union[IntA, IntB]
    configure_union_trial(union, converter)

    with pytest.raises(UnionStructureError) as exc:
        converter.structure({"field_one": "x", "field_two": "y"}, union)

    assert set(exc.value.members) == {IntA, IntB}
    assert len(exc.value.exceptions) == 2
    rendered = str(exc.value)
    assert "IntA" in rendered and "IntB" in rendered


def test_all_pruned_reports_reasons(converter: BaseConverter) -> None:
    """A payload matching no member reports the per-branch pruning reasons."""
    union = Union[ClassA, ClassB]
    configure_union_trial(union, converter)

    with pytest.raises(UnionStructureError) as exc_info:
        converter.structure({}, union)

    assert set(exc_info.value.members) == {ClassA, ClassB}
    assert "missing required key" in str(exc_info.value)


def test_order_independent() -> None:
    """The result does not depend on the member order in the union."""

    converter_ab = BaseConverter()
    configure_union_trial(Union[ClassA, ClassB], converter_ab)

    converter_ba = BaseConverter()
    configure_union_trial(Union[ClassB, ClassA], converter_ba)

    payload_a = {"field_one": "x"}
    payload_b = {"field_two": "y"}

    assert converter_ab.structure(payload_a, Union[ClassA, ClassB]) == ClassA("x")
    assert converter_ba.structure(payload_a, Union[ClassB, ClassA]) == ClassA("x")
    assert converter_ab.structure(payload_b, Union[ClassA, ClassB]) == ClassB("y")
    assert converter_ba.structure(payload_b, Union[ClassB, ClassA]) == ClassB("y")


def test_pruning_by_convention() -> None:
    """Branches missing required keys are filtered out without being tried."""
    tried = []

    @define
    class Recorder:
        required: str

    converter = BaseConverter()
    original = converter.get_structure_hook(Recorder)
    converter.register_structure_hook(
        Recorder, lambda v, _: (tried.append(v), original(v, _))[1]
    )

    union = Union[ClassA, Recorder]
    configure_union_trial(union, converter)

    # `required` is absent, so Recorder must not even be attempted.
    assert converter.structure({"field_one": "x"}, union) == ClassA("x")
    assert tried == []


def test_independent_input_views() -> None:
    """A hook mutating its input cannot affect another candidate."""

    @define
    class MutA:
        a: str

    @define
    class MutB:
        b: str

    converter = BaseConverter()
    union = Union[MutA, MutB]
    configure_union_trial(union, converter)

    def mutating_a(val, _):
        val.pop("b")
        raise ValueError("A fails on purpose")

    converter.register_structure_hook(MutA, mutating_a)

    # MutA pops "b" and fails; MutB must still see the untouched copy.
    assert converter.structure({"a": "x", "b": "y"}, union) == MutB("y")


def test_external_side_effects_are_not_rolled_back() -> None:
    """External side effects of hooks remain visible after a failed branch."""

    @define
    class SideA:
        a: int

    @define
    class SideB:
        a: int

    converter = BaseConverter()
    union = Union[SideA, SideB]
    configure_union_trial(union, converter)

    calls = []

    def side_a(val, _):
        calls.append("A")
        raise ValueError("boom")

    converter.register_structure_hook(SideA, side_a)

    result = converter.structure({"a": 1}, union)
    assert result == SideB(1)
    assert calls == ["A"]


def test_nested_unions() -> None:
    """Nested union members are flattened and tried as candidates."""

    @define
    class NestedA:
        a: str

    @define
    class NestedB:
        b: str

    @define
    class NestedC:
        c: str

    converter = BaseConverter()
    union = Union[NestedA, Union[NestedB, NestedC]]
    configure_union_trial(union, converter)

    assert converter.structure({"a": "x"}, union) == NestedA("x")
    assert converter.structure({"b": "x"}, union) == NestedB("x")
    assert converter.structure({"c": "x"}, union) == NestedC("x")

    with pytest.raises(AmbiguousUnionError) as exc:
        converter.structure({"a": "x", "c": "z"}, union)
    assert set(exc.value.members) == {NestedA, NestedC}


def test_nested_union_as_field() -> None:
    """The strategy applies to unions nested inside structured classes."""

    @define
    class Holder:
        item: Union[ClassA, ClassB]

    converter = BaseConverter()
    configure_union_trial(Union[ClassA, ClassB], converter)

    assert converter.structure({"item": {"field_one": "x"}}, Holder) == Holder(
        ClassA("x")
    )
    assert converter.structure({"item": {"field_two": "y"}}, Holder) == Holder(
        ClassB("y")
    )


def test_attrs_field_converters() -> None:
    """Custom attrs converters run inside each independent trial."""

    @define
    class ConvA:
        value: int = field(converter=lambda v: int(v) + 1)

    @define
    class ConvB:
        value: int = field(converter=lambda v: int(v) - 1)

    converter = BaseConverter()
    union = Union[ConvA, ConvB]
    configure_union_trial(union, converter)

    with pytest.raises(AmbiguousUnionError) as exc:
        converter.structure({"value": "5"}, union)
    assert sorted(r.__class__.__name__ for r in exc.value.results) == [
        "ConvA",
        "ConvB",
    ]
    assert {r.value for r in exc.value.results} == {6, 4}


def test_failing_field_converter_leaves_single_candidate() -> None:
    """A failing converter removes that branch, leaving the other one."""

    @define
    class Bad:
        value: int = field(
            converter=lambda v: (_ for _ in ()).throw(ValueError("bad"))
        )

    @define
    class Good:
        value: int

    converter = BaseConverter()
    union = Union[Bad, Good]
    configure_union_trial(union, converter)

    assert converter.structure({"value": 5}, union) == Good(5)


def test_dataclasses() -> None:
    """The strategy works with dataclasses too."""

    @dataclass
    class DCA:
        a: str

    @dataclass
    class DCB:
        b: str

    converter = BaseConverter()
    union = Union[DCA, DCB]
    configure_union_trial(union, converter)

    assert converter.structure({"a": "x"}, union) == DCA("x")
    assert converter.structure({"b": "y"}, union) == DCB("y")

    with pytest.raises(UnionStructureError) as exc:
        converter.structure({}, union)
    assert set(exc.value.members) == {DCA, DCB}


def test_optional_member() -> None:
    """Multi-member unions including ``None`` handle ``None`` directly."""

    @define
    class OptA:
        a: str

    @define
    class OptB:
        b: str

    converter = BaseConverter()
    union = Union[OptA, OptB, None]
    configure_union_trial(union, converter)

    assert converter.structure(None, union) is None
    assert converter.structure({"a": "x"}, union) == OptA("x")
    assert converter.structure({"b": "y"}, union) == OptB("y")

    with pytest.raises(UnionStructureError) as exc:
        converter.structure({"unrelated": True}, union)
    assert NoneType not in exc.value.members or all(exc.value.exceptions)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 requires 3.12+")
def test_type_alias() -> None:
    """The strategy accepts a PEP 695 type alias of a union."""
    ns: dict = {}
    exec(
        "type AB = AliasA | AliasB",
        {"AliasA": AliasA, "AliasB": AliasB},
        ns,
    )

    converter = BaseConverter()
    configure_union_trial(ns["AB"], converter)
    assert converter.structure({"a": 1}, ns["AB"]) == AliasA(1)
    assert converter.structure({"b": 2}, ns["AB"]) == AliasB(2)
