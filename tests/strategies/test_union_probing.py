"""Tests for the union probing strategy."""

from typing import Optional, Union

import pytest
from attrs import define

from cattrs import BaseConverter, UnionProbingError
from cattrs.strategies import configure_union_probing


@define
class _IntBox:
    x: int


@define
class _ListBox:
    x: list[int]


def test_single_success(converter: BaseConverter) -> None:
    """The sole member which structures successfully is returned."""
    union = Union[_IntBox, _ListBox]
    configure_union_probing(union, converter)

    assert converter.structure({"x": 1}, union) == _IntBox(1)
    assert converter.structure({"x": [1, 2]}, union) == _ListBox([1, 2])


def test_prefilter_skips_missing_required(converter: BaseConverter) -> None:
    """Members missing required fields in the payload are not even tried."""

    @define
    class Needy:
        x: int
        y: int

    @define
    class Modest:
        x: int

    calls = []
    converter.register_structure_hook(Needy, lambda v, c: calls.append(v) or Needy(**v))

    union = Union[Needy, Modest]
    configure_union_probing(union, converter)

    assert converter.structure({"x": 1}, union) == Modest(1)
    assert calls == []

    # Both members can structure this, so the result is ambiguous.
    with pytest.raises(UnionProbingError) as exc_info:
        converter.structure({"x": 1, "y": 2}, union)
    assert set(exc_info.value.successes) == {Needy, Modest}


def test_ambiguous_defaults_overlap(converter: BaseConverter) -> None:
    """Members with fully overlapping defaults are ambiguous, in any order."""

    @define
    class A:
        x: int = 1

    @define
    class B:
        x: int = 2

    for union in (Union[A, B], Union[B, A]):
        configure_union_probing(union, converter)

        with pytest.raises(UnionProbingError) as exc_info:
            converter.structure({}, union)

        assert set(exc_info.value.successes) == {A, B}
        assert exc_info.value.failures == {}


def test_all_fail_reports_reasons(converter: BaseConverter) -> None:
    """When no member succeeds, the error carries each member's exception."""
    union = Union[_IntBox, _ListBox]
    configure_union_probing(union, converter)

    with pytest.raises(UnionProbingError) as exc_info:
        converter.structure({"x": "not an int"}, union)

    exc = exc_info.value
    assert exc.successes == {}
    assert set(exc.failures) == {_IntBox, _ListBox}
    assert all(isinstance(e, Exception) for e in exc.failures.values())


def test_input_views_are_independent(converter: BaseConverter) -> None:
    """A hook mutating its input must not affect other members or the caller."""

    @define
    class A:
        x: int
        y: int

    @define
    class B:
        x: int

    def mutating_hook(val, cl):
        val.pop("y", None)
        raise ValueError("nope")

    seen = []

    def recording_hook(val, cl):
        seen.append(dict(val))
        return B(val["x"])

    converter.register_structure_hook(A, mutating_hook)
    converter.register_structure_hook(B, recording_hook)

    union = Union[A, B]
    configure_union_probing(union, converter)

    payload = {"x": 1, "y": 2}
    assert converter.structure(payload, union) == B(1)
    # B saw the unmutated input, and the caller's dict is untouched.
    assert seen == [{"x": 1, "y": 2}]
    assert payload == {"x": 1, "y": 2}


def test_nested_unions(converter: BaseConverter) -> None:
    """Unions nested inside members are handled by the converter as usual."""

    @define
    class Wrapper:
        payload: Union[_IntBox, _ListBox]

    @define
    class Other:
        payload: list[int]

    configure_union_probing(Union[_IntBox, _ListBox], converter)
    union = Union[Wrapper, Other]
    configure_union_probing(union, converter)

    assert converter.structure({"payload": {"x": 1}}, union) == Wrapper(_IntBox(1))
    assert converter.structure({"payload": {"x": [1]}}, union) == Wrapper(_ListBox([1]))
    assert converter.structure({"payload": [1, 2]}, union) == Other([1, 2])


def test_custom_member_hook(converter: BaseConverter) -> None:
    """Custom hooks registered for members are used when probing."""

    @define
    class A:
        x: int

    @define
    class B:
        y: int

    def picky_hook(val, cl):
        if val.get("kind") != "a":
            raise ValueError("not an A")
        return A(val["x"])

    converter.register_structure_hook(A, picky_hook)

    union = Union[A, B]
    configure_union_probing(union, converter)

    assert converter.structure({"x": 5, "kind": "a"}, union) == A(5)
    assert converter.structure({"y": 5}, union) == B(5)

    with pytest.raises(UnionProbingError):
        converter.structure({"x": 5}, union)


def test_optional_member(converter: BaseConverter) -> None:
    """`None` only matches `NoneType`, and vice versa."""
    union = Optional[_IntBox]
    configure_union_probing(union, converter)

    assert converter.structure(None, union) is None
    assert converter.structure({"x": 1}, union) == _IntBox(1)

    with pytest.raises(UnionProbingError):
        converter.structure("nope", union)


def test_custom_prefilter(converter: BaseConverter) -> None:
    """A custom prefilter replaces the default conventions."""

    @define
    class A:
        x: int

    @define
    class B:
        x: int

    tried = []

    def prefilter(val, member):
        tried.append(member)
        return val.get("t") == member.__name__

    union = Union[A, B]
    configure_union_probing(union, converter, prefilter=prefilter)

    assert converter.structure({"t": "B", "x": 1}, union) == B(1)
    assert A in tried and B in tried

    # Only the prefiltered member is attempted, so no ambiguity.
    assert converter.structure({"t": "A", "x": 1}, union) == A(1)
