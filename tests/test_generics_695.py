"""Tests for PEP 695 (Type Parameter Syntax)."""

from dataclasses import dataclass
from typing import Annotated

import pytest
from attrs import define

from cattrs import BaseConverter, Converter
from cattrs.errors import IterableValidationError
from cattrs.strategies import configure_union_passthrough

from ._compat import is_py312_plus


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_simple_generic_roundtrip(converter: BaseConverter):
    """PEP 695 attrs generics work."""

    @define
    class A[T]:
        a: T

    assert converter.structure({"a": "1"}, A[int]) == A(1)
    assert converter.unstructure(A(1)) == {"a": 1}

    if isinstance(converter, Converter):
        # Only supported on a Converter
        assert converter.unstructure(A(1), A[int]) == {"a": 1}


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_simple_generic_roundtrip_dc(converter: BaseConverter):
    """PEP 695 dataclass generics work."""

    @dataclass
    class A[T]:
        a: T

    assert converter.structure({"a": "1"}, A[int]) == A(1)
    assert converter.unstructure(A(1)) == {"a": 1}

    if isinstance(converter, Converter):
        # Only supported on a Converter
        assert converter.unstructure(A(1), A[int]) == {"a": 1}


def test_type_aliases(converter: BaseConverter):
    """PEP 695 type aliases work."""
    type my_int = int

    assert converter.structure("42", my_int) == 42
    assert converter.unstructure(42, my_int) == 42

    type my_other_int = int

    # Manual hooks should work.

    converter.register_structure_hook_func(
        lambda t: t is my_other_int, lambda v, _: v + 10
    )
    converter.register_unstructure_hook_func(
        lambda t: t is my_other_int, lambda v: v - 20
    )

    assert converter.structure(1, my_other_int) == 11
    assert converter.unstructure(100, my_other_int) == 80


def test_type_aliases_simple_hooks(converter: BaseConverter):
    """PEP 695 type aliases work with `register_un/structure_hook`."""
    type my_other_int = int

    converter.register_structure_hook(my_other_int, lambda v, _: v + 10)
    converter.register_unstructure_hook(my_other_int, lambda v: v - 20)

    assert converter.structure(1, my_other_int) == 11
    assert converter.unstructure(100, my_other_int) == 80


def test_type_aliases_overwrite_base_hooks(converter: BaseConverter):
    """Overwriting base hooks should affect type aliases."""
    converter.register_structure_hook(int, lambda v, _: v + 10)
    converter.register_unstructure_hook(int, lambda v: v - 20)

    type my_int = int

    assert converter.structure(1, my_int) == 11
    assert converter.unstructure(100, my_int) == 80


def test_type_alias_with_children(converter: BaseConverter):
    """A type alias that chains to a hook that requires the type parameter works."""

    class TestClass:
        pass

    def structure_testclass(val, type):
        assert type is TestClass
        return TestClass

    converter.register_structure_hook(TestClass, structure_testclass)

    type TestAlias = TestClass
    assert converter.structure(None, TestAlias) is TestClass


def test_generic_type_alias(converter: BaseConverter):
    """Generic type aliases work.

    See https://docs.python.org/3/reference/compound_stmts.html#generic-type-aliases
    for details.
    """

    type Gen1[T] = T

    assert converter.structure("1", Gen1[int]) == 1

    type Gen2[K, V] = dict[K, V]

    assert converter.structure({"a": "1"}, Gen2[str, int]) == {"a": 1}


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_recursive_type_alias_hooks_are_reusable(converter: BaseConverter):
    """Recursive aliases generate reusable hooks without recursive expansion."""
    type Json = str | int | list[Json] | dict[str, Json]

    configure_union_passthrough(Json, converter)

    structure_hook = converter.get_structure_hook(Json)
    unstructure_hook = converter.get_unstructure_hook(Json)

    assert converter.get_structure_hook(Json) is structure_hook
    assert converter.get_unstructure_hook(Json) is unstructure_hook

    payload = ["str", 1, {"nested": [2, "json"]}]
    assert converter.structure(payload, Json) == payload
    assert converter.unstructure(payload, Json) == payload


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_recursive_generic_type_alias_hooks_are_separate(converter: BaseConverter):
    """Different generic alias specializations use different hooks."""
    type Tree[T] = T | list[Tree[T]]

    configure_union_passthrough(Tree[int], converter)
    configure_union_passthrough(Tree[str], converter)

    int_hook = converter.get_structure_hook(Tree[int])
    str_hook = converter.get_structure_hook(Tree[str])
    int_unstructure_hook = converter.get_unstructure_hook(Tree[int])
    str_unstructure_hook = converter.get_unstructure_hook(Tree[str])

    assert int_hook is not str_hook
    assert int_unstructure_hook is not str_unstructure_hook
    assert converter.get_structure_hook(Tree[int]) is int_hook
    assert converter.get_structure_hook(Tree[str]) is str_hook

    assert converter.structure([1, [2]], Tree[int]) == [1, [2]]
    assert converter.structure(["a", ["b"]], Tree[str]) == ["a", ["b"]]
    assert converter.unstructure([1, [2]], Tree[int]) == [1, [2]]
    assert converter.unstructure(["a", ["b"]], Tree[str]) == ["a", ["b"]]


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_recursive_generic_collection_alias(converter: BaseConverter):
    """Recursive generic collection aliases work without a union strategy."""
    type RecursiveList[T] = list[RecursiveList[T]]

    structure_hook = converter.get_structure_hook(RecursiveList[int])
    unstructure_hook = converter.get_unstructure_hook(RecursiveList[int])
    payload = [[[]], [[]]]

    assert converter.get_structure_hook(RecursiveList[int]) is structure_hook
    assert converter.get_unstructure_hook(RecursiveList[int]) is unstructure_hook
    assert converter.structure(payload, RecursiveList[int]) == payload
    assert converter.unstructure(payload, RecursiveList[int]) == payload


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_recursive_type_alias_preserves_annotated_leaf_hook(
    converter: BaseConverter,
):
    """Annotated metadata reaches custom recursive leaf hooks in both directions."""
    AnnotatedLeaf = Annotated[int, "metadata"]
    type Json[LeafType] = LeafType | list[Json[LeafType]]

    configure_union_passthrough(Json[AnnotatedLeaf], converter)

    structured_metadata = []

    def structure_marked(value, type_):
        structured_metadata.append(type_.__metadata__)
        return int(value)

    def unstructure_marked(value):
        return value + 1

    converter.register_structure_hook_func(
        lambda t: t is AnnotatedLeaf, structure_marked
    )
    converter.register_unstructure_hook_func(
        lambda t: t is AnnotatedLeaf, unstructure_marked
    )

    assert converter.structure([1, [2]], Json[AnnotatedLeaf]) == [1, [2]]
    assert converter.unstructure([1, [2]], Json[AnnotatedLeaf]) == [2, [3]]
    assert structured_metadata == [("metadata",), ("metadata",)]


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_mutually_recursive_type_aliases(converter: BaseConverter):
    """Mutually recursive aliases keep dispatching to their specialized hooks."""
    type Left = str | list[Right]
    type Right = int | dict[str, Left]

    configure_union_passthrough(Left, converter)
    configure_union_passthrough(Right, converter)

    payload = [{"a": "left"}, {"b": [1, 2]}]
    assert converter.structure(payload, Left) == payload
    assert converter.unstructure(payload, Left) == payload


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_recursive_type_alias_deep_invalid_leaf(converter: BaseConverter):
    """Deep invalid leaves retain recursive index and key paths."""
    type Json = str | int | list[Json] | dict[str, Json]

    configure_union_passthrough(Json, converter)

    with pytest.raises(Exception) as exc_info:
        converter.structure([["ok", {"deep": [1.5]}]], Json)

    if converter.detailed_validation:
        error = exc_info.value
        path = []
        while hasattr(error, "exceptions"):
            notes = [
                note
                for exception in error.exceptions
                for note in getattr(exception, "__notes__", [])
            ]
            if notes:
                path.append(notes[0].index)
            error = error.exceptions[0]
        assert path == [0, 1, "deep", 0]
    else:
        assert isinstance(exc_info.value, TypeError)


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_recursive_type_alias_repeated_hook_lookup_does_not_expand(
    genconverter: Converter,
):
    """Repeated lookups return the same generated recursive hooks."""
    type Json = str | int | list[Json] | dict[str, Json]

    configure_union_passthrough(Json, genconverter)

    structure_hook = genconverter.get_structure_hook(Json)
    unstructure_hook = genconverter.get_unstructure_hook(Json)

    for _ in range(10):
        assert genconverter.get_structure_hook(Json) is structure_hook
        assert genconverter.get_unstructure_hook(Json) is unstructure_hook


@pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")
def test_recursive_type_alias_cyclic_input(genconverter: Converter):
    """Cyclic input follows the existing recursive-call behavior."""
    type Json = str | int | list[Json] | dict[str, Json]

    configure_union_passthrough(Json, genconverter)

    cyclic_list = []
    cyclic_list.append(cyclic_list)
    cyclic_dict = {}
    cyclic_dict["self"] = cyclic_dict

    with pytest.raises((RecursionError, IterableValidationError)):
        genconverter.structure(cyclic_list, Json)
    with pytest.raises(RecursionError):
        genconverter.unstructure(cyclic_dict, Json)
