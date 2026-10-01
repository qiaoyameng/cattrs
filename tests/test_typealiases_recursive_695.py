"""Recursive PEP 695 type aliases."""

from __future__ import annotations

from typing import Annotated

import pytest

from cattrs import BaseConverter, Converter, transform_error
from cattrs.errors import IterableValidationError
from cattrs.preconf.json import make_converter
from cattrs.strategies import configure_union_passthrough

from ._compat import is_py312_plus

pytestmark = pytest.mark.skipif(not is_py312_plus, reason="3.12+ syntax")


class Leaf:
    def __init__(self, value: object) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Leaf) and other.value == self.value


def test_recursive_json_alias(converter: BaseConverter) -> None:
    """Recursive JSON aliases round-trip without generating infinitely."""
    type Json = str | int | list[Json] | dict[str, Json]

    configure_union_passthrough(Json, converter)

    structure_hook = converter.get_structure_hook(Json)
    unstructure_hook = converter.get_unstructure_hook(Json)
    data = ["str", 1, ["nested"], {"key": [2, {"deep": ["value"]}]}]

    assert converter.get_structure_hook(Json) is structure_hook
    assert converter.get_unstructure_hook(Json) is unstructure_hook
    assert converter.structure(data, Json) == data
    assert converter.unstructure(data, Json) == data


def test_preconfigured_json_converter_supports_recursive_alias() -> None:
    """The JSON preconfiguration supports the documented recursive alias shape."""
    type Json = str | int | list[Json] | dict[str, Json]

    converter = make_converter()
    data = ["str", 1, ["nested"], {"key": [2, {"deep": ["value"]}]}]

    assert converter.structure(data, Json) == data
    assert converter.dumps(data, unstructure_as=Json)


def test_generic_recursive_alias_hooks_are_parameterized(
    converter: BaseConverter,
) -> None:
    """Different generic alias arguments get distinct hooks and errors."""
    type Tree[T] = T | list[Tree[T]]

    configure_union_passthrough(int | str, converter)

    int_hook = converter.get_structure_hook(Tree[int])
    str_hook = converter.get_structure_hook(Tree[str])

    assert int_hook is not str_hook
    assert converter.get_structure_hook(Tree[int]) is int_hook
    assert converter.get_structure_hook(Tree[str]) is str_hook
    assert converter.structure([1, [2]], Tree[int]) == [1, [2]]
    assert converter.structure(["a", ["b"]], Tree[str]) == ["a", ["b"]]

    with pytest.raises(Exception) as int_exc:
        converter.structure([["bad"]], Tree[int])
    with pytest.raises(Exception) as str_exc:
        converter.structure([[1]], Tree[str])

    assert "Tree[int]" in str(int_exc.value)
    assert "Tree[str]" not in str(int_exc.value)
    assert "Tree[str]" in str(str_exc.value)
    assert "Tree[int]" not in str(str_exc.value)


def test_mutually_recursive_generic_aliases(converter: BaseConverter) -> None:
    """Mutually recursive aliases preserve their type arguments."""
    type Left[T] = T | list[Right[T]]
    type Right[T] = T | dict[str, Left[T]]

    configure_union_passthrough(Left[int], converter)

    value = [1, {"x": [2, {"y": [3]}]}]
    assert converter.structure(value, Left[int]) == value

    structure_hook = converter.get_structure_hook(Right[int])
    unstructure_hook = converter.get_unstructure_hook(Right[int])
    assert converter.get_structure_hook(Right[int]) is structure_hook
    assert converter.get_unstructure_hook(Right[int]) is unstructure_hook
    assert converter.unstructure({"a": [1, {"b": [2]}]}, Right[int]) == {
        "a": [1, {"b": [2]}]
    }
    assert converter.get_structure_hook(Right[int]) is structure_hook
    assert converter.get_unstructure_hook(Right[int]) is unstructure_hook


def test_recursive_collection_hooks_remain_cached(genconverter: Converter) -> None:
    """Converter collection generation does not evict the alias root hooks."""
    type RecList[T] = list[RecList[T]]

    structure_hook = genconverter.get_structure_hook(RecList[int])
    unstructure_hook = genconverter.get_unstructure_hook(RecList[int])

    assert genconverter.structure([[], [[]]], RecList[int]) == [[], [[]]]
    assert genconverter.unstructure([[], [[]]], RecList[int]) == [[], [[]]]
    assert genconverter.get_structure_hook(RecList[int]) is structure_hook
    assert genconverter.get_unstructure_hook(RecList[int]) is unstructure_hook


def test_annotated_custom_leaf_hooks_are_preserved(genconverter: Converter) -> None:
    """Annotated metadata and custom leaf hooks survive recursive aliases."""
    type Leafy = Annotated[Leaf, "leafy"]
    type Node = str | Leafy | list[Node]

    seen = []

    def structure_leaf(value, _):
        seen.append(_)
        return Leaf(value["value"])

    genconverter.register_structure_hook(Leaf, structure_leaf)
    genconverter.register_unstructure_hook(
        Leaf, lambda value: {"value": value.value}
    )
    configure_union_passthrough(Node, genconverter)

    raw = ["x", {"value": 1}, [{"value": 2}]]
    structured = genconverter.structure(raw, Node)

    assert structured == ["x", Leaf(1), [Leaf(2)]]
    assert seen == [Leaf, Leaf]
    assert genconverter.unstructure(structured, Node) == [
        "x",
        {"value": 1},
        [{"value": 2}],
    ]


def test_deep_invalid_leaf_has_detailed_path() -> None:
    """Detailed collection errors retain the recursive alias path."""
    type Json = str | int | list[Json] | dict[str, Json]

    genconverter = Converter()
    configure_union_passthrough(Json, genconverter)

    with pytest.raises(Exception) as exc_info:
        genconverter.structure(
            {"outer": [{"inner": [{"deep": [1.5]}]}]}, Json
        )

    assert transform_error(exc_info.value) == [
        "invalid value for type, expected Json @ $['outer'][0]['inner'][0]['deep'][0]"
    ]


def test_cyclic_input_uses_existing_recursive_graph_rule(
    converter: BaseConverter,
) -> None:
    """Cyclic input follows existing recursive collection rules."""
    type RecList = list[RecList]

    converter.get_structure_hook(RecList)
    converter.get_unstructure_hook(RecList)

    cyclic = []
    cyclic.append(cyclic)

    if not converter.detailed_validation:
        with pytest.raises(RecursionError):
            converter.structure(cyclic, RecList)
        with pytest.raises(RecursionError):
            converter.unstructure(cyclic, RecList)
    else:
        with pytest.raises(IterableValidationError):
            converter.structure(cyclic, RecList)
        with pytest.raises(RecursionError):
            converter.unstructure(cyclic, RecList)
