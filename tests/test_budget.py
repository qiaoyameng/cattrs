"""Tests for per-conversion :class:`ConversionBudget` support."""

from __future__ import annotations

import inspect
from typing import Deque, Dict, Generic, List, Tuple, TypeVar, TypedDict

import pytest
from attrs import define

from cattrs import (
    BaseConverter,
    ConversionBudget,
    ConversionBudgetExceeded,
    Converter,
)
from cattrs._compat import ExceptionGroup
from cattrs.errors import contains_budget_exceeded

T = TypeVar("T")


@define
class Root:
    xs: List[List[int]]
    d: Dict[str, int]


@define
class Box(Generic[T]):
    item: T


@define
class Recursive:
    inner: List[Recursive]


class TD(TypedDict):
    xs: List[int]


OK_DATA = {"xs": [[1, 2], [3, 4], [5, 6]], "d": {"a": 7, "b": 8}}
BAD_DATA = {"xs": [["a", "b"], ["c", "d"]], "d": {"a": "x"}}

CONVERTERS = (BaseConverter, Converter)


def _int_counter():
    calls = [0]

    def hook(value, _):
        calls[0] += 1
        return int(value)

    return hook, calls


def _int_unstructure_counter():
    calls = [0]

    def hook(value):
        calls[0] += 1
        return value

    return hook, calls


def test_budget_defaults_and_validation():
    """Unlimited budgets stay inactive and negative limits are rejected."""
    budget = ConversionBudget.of()
    assert not budget.active
    assert budget.nodes == 0
    assert budget.depth == 0
    assert budget.max_depth == 0
    assert budget.container_elements == 0
    assert budget.errors == 0
    assert budget.exhausted is False
    with pytest.raises(ValueError):
        ConversionBudget.of(node_limit=-1)
    with pytest.raises(ValueError):
        ConversionBudget.of(container_limit=-1)


def test_unmetered_converter_keeps_fast_path():
    """Without a budget the generated hooks contain no budget machinery."""
    converter = Converter()
    assert converter.budget is None
    source = inspect.getsource(converter.get_structure_hook(Root))
    assert "budget" not in source
    source = inspect.getsource(converter.get_unstructure_hook(Root))
    assert "budget" not in source


def test_budget_installation_regenerates_hooks():
    """Installing and clearing a budget swaps generated hook variants."""
    converter = Converter()
    assert "budget" not in inspect.getsource(converter.get_structure_hook(Root))
    converter.configure_budget(ConversionBudget.of(node_limit=100))
    assert "budget" in inspect.getsource(converter.get_structure_hook(Root))
    converter.configure_budget(None)
    assert "budget" not in inspect.getsource(converter.get_structure_hook(Root))


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_node_limit(converter_cls):
    """The node limit stops the conversion and records the visited nodes."""
    converter = converter_cls()
    hook, calls = _int_counter()
    converter.register_structure_hook(int, hook)
    budget = ConversionBudget.of(node_limit=6)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure(OK_DATA, Root)
    finally:
        converter.configure_budget(None)

    exc = exc_info.value
    assert exc.resource == "nodes"
    assert exc.limit == 6
    assert exc.value == 7
    assert exc.budget is budget
    assert "nodes" in str(exc)
    assert budget.nodes == 7
    assert budget.max_depth == 4
    assert budget.container_elements == 7
    assert budget.errors == 0
    assert budget.exhausted is True
    assert budget.depth == 0
    assert calls[0] == 2


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_depth_limit(converter_cls):
    """The depth limit stops before descending further."""
    converter = converter_cls()
    hook, calls = _int_counter()
    converter.register_structure_hook(int, hook)
    budget = ConversionBudget.of(depth_limit=3)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure(OK_DATA, Root)
    finally:
        converter.configure_budget(None)

    exc = exc_info.value
    assert exc.resource == "depth"
    assert exc.limit == 3
    assert exc.value == 4
    assert budget.nodes == 4
    assert budget.max_depth == 4
    assert budget.container_elements == 5
    assert budget.depth == 0
    assert calls[0] == 0


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_container_limit_applies_before_allocation(converter_cls):
    """Container elements are reserved before the target container exists."""
    converter = converter_cls()
    hook, calls = _int_counter()
    converter.register_structure_hook(int, hook)
    budget = ConversionBudget.of(container_limit=3)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure(OK_DATA, Root)
    finally:
        converter.configure_budget(None)

    exc = exc_info.value
    assert exc.resource == "containers"
    assert exc.limit == 3
    assert exc.value == 5
    assert budget.nodes == 3
    assert budget.max_depth == 3
    assert budget.container_elements == 5
    assert calls[0] == 0


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_error_limit_stops_detailed_aggregation(converter_cls):
    """The error limit escapes detailed validation instead of being grouped."""
    converter = converter_cls()
    hook, calls = _int_counter()
    converter.register_structure_hook(int, hook)
    budget = ConversionBudget.of(error_limit=1)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure(BAD_DATA, Root)
    finally:
        converter.configure_budget(None)

    exc = exc_info.value
    assert exc.resource == "errors"
    assert exc.limit == 1
    assert exc.value == 2
    # The termination is not wrapped in a ClassValidationError or similar.
    assert type(exc) is ConversionBudgetExceeded
    assert budget.errors == 2
    assert budget.nodes == 5
    assert budget.max_depth == 4
    assert budget.container_elements == 4
    assert budget.depth == 0
    assert calls[0] == 2


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_error_budget_ignored_without_detailed_validation(converter_cls):
    """Errors are only counted when detailed validation collects them."""
    converter = converter_cls(detailed_validation=False)
    budget = ConversionBudget.of(error_limit=1)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ValueError):
            converter.structure(BAD_DATA, Root)
    finally:
        converter.configure_budget(None)
    assert budget.errors == 0
    assert budget.exhausted is False


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_unstructure_limits(converter_cls):
    """Node, depth and container limits cover unstructure too."""
    converter = converter_cls()
    instance = Root(xs=[[1, 2], [3, 4]], d={"a": 5})

    hook, calls = _int_unstructure_counter()
    converter.register_unstructure_hook(int, hook)

    budget = ConversionBudget.of(node_limit=5)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.unstructure(instance)
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "nodes"
    assert exc_info.value.value == 6
    assert budget.nodes == 6
    assert budget.max_depth == 4
    assert budget.container_elements == 4
    assert budget.depth == 0
    assert calls[0] == 2

    converter2 = converter_cls()
    hook2, calls2 = _int_unstructure_counter()
    converter2.register_unstructure_hook(int, hook2)
    budget2 = ConversionBudget.of(container_limit=2)
    converter2.configure_budget(budget2)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter2.unstructure(instance)
    finally:
        converter2.configure_budget(None)
    assert exc_info.value.resource == "containers"
    assert exc_info.value.value == 4
    assert budget2.container_elements == 4
    assert calls2[0] == 0

    converter3 = converter_cls()
    budget3 = ConversionBudget.of(depth_limit=3)
    converter3.configure_budget(budget3)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter3.unstructure(instance)
    finally:
        converter3.configure_budget(None)
    assert exc_info.value.resource == "depth"
    assert budget3.max_depth == 4
    assert budget3.depth == 0


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_container_limit_covers_all_collections(converter_cls):
    """The same budget survives collection, tuple and deque boundaries."""
    converter = converter_cls()
    cases = [
        (List[int], [1, 2, 3, 4], 4),
        (Tuple[int, ...], (1, 2, 3), 3),
        (Deque[int], [1, 2], 2),
        (Tuple[int, str, int], (1, "2", 3), 3),
    ]
    for cl, value, expected in cases:
        budget = ConversionBudget.of(container_limit=1)
        converter.configure_budget(budget)
        try:
            with pytest.raises(ConversionBudgetExceeded) as exc_info:
                converter.structure(value, cl)
        finally:
            converter.configure_budget(None)
        assert exc_info.value.resource == "containers"
        assert exc_info.value.value == expected


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_set_and_dict_container_limits(converter_cls):
    """Sets and mappings charge their elements against the same budget."""
    converter = converter_cls()
    budget = ConversionBudget.of(container_limit=1)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure([1, 2, 3], set[int])
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "containers"
    assert exc_info.value.value == 3

    converter = converter_cls()
    budget = ConversionBudget.of(container_limit=1)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure({"a": 1, "b": 2}, Dict[str, int])
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "containers"
    assert exc_info.value.value == 2


def test_typeddict_budget():
    """TypedDicts are metered like generated class hooks."""
    converter = Converter()
    budget = ConversionBudget.of(node_limit=3)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure({"xs": [1, 2, 3]}, TD)
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "nodes"

    converter = Converter()
    budget = ConversionBudget.of(container_limit=1)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure({"xs": [1, 2, 3]}, TD)
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "containers"
    assert budget.container_elements == 4


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_reusing_converter_after_exhaustion(converter_cls):
    """An exhausted budget leaves no counters behind on the converter."""
    converter = converter_cls()
    budget = ConversionBudget.of(node_limit=6)
    converter.configure_budget(budget)
    with pytest.raises(ConversionBudgetExceeded):
        converter.structure(OK_DATA, Root)
    converter.configure_budget(None)

    snapshot = (
        budget.nodes,
        budget.depth,
        budget.max_depth,
        budget.container_elements,
        budget.errors,
        budget.exhausted,
    )
    expected = Root(xs=[[1, 2], [3, 4], [5, 6]], d={"a": 7, "b": 8})

    # The same converter works normally once the budget is removed.
    assert converter.structure(OK_DATA, Root) == expected
    assert converter.unstructure(expected) == OK_DATA
    assert converter.structure(OK_DATA, Root) == expected

    # The old budget is untouched by the unbudgeted conversions.
    assert (
        budget.nodes,
        budget.depth,
        budget.max_depth,
        budget.container_elements,
        budget.errors,
        budget.exhausted,
    ) == snapshot

    # A fresh budget starts from zero on the same converter.
    fresh = ConversionBudget.of(node_limit=6)
    converter.configure_budget(fresh)
    with pytest.raises(ConversionBudgetExceeded):
        converter.structure(OK_DATA, Root)
    converter.configure_budget(None)
    assert fresh.nodes == 7
    assert budget.nodes == snapshot[0]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_conversion_budget_context_manager(converter_cls):
    """The context manager installs and removes a budget reliably."""
    converter = converter_cls()
    with converter.conversion_budget(ConversionBudget.of(node_limit=2)) as budget:
        assert converter.budget is budget
        with pytest.raises(ConversionBudgetExceeded):
            converter.structure([1, 2, 3], List[int])
    assert converter.budget is None
    assert converter.structure([1, 2, 3], List[int]) == [1, 2, 3]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_copy_does_not_share_budgets(converter_cls):
    """Budgets are single-use and are not copied by default."""
    budget = ConversionBudget.of(node_limit=5)
    converter = converter_cls(budget=budget)
    assert converter.budget is budget

    copy = converter.copy()
    assert copy.budget is None
    assert copy.structure([1, 2], List[int]) == [1, 2]

    with_budget = converter.copy(budget=ConversionBudget.of(node_limit=1))
    assert with_budget.budget is not None
    assert with_budget.budget is not budget


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_zero_limit_triggers_immediately(converter_cls):
    """A zero limit is valid and terminates on the first visit."""
    converter = converter_cls()
    budget = ConversionBudget.of(node_limit=0)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure([1], List[int])
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "nodes"
    assert exc_info.value.value == 1
    assert budget.nodes == 1


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_generic_and_user_hook_boundaries_share_budget(converter_cls):
    """Generics and nested user hooks never reset the budget."""
    converter = converter_cls()
    outer_calls = [0]
    inner_calls = [0]

    def outer_hook(value, _):
        outer_calls[0] += 1
        return [converter.structure(item, int) for item in value]

    def inner_hook(value, _):
        inner_calls[0] += 1
        return int(value)

    converter.register_structure_hook_func(
        lambda typ: typ == List[int], outer_hook
    )
    converter.register_structure_hook(int, inner_hook)

    budget = ConversionBudget.of(node_limit=3)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure([1, 2, 3], List[int])
    finally:
        converter.configure_budget(None)

    assert exc_info.value.resource == "nodes"
    assert exc_info.value.value == 4
    assert budget.nodes == 4
    assert budget.max_depth == 2
    assert budget.depth == 0
    # The outer hook ran once and two inner int hooks ran before the stop.
    assert outer_calls == [1]
    assert inner_calls == [2]

    # A generic class hook shares the very same budget counters.
    generic_converter = converter_cls()
    budget = ConversionBudget.of(node_limit=3)
    generic_converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            generic_converter.structure({"item": [1, 2, 3]}, Box[List[int]])
    finally:
        generic_converter.configure_budget(None)
    assert exc_info.value.resource == "nodes"
    assert exc_info.value.value == 4
    assert budget.nodes == 4
    assert budget.max_depth == 3
    assert budget.depth == 0
    assert budget.container_elements == 3


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_termination_escapes_exception_groups(converter_cls):
    """Budget terminations are never aggregated as ordinary field errors."""
    converter = converter_cls()
    budget = ConversionBudget.of(node_limit=2)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure({"xs": [1, 2, 3]}, Root)
    finally:
        converter.configure_budget(None)
    assert type(exc_info.value) is ConversionBudgetExceeded
    assert budget.nodes == 3

    # A user hook that wraps the termination still halts the conversion.
    converter = converter_cls()

    def wrapping_hook(value, _):
        try:
            return converter.structure(value, int)
        except ConversionBudgetExceeded as exc:
            raise ExceptionGroup("user-wrapped", [exc])

    converter.register_structure_hook_func(
        lambda typ: typ == List[int], wrapping_hook
    )
    budget = ConversionBudget.of(node_limit=1)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ExceptionGroup) as exc_info:
            converter.structure([1, 2, 3], List[int])
    finally:
        converter.configure_budget(None)
    assert contains_budget_exceeded(exc_info.value)
    assert budget.nodes == 2


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_unlimited_budget_completes_normal_conversion(converter_cls):
    """An unlimited budget records counters but never terminates."""
    converter = converter_cls()
    budget = ConversionBudget.of()
    converter.configure_budget(budget)
    try:
        result = converter.structure(OK_DATA, Root)
        unstructured = converter.unstructure(result)
    finally:
        converter.configure_budget(None)
    assert result == Root(xs=[[1, 2], [3, 4], [5, 6]], d={"a": 7, "b": 8})
    assert unstructured == OK_DATA
    assert budget.nodes > 0
    assert budget.max_depth > 0
    assert budget.container_elements > 0
    assert budget.errors == 0
    assert budget.exhausted is False
    assert budget.depth == 0


def test_recursive_class_budget():
    """Recursive late-bound hooks share and honor one budget."""
    data = {"inner": [{"inner": [{"inner": []}]}]}
    converter = Converter()

    budget = ConversionBudget.of(node_limit=4)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure(data, Recursive)
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "nodes"
    assert budget.nodes == 5
    assert budget.max_depth == 4
    assert budget.depth == 0

    budget = ConversionBudget.of(depth_limit=3)
    converter.configure_budget(budget)
    try:
        with pytest.raises(ConversionBudgetExceeded) as exc_info:
            converter.structure(data, Recursive)
    finally:
        converter.configure_budget(None)
    assert exc_info.value.resource == "depth"
    assert budget.max_depth == 4
    assert budget.depth == 0

    # The same converter remains usable after budget termination.
    assert converter.structure(data, Recursive) == Recursive(
        [Recursive([Recursive([])])]
    )
