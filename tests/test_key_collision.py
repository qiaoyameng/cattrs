"""Tests for mapping key collision detection while structuring and unstructuring."""

import pickle
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Mapping

import pytest
from attrs import define

from cattrs import BaseConverter, Converter, KeyCollisionError, transform_error
from cattrs.errors import IterableValidationError
from cattrs.gen import (
    make_mapping_structure_fn,
    make_mapping_unstructure_fn,
)

CONVERTERS = (BaseConverter, Converter)


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_default_keeps_legacy_behavior(converter_cls):
    """Without the flag, colliding keys still silently overwrite each other."""
    converter = converter_cls()
    assert converter.structure({"01": "a", "1": "b"}, Dict[int, str]) == {1: "b"}
    assert converter.unstructure({"01": "a", "1": "b"}) == {"01": "a", "1": "b"}


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_structure_string_to_int_collision(converter_cls):
    """String keys collapsing onto the same int key are reported, not deduplicated."""
    converter = converter_cls(check_key_collision=True)
    with pytest.raises(Exception) as exc_info:
        converter.structure({"01": "a", "1": "b"}, Dict[int, str])

    collision = _extract_collision(exc_info.value)
    assert isinstance(collision, KeyCollisionError)
    assert collision.first_key == "01"
    assert collision.second_key == "1"
    assert collision.first_position == 0
    assert collision.second_position == 1
    assert collision.target_key_type is int
    assert collision.operation == "structuring"
    # The message hides the original key values but locates both source entries.
    message = str(collision)
    assert "'01'" not in message and "'1'" not in message
    assert "0" in message and "1" in message


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_structure_collision_fast_path(converter_cls):
    """Collisions are detected in the non-detailed validation path too."""
    converter = converter_cls(
        detailed_validation=False, check_key_collision=True
    )
    with pytest.raises(KeyCollisionError) as exc_info:
        converter.structure({"01": "a", "1": "b"}, Dict[int, str])
    collision = exc_info.value
    assert collision.first_position == 0
    assert collision.second_position == 1
    assert collision.target_key_type is int


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_structure_reports_later_collisions(converter_cls):
    """The first occurrence is kept in `seen`; the duplicate carries its position."""
    converter = converter_cls(check_key_collision=True)
    with pytest.raises(Exception) as exc_info:
        converter.structure(
            {"1": "a", "2": "b", "02": "c"}, Dict[int, str]
        )
    collision = _extract_collision(exc_info.value)
    assert collision.first_key == "2"
    assert collision.first_position == 1
    assert collision.second_key == "02"
    assert collision.second_position == 2


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_enum_unstructure_collision(converter_cls):
    """Enum members exporting to the same primitive value collide on unstructuring."""

    class First(Enum):
        MEMBER = 1

    class Second(Enum):
        MEMBER = 1

    converter = converter_cls(check_key_collision=True)
    with pytest.raises(KeyCollisionError) as exc_info:
        converter.unstructure({First.MEMBER: "a", Second.MEMBER: "b"})

    collision = exc_info.value
    assert collision.operation == "unstructuring"
    assert collision.first_key is First.MEMBER
    assert collision.second_key is Second.MEMBER
    assert collision.first_position == 0
    assert collision.second_position == 1
    assert collision.target_key_type is int


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_unstructure_no_collision_passthrough(converter_cls):
    """Non-colliding mappings keep working with the flag enabled."""
    converter = converter_cls(check_key_collision=True)
    assert converter.unstructure({1: "a", 2: "b"}) == {1: "a", 2: "b"}
    assert converter.structure({"1": "a", "2": "b"}, Dict[int, str]) == {
        1: "a",
        2: "b",
    }


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_nested_mapping_collision_in_validation_tree(converter_cls):
    """Collisions nested inside attrs classes join the existing validation tree."""

    @define
    class Box:
        items: Dict[int, str]

    converter = converter_cls(check_key_collision=True)
    with pytest.raises(Exception) as exc_info:
        converter.structure({"items": {"01": "a", "1": "b"}}, Box)

    collision = _extract_collision(exc_info.value)
    assert isinstance(collision, KeyCollisionError)

    if converter_cls is Converter:
        messages = transform_error(exc_info.value)
        assert len(messages) == 1
        assert "key collision" in messages[0]
        assert "$.items['1']" in messages[0]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_hooks_called_once_per_entry(converter_cls):
    """Key and value hooks each run exactly once, even when a collision follows."""
    converter = converter_cls(check_key_collision=True)
    key_calls = []
    value_calls = []
    converter.register_structure_hook(
        int, lambda value, _: key_calls.append(value) or int(value)
    )
    converter.register_structure_hook(
        str, lambda value, _: value_calls.append(value) or value
    )

    with pytest.raises(Exception):
        converter.structure({"01": "a", "1": "b"}, Dict[int, str])

    assert key_calls == ["01", "1"]
    assert value_calls == ["a", "b"]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_unhashable_target_key_keeps_type_error(converter_cls):
    """An unhashable structured key still raises TypeError, not a collision error."""
    fast_converter = converter_cls(
        detailed_validation=False, check_key_collision=True
    )
    fast_converter.register_structure_hook(str, lambda value, _: [value])
    with pytest.raises(TypeError):
        fast_converter.structure({"a": 1, "b": 2}, Dict[list, int])

    detailed_converter = converter_cls(check_key_collision=True)
    detailed_converter.register_structure_hook(str, lambda value, _: [value])
    with pytest.raises(IterableValidationError) as exc_info:
        detailed_converter.structure({"a": 1, "b": 2}, Dict[list, int])
    assert isinstance(exc_info.value.exceptions[0], TypeError)
    assert not _has_collision(exc_info.value)


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_hook_failure_keeps_original_exception(converter_cls):
    """Failing hooks surface their original exception class."""
    fast_converter = converter_cls(
        detailed_validation=False, check_key_collision=True
    )
    with pytest.raises(ValueError):
        fast_converter.structure({"a": 1}, Dict[int, int])

    detailed_converter = converter_cls(check_key_collision=True)
    with pytest.raises(IterableValidationError) as exc_info:
        detailed_converter.structure({"a": 1}, Dict[int, int])
    assert isinstance(exc_info.value.exceptions[0], ValueError)


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_abstract_mapping_target_collision(converter_cls):
    """Abstract mapping annotations (materialized as dicts) are checked too."""
    converter = converter_cls(check_key_collision=True)
    with pytest.raises(Exception) as exc_info:
        converter.structure({"01": 1, "1": 2}, Mapping[int, int])
    collision = _extract_collision(exc_info.value)
    assert collision.first_position == 0
    assert collision.second_position == 1


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_ordered_mapping_positions(converter_cls):
    """Positions reflect source iteration order for ordered mappings."""
    converter = converter_cls(check_key_collision=True)
    source = OrderedDict([("01", 1), ("x", 2), ("1", 3)])
    with pytest.raises(Exception) as exc_info:
        converter.structure(source, Dict[int, int])
    collision = _extract_collision(exc_info.value)
    assert collision.first_position == 0
    assert collision.second_position == 2


def test_generated_factories_explicit_flag():
    """The gen factories accept the flag independently of the converter."""
    converter = Converter()

    hook = make_mapping_structure_fn(
        Dict[int, str], converter, check_key_collision=True
    )
    with pytest.raises(IterableValidationError) as exc_info:
        hook({"01": "a", "1": "b"}, Dict[int, str])
    assert isinstance(exc_info.value.exceptions[0], KeyCollisionError)

    disabled = make_mapping_structure_fn(
        Dict[int, str], converter, check_key_collision=False
    )
    assert disabled({"01": "a", "1": "b"}, Dict[int, str]) == {1: "b"}

    unstructure_hook = make_mapping_unstructure_fn(
        Dict[int, int],
        Converter(check_key_collision=True),
        unstructure_to=OrderedDict,
    )
    assert unstructure_hook({1: "a", 2: "b"}) == OrderedDict(
        [(1, "a"), (2, "b")]
    )


def test_converter_copy_preserves_flag():
    """The flag survives copies and can be overridden."""
    converter = Converter(check_key_collision=True)
    copied = converter.copy()
    with pytest.raises(Exception):
        copied.structure({"01": 1, "1": 2}, Dict[int, int])

    disabled = converter.copy(check_key_collision=False)
    assert disabled.structure({"01": 1, "1": 2}, Dict[int, int]) == {1: 2}


def test_base_and_generated_converters_agree():
    """BaseConverter and Converter produce equivalent collision reports."""
    base = BaseConverter(check_key_collision=True)
    generated = Converter(check_key_collision=True)

    for converter in (base, generated):
        with pytest.raises(Exception) as exc_info:
            converter.structure({"01": "a", "1": "b"}, Dict[int, str])
        collision = _extract_collision(exc_info.value)
        assert (
            collision.first_position,
            collision.second_position,
            collision.target_key_type,
        ) == (0, 1, int)


def test_key_collision_error_is_pickleable():
    """Collision errors survive a pickle round-trip, including all source info."""
    error = KeyCollisionError(dict, int, "01", 0, "1", 1, "unstructuring")
    restored = pickle.loads(pickle.dumps(error))

    assert restored.cl is dict
    assert restored.target_key_type is int
    assert (restored.first_key, restored.first_position) == ("01", 0)
    assert (restored.second_key, restored.second_position) == ("1", 1)
    assert restored.operation == "unstructuring"
    assert str(restored) == str(error)


def _extract_collision(exception):
    """Find the KeyCollisionError potentially nested in an exception group."""
    found = []

    def walk(exc):
        if isinstance(exc, KeyCollisionError):
            found.append(exc)
        for sub in getattr(exc, "exceptions", ()):
            walk(sub)

    walk(exception)
    assert found, f"No KeyCollisionError found in {exception!r}"
    return found[0]


def _has_collision(exception):
    found = []

    def walk(exc):
        if isinstance(exc, KeyCollisionError):
            found.append(exc)
        for sub in getattr(exc, "exceptions", ()):
            walk(sub)

    walk(exception)
    return bool(found)
