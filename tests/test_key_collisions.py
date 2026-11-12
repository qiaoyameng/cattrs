"""Tests for mapping key collision detection while (un)structuring."""

from collections import Counter
from collections.abc import Mapping
from enum import Enum
from typing import Any, Dict, Union

import pytest
from attrs import define

from cattrs import (
    BaseConverter,
    Converter,
    MappingKeyCollisionError,
    transform_error,
)
from cattrs.errors import (
    DictKeyCollisionError,
    IterableValidationError,
    IterableValidationNote,
    KeyCollisionError,
)
from cattrs.gen import make_mapping_structure_fn, make_mapping_unstructure_fn

COLLIDING_INPUT = {"01": "a", "1": "b"}


@pytest.fixture(params=(BaseConverter, Converter))
def collision_converter_cls(request):
    return request.param


@pytest.fixture(params=(True, False))
def collision_converter(request, collision_converter_cls):
    return collision_converter_cls(
        detailed_validation=request.param, check_key_collisions=True
    )


def test_default_keeps_legacy_silent_overwrite(converter):
    """Without the flag, colliding keys are overwritten, as before."""
    assert converter.check_key_collisions is False
    assert converter.structure(COLLIDING_INPUT, Dict[int, str]) == {1: "b"}


def test_structure_collision_detects_both_source_entries(collision_converter):
    """The error reports both source entries and the target key type."""
    detailed = collision_converter.detailed_validation

    with pytest.raises(Exception) as exc_info:
        collision_converter.structure(COLLIDING_INPUT, Dict[int, str])

    if detailed:
        assert isinstance(exc_info.value, IterableValidationError)
        assert len(exc_info.value.exceptions) == 1
        error = exc_info.value.exceptions[0]
        assert isinstance(error, MappingKeyCollisionError)
        assert error.__notes__ == ["Structuring mapping key @ key '1'"]
        note = error.__notes__[0]
        assert isinstance(note, IterableValidationNote)
        assert note.index == "1"
        assert note.type is int
    else:
        error = exc_info.value
        assert isinstance(error, MappingKeyCollisionError)

    assert error.key_type is int
    assert error.source_key == "1"
    assert error.colliding_key == "01"
    assert "'1'" in str(error)
    assert "'01'" in str(error)
    assert "int" in str(error)


def test_structure_collision_does_not_return_partial_result(collision_converter):
    """A collision never surfaces a truncated mapping, even without other errors."""
    with pytest.raises(Exception):
        collision_converter.structure(COLLIDING_INPUT, Dict[int, str])


def test_structure_hooks_called_once(collision_converter_cls):
    """Key and value hooks run exactly once per entry before the collision."""
    converter = collision_converter_cls(check_key_collisions=True)
    key_calls = []
    val_calls = []
    converter.register_structure_hook(
        int, lambda val, t: (key_calls.append(val), int(val))[1]
    )
    converter.register_structure_hook(
        str, lambda val, t: (val_calls.append(val), val)[1]
    )

    with pytest.raises(Exception):
        converter.structure(COLLIDING_INPUT, Dict[int, str])

    assert key_calls == ["01", "1"]
    assert val_calls == ["a", "b"]


def test_enum_structure_collision(collision_converter):
    """Enum keys structuring to the same member collide."""

    class E(Enum):
        A = 1

    with pytest.raises(Exception) as exc_info:
        collision_converter.structure({E.A: "a", 1: "b"}, Dict[E, str])

    error = (
        exc_info.value.exceptions[0]
        if collision_converter.detailed_validation
        else exc_info.value
    )
    assert isinstance(error, MappingKeyCollisionError)
    assert error.key_type is E
    assert error.source_key == 1
    assert error.colliding_key == E.A


def test_enum_unstructure_collision(collision_converter):
    """Two enum classes exporting to the same value collide while unstructuring."""

    class E(Enum):
        A = 1

    class F(Enum):
        A = 1

    source = {E.A: "a", F.A: "b"}

    with pytest.raises(MappingKeyCollisionError) as exc_info:
        collision_converter.unstructure(source)

    assert exc_info.value.key_type is int
    assert exc_info.value.source_key == F.A
    assert exc_info.value.colliding_key == E.A


def test_enum_unstructure_collision_via_unstructure_as(collision_converter):
    """A generic unstructure target also detects enum key collisions."""

    class E(Enum):
        A = 1

    class F(Enum):
        A = 1

    target = Dict[Union[E, F], str]
    with pytest.raises(MappingKeyCollisionError) as exc_info:
        collision_converter.unstructure({E.A: "a", F.A: "b"}, unstructure_as=target)

    assert exc_info.value.key_type is int
    assert exc_info.value.source_key == F.A
    assert exc_info.value.colliding_key == E.A
    assert collision_converter.unstructure({E.A: "a"}, unstructure_as=target) == {
        1: "a"
    }


def test_unstructure_hooks_called_once(collision_converter_cls):
    """Unstructure key and value hooks run once per entry."""

    class E(Enum):
        A = 1

    class F(Enum):
        A = 1

    converter = collision_converter_cls(check_key_collisions=True)
    key_calls = []
    converter.register_unstructure_hook(
        E, lambda e: (key_calls.append(e), e.value)[1]
    )

    with pytest.raises(MappingKeyCollisionError):
        converter.unstructure({E.A: "a", F.A: "b"})

    assert key_calls == [E.A]


def test_no_collision_roundtrip(collision_converter):
    """Non-colliding mappings keep converting normally."""
    assert collision_converter.structure(
        {"1": "a", "2": "b"}, Dict[int, str]
    ) == {1: "a", 2: "b"}
    assert collision_converter.unstructure({1: "a", 2: "b"}) == {
        1: "a",
        2: "b",
    }


def test_nested_collision_enters_validation_tree(collision_converter_cls):
    """Collision errors in nested mappings are part of the existing error tree."""

    @define
    class Inner:
        n: int

    @define
    class Outer:
        mapping: Dict[int, Inner]

    converter = collision_converter_cls(
        detailed_validation=True, check_key_collisions=True
    )
    with pytest.raises(Exception) as exc_info:
        converter.structure(
            {"mapping": {"01": {"n": 1}, "1": {"n": 2}}}, Outer
        )

    messages = transform_error(exc_info.value)
    assert len(messages) == 1
    message = messages[0]
    assert "Key collision" in message
    assert message.endswith("['1']")
    assert "'1'" in message
    assert "'01'" in message


def test_unhashable_target_key_keeps_type_error(collision_converter_cls):
    """Unhashable converted keys still raise TypeError, in the usual wrapping."""
    for detailed in (True, False):
        converter = collision_converter_cls(
            detailed_validation=detailed, check_key_collisions=True
        )
        with pytest.raises(Exception) as exc_info:
            converter.structure({"a": 1, "b": 2}, Dict[list, int])
        if detailed:
            assert isinstance(exc_info.value, IterableValidationError)
            assert isinstance(exc_info.value.exceptions[0], TypeError)
        else:
            assert isinstance(exc_info.value, TypeError)


def test_failing_key_hook_keeps_original_exception(collision_converter_cls):
    """A hook failure is not replaced by a collision error."""
    for detailed in (True, False):
        converter = collision_converter_cls(
            detailed_validation=detailed, check_key_collisions=True
        )
        with pytest.raises(Exception) as exc_info:
            converter.structure({"x": 1}, Dict[int, int])
        if detailed:
            assert isinstance(exc_info.value.exceptions[0], ValueError)
        else:
            assert isinstance(exc_info.value, ValueError)


def test_multiple_errors_include_collision(collision_converter_cls):
    """Collisions are grouped together with other per-entry validation errors."""
    converter = collision_converter_cls(
        detailed_validation=True, check_key_collisions=True
    )
    with pytest.raises(IterableValidationError) as exc_info:
        converter.structure(
            {"01": 1, "1": 2, "nope": 3}, Dict[int, int]
        )

    inner = exc_info.value.exceptions
    assert any(isinstance(e, MappingKeyCollisionError) for e in inner)
    assert any(isinstance(e, ValueError) for e in inner)


def test_generated_and_base_converter_agree():
    """BaseConverter and Converter produce equivalent collision behavior."""
    for detailed in (True, False):
        errors = []
        for converter_cls in (BaseConverter, Converter):
            converter = converter_cls(
                detailed_validation=detailed, check_key_collisions=True
            )
            with pytest.raises(Exception) as exc_info:
                converter.structure(COLLIDING_INPUT, Dict[int, str])
            error = (
                exc_info.value.exceptions[0]
                if detailed
                else exc_info.value
            )
            errors.append(error)

        assert isinstance(errors[0], MappingKeyCollisionError)
        assert isinstance(errors[1], MappingKeyCollisionError)
        assert (
            errors[0].key_type,
            errors[0].source_key,
            errors[0].colliding_key,
        ) == (
            errors[1].key_type,
            errors[1].source_key,
            errors[1].colliding_key,
        )
        assert str(errors[0]) == str(errors[1])


def test_mapping_structure_factory_explicit_flag():
    """The generated structure factory accepts an explicit collision flag."""
    converter = Converter()
    fn = make_mapping_structure_fn(
        Dict[int, str], converter, check_key_collisions=True
    )
    with pytest.raises(IterableValidationError) as exc_info:
        fn(COLLIDING_INPUT, Dict[int, str])
    assert isinstance(exc_info.value.exceptions[0], MappingKeyCollisionError)

    legacy_fn = make_mapping_structure_fn(
        Dict[int, str], converter, check_key_collisions=False
    )
    assert legacy_fn(COLLIDING_INPUT, Dict[int, str]) == {1: "b"}


def test_mapping_unstructure_factory_explicit_flag():
    """The generated unstructure factory accepts an explicit collision flag."""

    class E(Enum):
        A = 1

    class F(Enum):
        A = 1

    converter = BaseConverter()
    fn = make_mapping_unstructure_fn(
        Dict[E, str], converter, check_key_collisions=True
    )
    with pytest.raises(MappingKeyCollisionError):
        fn({E.A: "a", F.A: "b"})


def test_unstructure_to_list_keeps_pairs_on_collision_path():
    """Non-dict collection overrides still receive key-value pairs."""
    converter = Converter(
        check_key_collisions=True,
        unstruct_collection_overrides={Mapping: list},
    )
    hook = converter.gen_unstructure_mapping(Dict[int, str], unstructure_to=list)
    assert hook({1: "a", 2: "b"}) == [(1, "a"), (2, "b")]


def test_collision_check_alias(collision_converter_cls):
    """The `collision_check` alias configures and reflects the same flag."""
    assert KeyCollisionError is MappingKeyCollisionError
    assert DictKeyCollisionError is MappingKeyCollisionError

    converter = collision_converter_cls(collision_check=True)
    assert converter.check_key_collisions is True
    assert converter.collision_check is True
    converter.collision_check = False
    assert converter.check_key_collisions is False


def test_copy_preserves_and_overrides_flag(collision_converter_cls):
    """Converter.copy carries the flag and allows overriding it."""
    converter = collision_converter_cls(check_key_collisions=True)
    assert converter.copy().check_key_collisions is True
    assert converter.copy(check_key_collisions=False).check_key_collisions is False
    assert converter.copy(collision_check=False).check_key_collisions is False

    default = collision_converter_cls()
    assert default.copy().check_key_collisions is False
    assert default.copy(collision_check=True).check_key_collisions is True


def test_bare_and_any_mappings_skip_the_check(collision_converter):
    """Bare and Any-key mappings have no conversion, so no collision check."""
    assert collision_converter.structure(COLLIDING_INPUT, dict) == COLLIDING_INPUT
    assert collision_converter.structure(
        COLLIDING_INPUT, Dict[Any, Any]
    ) == COLLIDING_INPUT


def test_counter_structure_collision():
    """Counter structuring detects key collisions in both modes."""
    for detailed in (True, False):
        converter = Converter(
            detailed_validation=detailed, check_key_collisions=True
        )
        with pytest.raises(Exception) as exc_info:
            converter.structure({"01": 1, "1": 2}, Counter[int])
        error = (
            exc_info.value.exceptions[0] if detailed else exc_info.value
        )
        assert isinstance(error, MappingKeyCollisionError)
        assert error.source_key == "1"
        assert error.colliding_key == "01"


def test_counter_unstructure_unchanged(collision_converter_cls):
    """The flag does not change Counter unstructuring semantics."""
    checked = collision_converter_cls(check_key_collisions=True)
    legacy = collision_converter_cls()
    source = Counter({"a": 1, "b": 2})
    assert checked.unstructure(source) == legacy.unstructure(source)
