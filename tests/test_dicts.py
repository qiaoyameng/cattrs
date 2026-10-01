from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict

import pytest

from cattrs import BaseConverter, Converter, transform_error
from cattrs.errors import IterableValidationError, MappingKeyCollisionError
from cattrs.gen import make_mapping_structure_fn


class MultiKeyMapping(Mapping):
    def __init__(self, *items):
        self._items = items

    def __getitem__(self, key):
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return iter(key for key, _ in self._items)

    def __len__(self):
        return len(self._items)


class CustomKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(int(self.value))

    def __eq__(self, other):
        return isinstance(other, CustomKey) and int(self.value) == int(other.value)


class UnhashableKey:
    __hash__ = None


def test_any_keys(converter):
    """Dicts with any keys work."""
    assert converter.structure({b"": "1"}, Dict[Any, int]) == {b"": 1}


def test_any_values(converter):
    """Dicts with any values work."""
    assert converter.structure({"1": b"1"}, Dict[int, Any]) == {1: b"1"}


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_strict_key_validation_reports_collision(converter_cls):
    """Colliding structured keys report both originals and the target key."""
    converter = converter_cls(
        strict_key_validation=True, detailed_validation=False
    )

    with pytest.raises(MappingKeyCollisionError) as exc_info:
        converter.structure({"1": "甲", "01": "乙"}, Dict[int, str])

    assert exc_info.value.existing_original_key == "1"
    assert exc_info.value.original_key == "01"
    assert exc_info.value.target_key == 1


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_strict_key_validation_uses_hash_and_equality(converter_cls):
    """Bool keys and custom keys follow Python hash/equality semantics."""
    converter = converter_cls(
        strict_key_validation=True, detailed_validation=False
    )

    with pytest.raises(MappingKeyCollisionError) as exc_info:
        converter.structure(
            MultiKeyMapping((1, "a"), (True, "b")), Dict[int, str]
        )

    assert exc_info.value.existing_original_key == 1
    assert exc_info.value.original_key is True
    assert exc_info.value.target_key == 1

    with pytest.raises(MappingKeyCollisionError) as exc_info:
        converter.structure(
            MultiKeyMapping((1, "a"), (True, "b")), Dict[Any, Any]
        )

    assert exc_info.value.existing_original_key == 1
    assert exc_info.value.original_key is True
    assert exc_info.value.target_key is True

    converter.register_structure_hook(CustomKey, lambda value, _: CustomKey(value))
    with pytest.raises(MappingKeyCollisionError) as exc_info:
        converter.structure(
            MultiKeyMapping(("1", "a"), ("01", "b")), Dict[CustomKey, str]
        )

    assert exc_info.value.existing_original_key == "1"
    assert exc_info.value.original_key == "01"
    assert exc_info.value.target_key == CustomKey("1")


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_strict_key_validation_unhashable_target_key(converter_cls):
    """An unhashable structured key raises a TypeError."""
    converter = converter_cls(
        strict_key_validation=True, detailed_validation=False
    )
    converter.register_structure_hook(
        UnhashableKey, lambda value, _: UnhashableKey()
    )

    with pytest.raises(TypeError, match="unhashable"):
        converter.structure(
            MultiKeyMapping(("1", "a")), Dict[UnhashableKey, str]
        )


def test_mapping_hook_factory_uses_converter_strict_key_validation():
    """Generated mapping hooks use the converter setting by default."""
    converter = Converter(
        strict_key_validation=True, detailed_validation=False
    )
    hook = make_mapping_structure_fn(
        Dict[int, str], converter, detailed_validation=False
    )

    with pytest.raises(MappingKeyCollisionError):
        hook({"1": "甲", "01": "乙"}, Dict[int, str])


@pytest.mark.parametrize("detailed_validation", [True, False])
def test_mapping_hook_factory_can_override_strict_key_validation(
    detailed_validation: bool,
):
    """The factory setting takes precedence over the converter setting."""
    converter = Converter(strict_key_validation=True)
    hook = make_mapping_structure_fn(
        Dict[int, str],
        converter,
        detailed_validation=detailed_validation,
        strict_key_validation=False,
    )

    assert hook({"1": "甲", "01": "乙"}, Dict[int, str]) == {1: "乙"}


def test_copy_preserves_and_overrides_strict_key_validation():
    """Converter copies preserve the strict key setting."""
    converter = Converter(
        strict_key_validation=True, detailed_validation=False
    )
    copied = converter.copy()
    disabled = converter.copy(strict_key_validation=False)

    assert copied.strict_key_validation is True
    assert disabled.strict_key_validation is False
    with pytest.raises(MappingKeyCollisionError):
        copied.structure({"1": "甲", "01": "乙"}, Dict[int, str])
    assert disabled.structure({"1": "甲", "01": "乙"}, Dict[int, str]) == {1: "乙"}


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_strict_key_validation_defaultdict(converter_cls):
    """Generic defaultdicts use the same strict key handling."""
    converter = converter_cls(strict_key_validation=True, detailed_validation=False)

    with pytest.raises(MappingKeyCollisionError):
        converter.structure(
            MultiKeyMapping((1, "a"), (True, "b")), defaultdict[int, str]
        )


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_strict_key_validation_aggregates_other_errors(converter_cls):
    """Detailed validation reports collisions with other mapping errors."""
    converter = converter_cls(
        strict_key_validation=True, detailed_validation=True
    )

    with pytest.raises(IterableValidationError) as exc_info:
        converter.structure(
            MultiKeyMapping((1, 1), (2, "invalid"), ("01", 2)),
            Dict[int, int],
        )

    exceptions = exc_info.value.exceptions
    assert len(exceptions) == 2
    assert isinstance(exceptions[0], ValueError)
    assert isinstance(exceptions[1], MappingKeyCollisionError)
    assert exceptions[1].__notes__[0].index == "01"
    messages = transform_error(exc_info.value)
    assert len(messages) == 2
    assert "both structure to 1" in messages[1]
