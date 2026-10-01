import pytest

from cattrs import BaseConverter, Converter, transform_error
from cattrs.errors import DuplicateKeysError, IterableValidationError
from cattrs.gen import make_mapping_structure_fn


class SequenceMapping:
    def __init__(self, items):
        self._items = tuple(items)

    def items(self):
        return self._items


@pytest.fixture(params=[True, False])
def detailed_validation(request):
    return request.param


@pytest.fixture(params=(BaseConverter, Converter))
def duplicate_keys_converter(request, detailed_validation):
    return request.param(
        detailed_validation=detailed_validation,
        forbid_duplicate_keys=True,
    )


@pytest.mark.parametrize("detailed_validation", [True, False])
@pytest.mark.parametrize(
    ("payload", "first_key", "duplicate_key", "structured_key"),
    [
        ({"1": "甲", "01": "乙"}, "1", "01", 1),
        (SequenceMapping(((1, "甲"), (True, "乙"))), 1, True, 1),
        (SequenceMapping(((True, "甲"), (1, "乙"))), True, 1, True),
    ],
)
def test_duplicate_mapping_keys(
    duplicate_keys_converter,
    payload,
    first_key,
    duplicate_key,
    structured_key,
):
    with pytest.raises(Exception) as exc_info:
        duplicate_keys_converter.structure(payload, dict[int, str])

    exc = exc_info.value
    if duplicate_keys_converter.detailed_validation:
        assert isinstance(exc, IterableValidationError)
        assert len(exc.exceptions) == 1
        exc = exc.exceptions[0]
        assert exc.__notes__ == [
            f"Structuring mapping key @ key {duplicate_key!r}"
        ]

    assert isinstance(exc, DuplicateKeysError)
    assert exc.first_key == first_key
    assert exc.duplicate_key == duplicate_key
    assert exc.structured_key == structured_key
    assert exc.original_key == duplicate_key
    assert exc.target_key == structured_key
    assert exc.existing_key == first_key


def test_duplicate_mapping_keys_use_hash_and_equality(detailed_validation):
    class Key:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return hash(str(self.value))

        def __eq__(self, other):
            return isinstance(other, Key) and str(self.value) == str(other.value)

    converter = Converter(
        detailed_validation=detailed_validation,
        forbid_duplicate_keys=True,
    )
    converter.register_structure_hook(Key, lambda value, _: Key(int(value)))

    with pytest.raises(Exception) as exc_info:
        converter.structure({"1": "甲", "01": "乙"}, dict[Key, str])

    exc = exc_info.value.exceptions[0] if detailed_validation else exc_info.value
    assert isinstance(exc, DuplicateKeysError)
    assert exc.first_key == "1"
    assert exc.duplicate_key == "01"
    assert exc.structured_key == Key(1)


@pytest.mark.parametrize("detailed_validation", [True, False])
def test_duplicate_keys_for_generated_mapping_hook(detailed_validation):
    converter = Converter()
    hook = make_mapping_structure_fn(
        dict[int, str],
        converter,
        detailed_validation=detailed_validation,
        forbid_duplicate_keys=True,
    )

    with pytest.raises(Exception) as exc_info:
        hook({"1": "甲", "01": "乙"}, dict[int, str])

    exc = exc_info.value.exceptions[0] if detailed_validation else exc_info.value
    assert isinstance(exc, DuplicateKeysError)
    assert (exc.first_key, exc.duplicate_key, exc.structured_key) == ("1", "01", 1)


def test_duplicate_keys_aggregate_other_mapping_errors():
    converter = Converter(
        detailed_validation=True,
        forbid_duplicate_keys=True,
    )

    with pytest.raises(IterableValidationError) as exc_info:
        converter.structure({"1": "bad", "01": 2, "x": 3}, dict[int, int])

    assert [type(exc) for exc in exc_info.value.exceptions] == [
        ValueError,
        ValueError,
    ]
    assert transform_error(exc_info.value) == [
        "invalid value for type, expected int @ $['1']",
        "invalid value for type, expected int @ $['x']",
    ]


def test_duplicate_key_error_with_other_errors():
    converter = Converter(
        detailed_validation=True,
        forbid_duplicate_keys=True,
    )

    with pytest.raises(IterableValidationError) as exc_info:
        converter.structure({"1": 1, "01": 2, "x": "bad"}, dict[int, int])

    assert type(exc_info.value.exceptions[0]) is DuplicateKeysError
    assert type(exc_info.value.exceptions[1]) is ValueError
    assert transform_error(exc_info.value) == [
        "duplicate key 1 from keys '1' and '01' @ $['01']",
        "invalid value for type, expected int @ $['x']",
    ]


@pytest.mark.parametrize("detailed_validation", [True, False])
def test_unhashable_structured_key(detailed_validation):
    converter = Converter(
        detailed_validation=detailed_validation,
        forbid_duplicate_keys=True,
    )
    converter.register_structure_hook(list, lambda value, _: [value])

    with pytest.raises(Exception) as exc_info:
        converter.structure({"a": 1}, dict[list, int])

    if detailed_validation:
        assert isinstance(exc_info.value, IterableValidationError)
        assert type(exc_info.value.exceptions[0]) is TypeError
    else:
        assert type(exc_info.value) is TypeError


def test_default_behavior_still_overwrites_duplicate_structured_keys():
    converter = Converter()
    assert converter.structure({"1": "甲", "01": "乙"}, dict[int, str]) == {1: "乙"}


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_copy_preserves_forbid_duplicate_keys(converter_cls):
    converter = converter_cls(forbid_duplicate_keys=True)
    copied = converter.copy()

    assert copied.forbid_duplicate_keys is True
    with pytest.raises(Exception):
        copied.structure({"1": "甲", "01": "乙"}, dict[int, str])
