"""Tests for the `unknown_variant` option of the tagged union strategy."""

from dataclasses import dataclass
from typing import Any, List, Union

import pytest
from attrs import define

from cattrs import BaseConverter, Converter, transform_error
from cattrs.errors import ClassValidationError, IterableValidationError
from cattrs.strategies import UnknownVariant, configure_tagged_union


@define
class A:
    a: int


@define
class B:
    a: str


Event = Union[A, B]


def test_known_tags_stay_strict(converter: BaseConverter) -> None:
    """Known tags are structured through the member hook, coercing fields."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    assert converter.structure({"_type": "A", "a": "1"}, Event) == A(1)
    assert converter.structure({"_type": "B", "a": 1}, Event) == B("1")

    assert converter.unstructure(A(1), Event) == {"_type": "A", "a": 1}
    assert converter.unstructure(B("1"), Event) == {"_type": "B", "a": "1"}


def test_unknown_tag_wraps(converter: BaseConverter) -> None:
    """Unknown tags are wrapped, preserving the tag and the raw payload."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    res = converter.structure({"_type": "NewEvent", "a": 5, "extra": [1, 2]}, Event)

    assert isinstance(res, UnknownVariant)
    assert res.tag == "NewEvent"
    assert res.value == {"a": 5, "extra": [1, 2]}


def test_unknown_tag_round_trip(converter: BaseConverter) -> None:
    """Wrapped unknown variants unstructure back into the wire structure."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    raw = {"_type": "NewEvent", "a": 5, "extra": [1, 2], "nested": {"k": "v"}}
    res = converter.structure(raw, Event)

    assert converter.unstructure(res, Event) == raw
    # Also without an explicit type, via the registered class hook.
    assert converter.unstructure(res) == raw


def test_missing_tag_is_distinct_and_strict(converter: BaseConverter) -> None:
    """A missing tag raises, unlike an unknown tag which is wrapped."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    with pytest.raises(KeyError):
        converter.structure({"a": 1}, Event)

    # But an unknown tag is wrapped, not an error.
    res = converter.structure({"_type": "Z", "a": 1}, Event)
    assert isinstance(res, UnknownVariant)


def test_known_tag_field_errors_still_raise(converter: BaseConverter) -> None:
    """Field errors on known tags still raise; only unknown tags are wrapped."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    with pytest.raises((ClassValidationError, ValueError)):
        converter.structure({"_type": "A", "a": "not-an-int"}, Event)


def test_strict_default_behavior_unchanged(converter: BaseConverter) -> None:
    """Without `unknown_variant`, unknown and missing tags raise `KeyError`."""
    configure_tagged_union(Event, converter)

    with pytest.raises(KeyError):
        converter.structure({"_type": "NewEvent", "a": 1}, Event)
    with pytest.raises(KeyError):
        converter.structure({"a": 1}, Event)


def test_unknown_variant_false_is_strict(converter: BaseConverter) -> None:
    """`unknown_variant=False` keeps the historical strict behavior."""
    configure_tagged_union(Event, converter, unknown_variant=False)

    with pytest.raises(KeyError):
        converter.structure({"_type": "NewEvent", "a": 1}, Event)


def test_unknown_variant_true_defaults(converter: BaseConverter) -> None:
    """`unknown_variant=True` uses the default `UnknownVariant` wrapper."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    res = converter.structure({"_type": "Z", "a": 1}, Event)
    assert isinstance(res, UnknownVariant)
    assert res.tag == "Z"
    assert res.value == {"a": 1}


def test_unknown_variant_custom_wrapper(converter: BaseConverter) -> None:
    """A custom wrapper class can be used instead of `UnknownVariant`."""

    @define
    class MyUnknown:
        tag: Any
        value: dict

    configure_tagged_union(Event, converter, unknown_variant=MyUnknown)

    res = converter.structure({"_type": "Z", "a": 1}, Event)
    assert isinstance(res, MyUnknown)
    assert res.tag == "Z"
    assert res.value == {"a": 1}
    assert converter.unstructure(res, Event) == {"_type": "Z", "a": 1}


def test_unknown_variant_dataclass_wrapper(converter: BaseConverter) -> None:
    """A dataclass can be used as the wrapper too."""

    @dataclass
    class MyUnknown:
        tag: Any
        value: dict

    configure_tagged_union(Event, converter, unknown_variant=MyUnknown)

    res = converter.structure({"_type": "Z", "a": 1}, Event)
    assert isinstance(res, MyUnknown)
    assert converter.unstructure(res, Event) == {"_type": "Z", "a": 1}


def test_unknown_variant_custom_tag_name(converter: BaseConverter) -> None:
    """The wrapper works with a custom tag name."""
    configure_tagged_union(Event, converter, tag_name="kind", unknown_variant=True)

    res = converter.structure({"kind": "Z", "a": 1}, Event)
    assert isinstance(res, UnknownVariant)
    assert res.tag == "Z"
    assert res.value == {"a": 1}
    assert converter.unstructure(res, Event) == {"kind": "Z", "a": 1}


def test_unknown_variant_forbid_extra_keys() -> None:
    """Known tags are popped for `forbid_extra_keys`; the wrapper drops the tag."""
    c = Converter(forbid_extra_keys=True)
    configure_tagged_union(Event, c, unknown_variant=True)

    assert c.structure({"_type": "A", "a": 1}, Event) == A(1)

    res = c.structure({"_type": "New", "x": 1}, Event)
    assert isinstance(res, UnknownVariant)
    assert res.value == {"x": 1}  # The tag is not part of the payload.
    assert c.unstructure(res, Event) == {"_type": "New", "x": 1}


def test_unknown_variant_non_mapping_input(converter: BaseConverter) -> None:
    """Non-mapping inputs raise a clear `TypeError`."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    with pytest.raises(TypeError):
        converter.structure(123, Event)


def test_unknown_variant_does_not_call_wrapper_hook(converter: BaseConverter) -> None:
    """Unknown tags construct the wrapper directly, not via a registered hook."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    calls = []

    def structure_unknown(value, cl):
        calls.append(value)
        raise AssertionError("the UnknownVariant structure hook must not be called")

    converter.register_structure_hook(UnknownVariant, structure_unknown)

    res = converter.structure({"_type": "Z", "a": 1}, Event)
    assert isinstance(res, UnknownVariant)
    assert not calls


def test_unknown_variant_does_not_import_by_tag(converter: BaseConverter) -> None:
    """A tag that looks like a qualified class path is not imported."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    tag = f"{A.__module__}.{A.__qualname__}"
    res = converter.structure({"_type": tag, "a": 1}, Event)
    assert isinstance(res, UnknownVariant)
    assert res.tag == tag
    assert res.value == {"a": 1}


def test_unknown_variant_unstructure_bypasses_nested_hooks(
    converter: BaseConverter,
) -> None:
    """Unstructuring a wrapped payload restores it verbatim, without hooks."""

    class Opaque:
        pass

    configure_tagged_union(Event, converter, unknown_variant=True)

    calls = []

    def opaque_hook(value):
        calls.append(value)
        raise AssertionError("a nested unstructure hook must not be called")

    converter.register_unstructure_hook(Opaque, opaque_hook)

    obj = Opaque()
    res = converter.structure({"_type": "New", "obj": obj, "n": [obj]}, Event)
    wire = converter.unstructure(res, Event)

    assert not calls
    assert isinstance(wire["obj"], Opaque)
    assert isinstance(wire["n"][0], Opaque)


def test_unknown_variant_no_shared_mutable_state(converter: BaseConverter) -> None:
    """The wrapper deep-copies the payload; inputs and outputs stay isolated."""
    configure_tagged_union(Event, converter, unknown_variant=True)

    raw = {"_type": "Z", "nested": {"k": [1, 2]}, "lst": [{"m": 1}]}
    res = converter.structure(raw, Event)

    # Mutating the input after structuring must not affect the wrapper.
    raw["nested"]["k"].append(99)
    raw["lst"][0]["m"] = 42
    raw["new"] = "x"
    assert res.value == {"nested": {"k": [1, 2]}, "lst": [{"m": 1}]}

    # Mutating the unstructured output must not affect the wrapper either.
    out = converter.unstructure(res, Event)
    out["nested"]["k"].append(7)
    out["lst"].append({"extra": 1})
    assert res.value == {"nested": {"k": [1, 2]}, "lst": [{"m": 1}]}


def test_unknown_variant_wrapper_is_frozen() -> None:
    """The default wrapper is immutable."""
    res = UnknownVariant(tag="t", value={"k": [1]})

    with pytest.raises(AttributeError):
        res.tag = "x"
    with pytest.raises(AttributeError):
        res.value = {}


def test_nested_list_round_trip() -> None:
    """A nested list of old and new events round-trips, preserving unknown fields."""
    c = Converter(detailed_validation=True)
    configure_tagged_union(Event, c, unknown_variant=True)

    new_1 = c.structure({"_type": "New1", "deep": {"k": [1, 2, 3]}}, Event)
    new_2 = c.structure({"_type": "New2", "x": [{"y": 1}]}, Event)
    events = [[A(1), new_1], [new_2, B("s")]]

    wire = c.unstructure(events, List[List[Event]])
    assert wire == [
        [{"a": 1, "_type": "A"}, {"deep": {"k": [1, 2, 3]}, "_type": "New1"}],
        [{"x": [{"y": 1}], "_type": "New2"}, {"a": "s", "_type": "B"}],
    ]

    back = c.structure(wire, List[List[Event]])
    assert back == events
    # Unknown fields are preserved, nested ones included.
    assert back[0][1].value["deep"]["k"] == [1, 2, 3]
    assert back[1][0].value["x"] == [{"y": 1}]


def test_known_bad_event_error_path() -> None:
    """A bad known event in a list reports the correct array path."""
    c = Converter(detailed_validation=True)
    configure_tagged_union(Event, c, unknown_variant=True)

    wire = [
        {"_type": "A", "a": 1},
        {"_type": "A", "a": "bad"},
        {"_type": "New", "z": 9},
    ]

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure(wire, List[Event])

    assert transform_error(exc_info.value) == [
        "invalid value for type, expected int @ $[1].a"
    ]


def test_nested_known_bad_event_error_path() -> None:
    """A bad known event in a nested list reports the nested array path."""
    c = Converter(detailed_validation=True)
    configure_tagged_union(Event, c, unknown_variant=True)

    wire = [
        [{"_type": "A", "a": 1}, {"_type": "New1", "deep": {}}],
        [{"_type": "A", "a": "nope"}, {"_type": "B", "a": "x"}],
    ]

    with pytest.raises(IterableValidationError) as exc_info:
        c.structure(wire, List[List[Event]])

    assert transform_error(exc_info.value) == [
        "invalid value for type, expected int @ $[1][0].a"
    ]


def test_unknown_variant_cannot_combine_with_default() -> None:
    """`unknown_variant` and `default` are mutually exclusive."""
    with pytest.raises(ValueError):
        configure_tagged_union(Event, Converter(), default=A, unknown_variant=True)


def test_unknown_variant_member_field_conflict() -> None:
    """Union members defining wrapper fields are rejected."""

    @define
    class HasTag:
        tag: str

    @define
    class HasValue:
        value: int

    with pytest.raises(ValueError):
        configure_tagged_union(Union[A, HasTag], Converter(), unknown_variant=True)
    with pytest.raises(ValueError):
        configure_tagged_union(Union[A, HasValue], Converter(), unknown_variant=True)


def test_unknown_variant_tag_name_conflict() -> None:
    """A tag name colliding with a wrapper field is rejected."""
    with pytest.raises(ValueError):
        configure_tagged_union(Event, Converter(), tag_name="tag", unknown_variant=True)
    with pytest.raises(ValueError):
        configure_tagged_union(
            Event, Converter(), tag_name="value", unknown_variant=True
        )


def test_unknown_variant_wrapper_as_member_conflict() -> None:
    """The wrapper class cannot also be a union member."""
    with pytest.raises(ValueError):
        configure_tagged_union(
            Union[A, UnknownVariant], Converter(), unknown_variant=True
        )


def test_unknown_variant_invalid_type() -> None:
    """`unknown_variant` must be a class or a bool."""
    with pytest.raises(TypeError):
        configure_tagged_union(Event, Converter(), unknown_variant=123)


def test_unknown_variant_wrapper_contract() -> None:
    """A wrapper missing the required `tag`/`value` fields is rejected."""

    @define
    class BadWrapper:
        tag: Any

    with pytest.raises(ValueError):
        configure_tagged_union(Event, Converter(), unknown_variant=BadWrapper)


