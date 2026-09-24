"""Tests for `dataclasses.InitVar` support."""

from dataclasses import InitVar, dataclass
from typing import Generic, TypeVar

import pytest

from cattrs import BaseConverter, Converter
from cattrs.errors import ClassValidationError
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from ._compat import is_py310_plus

T = TypeVar("T")


@dataclass
class WithInitVar:
    a: int
    iv: InitVar[int] = 2
    b: str = "x"
    derived: int = 0

    def __post_init__(self, iv: int):
        self.derived = iv * 10


def test_initvar_structured_and_converted(converter: BaseConverter):
    """InitVars are structured from the mapping, converted per their type."""
    res = converter.structure({"a": "1", "iv": "3"}, WithInitVar)

    assert res == WithInitVar(1, 3)
    assert res.derived == 30


def test_initvar_default_used_when_missing(converter: BaseConverter):
    """A missing InitVar with a default falls back to the default."""
    res = converter.structure({"a": 1}, WithInitVar)

    assert res == WithInitVar(1, 2)


def test_initvar_roundtrip(converter: BaseConverter):
    """A successful round-trip; InitVars are not read back when unstructuring."""
    inst = WithInitVar(1, 2, "y")

    unstructured = converter.unstructure(inst)

    assert unstructured == {"a": 1, "b": "y", "derived": 20}
    assert converter.structure(unstructured, WithInitVar) == inst


def test_initvar_not_an_instance_attribute(converter: BaseConverter):
    """Unstructuring must not read InitVars off the instance."""
    fn = make_dict_unstructure_fn(WithInitVar, converter)

    assert fn(WithInitVar(1, 3)) == {"a": 1, "b": "x", "derived": 30}
    assert "iv" not in fn.overrides


def test_initvar_post_init_called_once(converter: BaseConverter):
    """`__post_init__` runs exactly once, as part of normal construction."""
    calls = []

    @dataclass
    class Counted:
        iv: InitVar[int]

        def __post_init__(self, iv: int):
            calls.append(iv)

    converter.structure({"iv": 1}, Counted)

    assert calls == [1]


def test_initvar_required_missing():
    """A missing required InitVar is an input validation failure."""
    converter = Converter(detailed_validation=True)

    @dataclass
    class Required:
        a: int
        iv: InitVar[int]

        def __post_init__(self, iv: int):
            pass

    with pytest.raises(ClassValidationError) as exc_info:
        converter.structure({"a": 1}, Required)

    with_notes, without_notes = exc_info.value.group_exceptions()
    assert len(with_notes) == 1
    exc, note = with_notes[0]
    assert isinstance(exc, KeyError)
    assert note.name == "iv"
    assert not without_notes


def test_initvar_nested_conversion_failure():
    """A failing nested InitVar conversion is an input validation failure."""

    @dataclass
    class Inner:
        x: int

    @dataclass
    class Outer:
        inner: InitVar[Inner]
        y: int = 0

        def __post_init__(self, inner: Inner):
            self.y = inner.x

    converter = Converter(detailed_validation=True)

    with pytest.raises(ClassValidationError) as exc_info:
        converter.structure({"inner": {"x": "not-an-int"}}, Outer)

    with_notes, without_notes = exc_info.value.group_exceptions()
    assert len(with_notes) == 1
    _, note = with_notes[0]
    assert note.name == "inner"
    assert not without_notes


def test_initvar_post_init_failure():
    """An error raised by the user's `__post_init__` is reported as a
    constructor failure, not an input validation failure."""

    @dataclass
    class Raises:
        iv: InitVar[int]

        def __post_init__(self, iv: int):
            raise ValueError("boom")

    converter = Converter(detailed_validation=True)

    with pytest.raises(ClassValidationError) as exc_info:
        converter.structure({"iv": 1}, Raises)

    with_notes, without_notes = exc_info.value.group_exceptions()
    assert not with_notes
    assert len(without_notes) == 1
    assert isinstance(without_notes[0], ValueError)
    assert str(without_notes[0]) == "boom"


def test_initvar_generic(converter: BaseConverter):
    """Generic InitVars are resolved and converted per the type parameters."""

    @dataclass
    class G(Generic[T]):
        iv: InitVar[list[T]]
        total: int = 0

        def __post_init__(self, iv: list[T]):
            self.total = sum(iv)

    res = converter.structure({"iv": ["1", "2"]}, G[int])

    assert res.total == 3


def test_initvar_inheritance_field_order(converter: BaseConverter):
    """InitVars keep their `__init__` position when inherited and overridden."""

    @dataclass
    class Base:
        a: int
        iv: InitVar[int]

        def __post_init__(self, iv: int):
            self.a = self.a + iv

    @dataclass
    class Child(Base):
        b: str = "y"
        iv: InitVar[int] = 10

    # `iv` keeps its original position (after `a`), despite the override.
    res = converter.structure({"a": "1", "iv": "3", "b": "z"}, Child)
    assert res == Child(1, 3, "z")

    # The overridden default is used when the key is missing.
    res = converter.structure({"a": 1}, Child)
    assert res == Child(1, 10)


def test_initvar_frozen(converter: BaseConverter):
    """Frozen dataclasses go through the normal construction path."""

    @dataclass(frozen=True)
    class Frozen:
        a: int
        iv: InitVar[int] = 5

        def __post_init__(self, iv: int):
            object.__setattr__(self, "a", self.a + iv)

    assert converter.structure({"a": 1}, Frozen) == Frozen(1, 5)


@pytest.mark.skipif(not is_py310_plus, reason="slots require Python 3.10+")
def test_initvar_slots(converter: BaseConverter):
    """Slotted dataclasses go through the normal construction path."""

    @dataclass(slots=True)
    class Slotted:
        a: int
        iv: InitVar[int] = 2

        def __post_init__(self, iv: int):
            self.a = self.a * iv

    assert converter.structure({"a": 3, "iv": 4}, Slotted) == Slotted(3, 4)


def test_initvar_rename_and_omit(converter: BaseConverter):
    """InitVars honor the field override mechanism, like normal fields."""
    fn = make_dict_structure_fn(WithInitVar, converter, iv=override(rename="IV"))

    assert fn({"a": 1, "IV": 4}, WithInitVar) == WithInitVar(1, 4)

    fn = make_dict_structure_fn(WithInitVar, converter, iv=override(omit=True))

    assert fn({"a": 1}, WithInitVar) == WithInitVar(1, 2)


def test_initvar_override_does_not_leak(converter: BaseConverter):
    """Overriding an InitVar on one class doesn't affect same-named fields
    on other classes."""

    @dataclass
    class Other:
        iv: int

    make_dict_structure_fn(WithInitVar, converter, iv=override(rename="IV"))

    assert converter.structure({"iv": 5}, Other) == Other(5)
    assert converter.structure({"a": 1, "iv": 3}, WithInitVar) == WithInitVar(1, 3)


def test_initvar_base_converter():
    """The `BaseConverter` dict structuring path supports InitVars."""
    converter = BaseConverter()

    res = converter.structure({"a": "1", "iv": "3"}, WithInitVar)

    assert res == WithInitVar(1, 3)
    assert converter.unstructure(res) == {"a": 1, "b": "x", "derived": 30}


def test_initvar_gen_converter():
    """The `Converter` (gen) path supports InitVars."""
    converter = Converter()

    res = converter.structure({"a": "1", "iv": "3"}, WithInitVar)

    assert res == WithInitVar(1, 3)
