"""Tests for `dataclasses.InitVar` support."""

import dataclasses
from dataclasses import InitVar
from typing import Generic, List, TypeVar

import pytest

from cattrs import BaseConverter, Converter
from cattrs.errors import ClassValidationError
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn, override

from ._compat import is_py310_plus

T = TypeVar("T")


@dataclasses.dataclass
class WithInitVar:
    iv: InitVar[int]
    x: int = 0

    def __post_init__(self, iv: int):
        self.x = iv * 2


@dataclasses.dataclass
class WithDefaultInitVar:
    x: int
    iv: InitVar[int] = 10

    def __post_init__(self, iv: int):
        self.x += iv


@dataclasses.dataclass
class Inner:
    n: int


@dataclasses.dataclass
class WithNestedInitVar:
    inner: InitVar[Inner]
    total: int = 0

    def __post_init__(self, inner: Inner):
        self.total = inner.n


@dataclasses.dataclass
class WithGenericInitVar(Generic[T]):
    iv: InitVar[T]
    items: List[T] = dataclasses.field(default_factory=list)

    def __post_init__(self, iv: T):
        self.items = [iv]


def test_initvar_roundtrip(converter: BaseConverter):
    """InitVars are structured by declared type, and never unstructured."""
    res = converter.structure({"iv": "3"}, WithInitVar)

    assert res == WithInitVar(3)
    assert res.x == 6

    assert converter.unstructure(res) == {"x": 6}


def test_initvar_default(converter: BaseConverter):
    """InitVars with defaults may be omitted when structuring."""
    res = converter.structure({"x": 1}, WithDefaultInitVar)

    assert res.x == 11
    assert converter.unstructure(res) == {"x": 11}


def test_initvar_post_init_called_once(genconverter: Converter):
    """The constructor (and `__post_init__`) runs exactly once."""
    calls = []

    @dataclasses.dataclass
    class Counted:
        iv: InitVar[int]
        x: int = 0

        def __post_init__(self, iv: int):
            calls.append(iv)
            self.x = iv

    res = genconverter.structure({"iv": 1}, Counted)

    assert res.x == 1
    assert calls == [1]


def test_missing_required_initvar(genconverter: Converter):
    """A missing required InitVar is an input validation failure."""
    with pytest.raises((ClassValidationError, KeyError)):
        genconverter.structure({}, WithInitVar)

    detailed = Converter(detailed_validation=True)
    with pytest.raises(ClassValidationError) as exc_info:
        detailed.structure({}, WithInitVar)
    # The missing key is reported as an input (KeyError) failure.
    assert isinstance(exc_info.value.exceptions[0], KeyError)


def test_nested_initvar_conversion_failure():
    """A failing nested InitVar conversion is an input validation failure."""
    converter = Converter(detailed_validation=True)

    with pytest.raises(ClassValidationError) as exc_info:
        converter.structure({"inner": {"n": "not-an-int"}}, WithNestedInitVar)

    # The inner class validation failure is wrapped, with a note attached.
    inner_error = exc_info.value.exceptions[0]
    assert isinstance(inner_error, ClassValidationError)
    assert inner_error.cl is Inner
    assert any("@ attribute inner" in str(n) for n in inner_error.__notes__)


def test_post_init_error_is_constructor_failure():
    """Errors raised by user constructors are distinguished from validation."""
    calls = []

    @dataclasses.dataclass
    class Validated:
        iv: InitVar[int]

        def __post_init__(self, iv: int):
            calls.append(iv)
            if iv < 0:
                raise ValueError("iv must be non-negative")

    converter = Converter(detailed_validation=True)

    with pytest.raises(ClassValidationError) as exc_info:
        converter.structure({"iv": -1}, Validated)

    # The user's own exception is wrapped directly, not a validation error.
    assert len(exc_info.value.exceptions) == 1
    assert isinstance(exc_info.value.exceptions[0], ValueError)
    assert str(exc_info.value.exceptions[0]) == "iv must be non-negative"
    # `__post_init__` ran exactly once.
    assert calls == [-1]


def test_generic_initvar(genconverter: Converter):
    """Generic InitVars are resolved through the generic mapping."""
    res = genconverter.structure({"iv": "1"}, WithGenericInitVar[int])

    assert res.items == [1]
    assert genconverter.unstructure(res) == {"items": [1]}


def test_initvar_inheritance_field_order(genconverter: Converter):
    """InitVars keep their declaration order when inherited."""

    @dataclasses.dataclass
    class Base:
        a: int
        iv1: InitVar[int]

        def __post_init__(self, iv1: int):
            self.a += iv1

    @dataclasses.dataclass
    class Child(Base):
        b: int = 0
        iv2: InitVar[int] = 0

        def __post_init__(self, iv1: int, iv2: int):
            super().__post_init__(iv1)
            self.b += iv2

    res = genconverter.structure({"a": 1, "iv1": 2, "b": 3, "iv2": 4}, Child)

    assert res.a == 3
    assert res.b == 7
    assert genconverter.unstructure(res) == {"a": 3, "b": 7}


def test_frozen_initvar(genconverter: Converter):
    """Frozen dataclasses with InitVars use the normal constructor."""

    @dataclasses.dataclass(frozen=True)
    class Frozen:
        iv: InitVar[int]
        x: int = 0

        def __post_init__(self, iv: int):
            object.__setattr__(self, "x", iv * 2)

    res = genconverter.structure({"iv": 5}, Frozen)

    assert res.x == 10
    assert genconverter.unstructure(res) == {"x": 10}


@pytest.mark.skipif(not is_py310_plus, reason="slots dataclasses are 3.10+")
def test_slots_initvar(genconverter: Converter):
    """Slotted dataclasses with InitVars use the normal constructor."""

    @dataclasses.dataclass(slots=True)
    class Slotted:
        iv: InitVar[int]
        x: int = 0

        def __post_init__(self, iv: int):
            self.x = iv * 2

    res = genconverter.structure({"iv": 5}, Slotted)

    assert res.x == 10
    assert genconverter.unstructure(res) == {"x": 10}


def test_initvar_overrides(genconverter: Converter):
    """InitVars honor the converter field override machinery."""

    @dataclasses.dataclass
    class Renamed:
        iv: InitVar[int]
        x: int = 0

        def __post_init__(self, iv: int):
            self.x = iv

    @dataclasses.dataclass
    class Other:
        iv: int

    genconverter.register_structure_hook(
        Renamed, make_dict_structure_fn(Renamed, genconverter, iv=override(rename="IV"))
    )
    genconverter.register_unstructure_hook(
        Renamed, make_dict_unstructure_fn(Renamed, genconverter)
    )

    res = genconverter.structure({"IV": "5"}, Renamed)
    assert res.x == 5
    assert genconverter.unstructure(res) == {"x": 5}

    # Hooks for a same-named normal attribute on another class are unaffected.
    other = genconverter.structure({"iv": 1}, Other)
    assert other.iv == 1
    assert genconverter.unstructure(other) == {"iv": 1}


def test_initvar_omit_override(genconverter: Converter):
    """Omitted InitVars are not read from the input."""

    @dataclasses.dataclass
    class Omitted:
        iv: InitVar[int] = 4
        x: int = 0

        def __post_init__(self, iv: int):
            self.x = iv

    genconverter.register_structure_hook(
        Omitted, make_dict_structure_fn(Omitted, genconverter, iv=override(omit=True))
    )

    assert genconverter.structure({"iv": 100}, Omitted).x == 4


def test_initvar_forbid_extra_keys(genconverter: Converter):
    """InitVar keys are allowed when extra keys are forbidden."""
    genconverter.register_structure_hook(
        WithInitVar,
        make_dict_structure_fn(
            WithInitVar, genconverter, _cattrs_forbid_extra_keys=True
        ),
    )

    assert genconverter.structure({"iv": 2}, WithInitVar).x == 4
