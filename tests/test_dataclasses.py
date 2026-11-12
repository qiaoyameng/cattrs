import dataclasses
from typing import Generic, List, TypeVar

import attr
import pytest

from cattrs import BaseConverter, Converter
from cattrs.errors import ClassValidationError

from ._compat import is_py310_plus


@dataclasses.dataclass
class Foo:
    bar: str


@attr.define
class Container:
    foos: List[Foo]


def test_dataclasses_in_attrs(converter: BaseConverter):
    struct = Container([Foo("bar")])

    unstruct = {"foos": [{"bar": "bar"}]}

    assert converter.unstructure(struct) == unstruct
    assert converter.structure(unstruct, Container) == struct


def test_dataclasses_in_container(converter: BaseConverter):
    struct = [Foo("bar"), Foo("bat")]

    unstruct = [{"bar": "bar"}, {"bar": "bat"}]

    assert converter.unstructure(struct) == unstruct
    assert converter.structure(unstruct, List[Foo]) == struct


def test_dataclasses(converter: BaseConverter):
    struct = Foo("bar")

    unstruct = {"bar": "bar"}

    assert converter.unstructure(struct) == unstruct
    assert converter.structure(unstruct, Foo) == struct


@pytest.mark.skipif(not is_py310_plus, reason="kwonly fields are Python 3.10+")
def test_kw_only_propagation(converter: BaseConverter):
    """KW-only args work.

    Reproducer from https://github.com/python-attrs/cattrs/issues/637.
    """

    @dataclasses.dataclass
    class PartialKeywords:
        a1: str = "Default"
        a2: str = dataclasses.field(kw_only=True)

    assert converter.structure({"a2": "Value"}, PartialKeywords) == PartialKeywords(
        a1="Default", a2="Value"
    )


def test_init_var_input_and_derived_field(converter: BaseConverter):
    """InitVars are constructor inputs, `init=False` fields are derived."""

    @dataclasses.dataclass
    class A:
        x: int
        raw: dataclasses.InitVar[int]
        derived: int = dataclasses.field(init=False)

        def __post_init__(self, raw):
            self.derived = self.x + raw

    instance = converter.structure({"x": 1, "raw": 2, "derived": 99}, A)
    # The derived field is produced by `__post_init__`, not set from input.
    assert instance == A(1, 2)
    assert instance.derived == 3

    unstructured = converter.unstructure(instance)
    assert "raw" not in unstructured
    if isinstance(converter, Converter):
        # The generated converter skips `init=False` fields by default.
        assert "derived" not in unstructured
    else:
        assert unstructured == {"x": 1, "derived": 3}


def test_init_var_inheritance_kw_only_generic(converter: BaseConverter):
    """Inherited, kw-only, generic dataclasses construct exactly once."""

    T = TypeVar("T")
    calls = {"factory": 0, "post_init": 0}

    def factory():
        calls["factory"] += 1
        return [calls["factory"]]

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Base:
        base_field: int
        raw_base: dataclasses.InitVar[int] = 0

        def __post_init__(self, raw_base):
            calls["post_init"] += 1
            object.__setattr__(self, "derived_base", self.base_field + raw_base)

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Middle(Base, Generic[T]):
        middle_field: T
        raw_middle: dataclasses.InitVar[str] = "!"

        def __post_init__(self, raw_base, raw_middle):
            calls["post_init"] += 1
            super().__post_init__(raw_base)
            object.__setattr__(
                self, "derived_middle", str(self.middle_field) + raw_middle
            )

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Leaf(Middle[int]):
        items: list = dataclasses.field(default_factory=factory)
        derived_init_false: int = dataclasses.field(init=False, default=7)

    instance = converter.structure(
        {
            "base_field": "1",
            "middle_field": "2",
            "raw_base": "10",
            "raw_middle": "?",
            "derived_init_false": 99,
        },
        Leaf,
    )

    # One construction: the factory once, and both chained `__post_init__`
    # hooks once each.
    assert calls == {"factory": 1, "post_init": 2}
    assert instance.base_field == 1
    assert instance.middle_field == 2
    assert instance.items == [1]
    assert instance.derived_base == 11
    assert instance.derived_middle == "2?"
    # The `init=False` field ignores the input and keeps its own default.
    assert instance.derived_init_false == 7

    unstructured = converter.unstructure(instance)
    assert "raw_base" not in unstructured
    assert "raw_middle" not in unstructured
    assert unstructured["base_field"] == 1
    assert unstructured["middle_field"] == 2


def test_init_var_conversion_error(converter: BaseConverter):
    """InitVar conversion failures are reported accurately, input stays intact."""

    @dataclasses.dataclass
    class A:
        x: int
        raw: dataclasses.InitVar[int]

        def __post_init__(self, raw):
            raise AssertionError("Must not run")

    data = {"x": "1", "raw": "not-an-int"}

    with pytest.raises(Exception) as exc:
        converter.structure(data, A)

    if isinstance(converter, Converter) and converter.detailed_validation:
        assert isinstance(exc.value, ClassValidationError)
        assert exc.value.exceptions[0].__notes__ == [
            "Structuring class test_init_var_conversion_error.<locals>.A "
            "@ attribute raw"
        ]
    else:
        assert isinstance(exc.value, ValueError)

    assert data == {"x": "1", "raw": "not-an-int"}


def test_init_var_missing_required(converter: BaseConverter):
    """Missing required InitVars have a precise path."""

    @dataclasses.dataclass
    class A:
        x: int
        raw: dataclasses.InitVar[int]

        def __post_init__(self, raw):
            raise AssertionError("Must not run")

    data = {"x": "1"}

    with pytest.raises(Exception) as exc:
        converter.structure(data, A)

    if isinstance(converter, Converter) and converter.detailed_validation:
        assert isinstance(exc.value, ClassValidationError)
        assert exc.value.exceptions[0].__notes__ == [
            "Structuring class test_init_var_missing_required.<locals>.A "
            "@ attribute raw"
        ]
    else:
        assert isinstance(exc.value, (KeyError, TypeError))
        assert "raw" in str(exc.value)

    assert data == {"x": "1"}


def test_init_var_post_init_error(converter: BaseConverter):
    """`__post_init__` failures run once and propagate their source."""

    calls = {"post_init": 0}

    @dataclasses.dataclass
    class A:
        x: int
        raw: dataclasses.InitVar[int]

        def __post_init__(self, raw):
            calls["post_init"] += 1
            raise ValueError("raw is bad")

    data = {"x": 1, "raw": 2}

    with pytest.raises(Exception) as exc:
        converter.structure(data, A)

    if isinstance(converter, Converter) and converter.detailed_validation:
        assert isinstance(exc.value, ClassValidationError)
        assert type(exc.value.exceptions[0]) is ValueError
        assert str(exc.value.exceptions[0]) == "raw is bad"
    else:
        assert type(exc.value) is ValueError
        assert str(exc.value) == "raw is bad"

    assert calls == {"post_init": 1}
    assert data == {"x": 1, "raw": 2}
