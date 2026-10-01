from dataclasses import dataclass
from typing import Union

import pytest
from attrs import define

from cattrs import BaseConverter, Converter


@define
class AttrsLeaf:
    value: int


@define
class AttrsBranch:
    first: AttrsLeaf
    rest: list[AttrsLeaf]
    by_key: dict[str, AttrsLeaf]
    maybe: AttrsLeaf | None


@dataclass
class DataclassBuddy:
    code: int


@dataclass
class DataclassBranch:
    leaves: tuple[AttrsLeaf, list[AttrsLeaf]]
    by_key: dict[str, AttrsLeaf]
    maybe: Union[AttrsLeaf, DataclassBuddy, None]


@define
class ReentryMark:
    path: str


CONTEXTS = [
    {"tenant": "a", "lang": "en", "depth": 0},
    {"tenant": "b", "lang": "fr", "depth": 0},
]


def install_hooks(converter):
    seen: list[tuple[str, object, str]] = []

    def unstructure_mark(mark):
        return {"path": mark.path}

    def structure_mark(data, _):
        return ReentryMark(data["path"])

    def unstructure_leaf(leaf, *, context):
        seen.append(("unstructure", context, f"leaf:{leaf.value}"))
        inherited = converter.unstructure(
            ReentryMark(f"inherit:{leaf.value}"), context=context
        )
        replaced = converter.unstructure(
            ReentryMark(f"replace:{leaf.value}"),
            context={**context, "depth": context["depth"] + 1},
        )
        assert inherited == {"path": f"inherit:{leaf.value}"}
        assert replaced == {"path": f"replace:{leaf.value}"}
        return {"value": leaf.value}

    def structure_leaf(data, _, *, context):
        seen.append(("structure", context, f"leaf:{data['value']}"))
        inherited = converter.structure(
            {"path": f"inherit:{data['value']}"}, ReentryMark, context=context
        )
        replaced = converter.structure(
            {"path": f"replace:{data['value']}"},
            ReentryMark,
            context={**context, "depth": context["depth"] + 1},
        )
        assert inherited.path == f"inherit:{data['value']}"
        assert replaced.path == f"replace:{data['value']}"
        return AttrsLeaf(data["value"])

    converter.register_unstructure_hook(ReentryMark, unstructure_mark)
    converter.register_structure_hook(ReentryMark, structure_mark)
    converter.register_unstructure_hook(
        DataclassBuddy,
        lambda buddy, *, context: (
            seen.append(("unstructure", context, f"buddy:{buddy.code}"))
            or {"code": buddy.code}
        ),
    )
    converter.register_structure_hook(
        DataclassBuddy,
        lambda data, _, *, context: (
            seen.append(("structure", context, f"buddy:{data['code']}"))
            or DataclassBuddy(data["code"])
        ),
    )
    converter.register_unstructure_hook(AttrsLeaf, unstructure_leaf)
    converter.register_structure_hook(AttrsLeaf, structure_leaf)
    return seen


def assert_reentry_contexts(seen):
    for _, context, path in seen:
        if path.startswith(("inherit:", "replace:")):
            value = int(path.rsplit(":", 1)[1])
            assert context["tenant"] == (
                "a" if value in {1, 2, 3, 4} else "b"
            )
            assert context["depth"] == (1 if path.startswith("replace:") else 0)


def test_context_reaches_baseconverter_nested_attrs_and_containers():
    converter = BaseConverter()
    seen = install_hooks(converter)

    converter.register_unstructure_hook_func(
        lambda t: getattr(t, "__origin__", None) is list,
        lambda values, *, context: [
            converter.unstructure(v, context=context) for v in values
        ],
    )
    converter.register_structure_hook_func(
        lambda t: getattr(t, "__origin__", None) is list,
        lambda values, t, *, context: [
            converter.structure(v, t.__args__[0], context=context) for v in values
        ],
    )
    converter.register_unstructure_hook_func(
        lambda t: getattr(t, "__origin__", None) is dict,
        lambda values, *, context: {
            key: converter.unstructure(value, context=context)
            for key, value in values.items()
        },
    )
    converter.register_structure_hook_func(
        lambda t: getattr(t, "__origin__", None) is dict,
        lambda values, t, *, context: {
            key: converter.structure(
                value, t.__args__[1], context=context
            )
            for key, value in values.items()
        },
    )

    values = [
        AttrsBranch(
            AttrsLeaf(1),
            [AttrsLeaf(2)],
            {"x": AttrsLeaf(3)},
            AttrsLeaf(4),
        ),
        AttrsBranch(
            AttrsLeaf(5),
            [AttrsLeaf(6)],
            {"y": AttrsLeaf(7)},
            None,
        ),
    ]

    payloads: dict[int, dict] = {}
    for index in (0, 1, 1, 0):
        payloads[index] = converter.unstructure(values[index], context=CONTEXTS[index])
        structured = converter.structure(
            payloads[index], AttrsBranch, context=CONTEXTS[index]
        )
        assert structured == values[index]
    assert converter.get_unstructure_hook(AttrsLeaf) is converter.get_unstructure_hook(
        AttrsLeaf
    )
    assert converter.get_structure_hook(AttrsLeaf) is converter.get_structure_hook(
        AttrsLeaf
    )

    event_values = [
        [1, 2, 3, 4],
        [5, 6, 7],
        [5, 6, 7],
        [1, 2, 3, 4],
    ]
    expected = [
        (operation, CONTEXTS[index], f"leaf:{value}")
        for index, vals in zip((0, 1, 1, 0), event_values)
        for operation in ("unstructure", "structure")
        for value in vals
    ]
    assert [event for event in seen if event[2].startswith("leaf:")] == expected
    assert_reentry_contexts(seen)


def test_context_reaches_generated_dataclasses_containers_and_union():
    converter = Converter()
    seen = install_hooks(converter)

    values = [
        DataclassBranch(
            leaves=(AttrsLeaf(1), [AttrsLeaf(2)]),
            by_key={"x": AttrsLeaf(3)},
            maybe=AttrsLeaf(4),
        ),
        DataclassBranch(
            leaves=(AttrsLeaf(5), [AttrsLeaf(6)]),
            by_key={"y": AttrsLeaf(7)},
            maybe=DataclassBuddy(8),
        ),
    ]

    payloads: dict[int, dict] = {}
    for index in (0, 1, 1, 0):
        payloads[index] = converter.unstructure(values[index], context=CONTEXTS[index])
        structured = converter.structure(
            payloads[index], DataclassBranch, context=CONTEXTS[index]
        )
        assert structured == values[index]

    event_values = [[1, 2, 3, 4], [5, 6, 7], [5, 6, 7], [1, 2, 3, 4]]
    expected = [
        (operation, CONTEXTS[index], f"leaf:{value}")
        for index, vals in zip((0, 1, 1, 0), event_values)
        for operation in ("unstructure", "structure")
        for value in vals
    ]
    assert [event for event in seen if event[2].startswith("leaf:")] == expected
    assert [
        event for event in seen if event[2].startswith("buddy:")
    ] == [
        event
        for _ in (0, 1)
        for event in (
            ("unstructure", CONTEXTS[1], "buddy:8"),
            ("structure", CONTEXTS[1], "buddy:8"),
        )
    ]
    assert_reentry_contexts(seen)


@pytest.mark.parametrize("converter_cls", [BaseConverter, Converter])
def test_legacy_hooks_remain_callable_and_receive_no_context(converter_cls):
    converter = converter_cls()
    calls = []

    def legacy_unstructure(value):
        calls.append(("unstructure", value))
        return {"value": value.value}

    def legacy_structure(data, type_):
        calls.append(("structure", data, type_))
        return AttrsLeaf(data["value"])

    converter.register_unstructure_hook(AttrsLeaf, legacy_unstructure)
    converter.register_structure_hook(AttrsLeaf, legacy_structure)

    context = {"tenant": "a", "lang": "en"}
    assert converter.unstructure(AttrsLeaf(8), context=context) == {"value": 8}
    assert converter.structure({"value": 9}, AttrsLeaf, context=context) == AttrsLeaf(9)
    assert calls[0] == ("unstructure", AttrsLeaf(8))
    assert calls[1][:2] == ("structure", {"value": 9})
