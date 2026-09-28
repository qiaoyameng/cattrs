"""Tests for the graph conversion strategy (`cattrs.strategies.GraphSupport`)."""

import dataclasses
import sys
from typing import Any

import pytest
from attrs import define, frozen

from cattrs import Converter
from cattrs.errors import (
    DanglingReferenceError,
    DuplicateReferenceError,
    GraphConversionError,
    GraphCycleError,
    GraphTypeMismatchError,
    UnknownGraphTypeError,
)
from cattrs.strategies import GraphSupport


def tn(cl: type) -> str:
    return f"{cl.__module__}.{cl.__qualname__}"


def node_def(ref_id: int, name: str) -> dict:
    return {
        "$type": tn(Node),
        "$id": ref_id,
        "fields": {"name": name, "child": None},
    }


@define
class Node:
    name: str
    child: "Node | None" = None


@define
class Holder:
    left: Node
    right: Node


@define
class Parent:
    name: str
    child: "Child | None" = None


@define
class Child:
    parent: "Parent | None" = None


@define
class Bag:
    items: list


@define
class Mapping:
    direct: Node
    table: dict[str, Node]


@dataclasses.dataclass
class DNode:
    name: str
    child: "DNode | None" = None


@frozen
class FrozenNode:
    peer: "FrozenNode | None" = None


@dataclasses.dataclass(frozen=True)
class FrozenDataclass:
    peer: "FrozenDataclass | None" = None


@define
class CustomInit:
    peer: "CustomInit | None" = None

    def __init__(self, peer=None):
        self.peer = peer


@define
class AnyHolder:
    payload: Any


@define
class Pair:
    xs: list
    n: "Node | None" = None


@define
class Num:
    n: int
    node: "Node | None" = None


@pytest.fixture
def converter() -> Converter:
    return Converter()


@pytest.fixture
def support(converter) -> GraphSupport:
    return GraphSupport(converter)


class TestSharedReferences:
    def test_shared_child_identity(self, support):
        shared = Node("shared")
        doc = support.unstructure(Holder(shared, shared))

        assert doc["fields"]["right"] == {"$ref": doc["fields"]["left"]["$id"]}

        res = support.structure(doc, Holder)
        assert res.left is res.right
        assert res.left.name == "shared"

    def test_shared_grandchild(self, support):
        shared = Node("leaf")
        holder = Holder(Node("a", shared), Node("b", shared))

        res = support.structure(support.unstructure(holder), Holder)

        assert res.left.child is res.right.child

    def test_shared_through_dict(self, support):
        shared = Node("shared")
        obj = Mapping(shared, {"one": shared, "two": Node("other")})

        res = support.structure(support.unstructure(obj), Mapping)

        assert res.direct is res.table["one"]
        assert res.direct is not res.table["two"]

    def test_shared_dataclass(self, support):
        d = DNode("dc", DNode("leaf"))
        res = support.structure(support.unstructure(d), DNode)
        assert res == d
        assert res is not d
        assert res.child is not d.child


class TestCycles:
    def test_bidirectional_relation(self, support):
        parent = Parent("p")
        child = Child()
        parent.child = child
        child.parent = parent

        res = support.structure(support.unstructure(parent), Parent)

        assert res.child.parent is res
        assert res.child.parent.child.parent is res

    def test_self_referencing_class(self, support):
        node = Node("loop")
        node.child = node

        res = support.structure(support.unstructure(node), Node)

        assert res.child is res

    def test_container_self_reference(self, support):
        bag = Bag([])
        bag.items.append(bag)
        bag.items.append(bag.items)

        res = support.structure(support.unstructure(bag), Bag)

        assert res.items[0] is res
        assert res.items[1] is res.items

    def test_root_list_self_reference(self, support):
        lst: list = []
        lst.append(lst)

        doc = support.unstructure(lst)
        assert doc == {"$id": 0, "items": [{"$ref": 0}]}

        res = support.structure(doc, list)
        assert res[0] is res

    def test_frozen_class_cycle_rejected(self, support):
        doc = {
            "$type": tn(FrozenNode),
            "$id": 0,
            "fields": {"peer": {"$ref": 0}},
        }
        with pytest.raises(GraphCycleError) as exc_info:
            support.structure(doc, FrozenNode)
        assert exc_info.value.path

    def test_frozen_dataclass_cycle_rejected(self, support):
        doc = {
            "$type": tn(FrozenDataclass),
            "$id": 0,
            "fields": {"peer": {"$ref": 0}},
        }
        with pytest.raises(GraphCycleError):
            support.structure(doc, FrozenDataclass)

    def test_custom_constructor_cycle_rejected(self, support):
        doc = {
            "$type": tn(CustomInit),
            "$id": 0,
            "fields": {"peer": {"$ref": 0}},
        }
        with pytest.raises(GraphCycleError) as exc_info:
            support.structure(doc, CustomInit)
        assert exc_info.value.path

    def test_frozen_class_acyclic_roundtrip(self, support):
        obj = FrozenNode(FrozenNode(None))
        res = support.structure(support.unstructure(obj), FrozenNode)
        assert res == obj
        assert res.peer is not obj.peer


class TestMalformedDocuments:
    def test_unknown_type_name(self, support):
        doc = {"$type": "no_such_module_xyz.Evil", "$id": 0, "fields": {}}
        with pytest.raises(UnknownGraphTypeError):
            support.structure(doc, Any)
        assert "no_such_module_xyz" not in sys.modules

    def test_type_name_does_not_trigger_import(self, support):
        assert "tabnanny" not in sys.modules
        doc = {"$type": "tabnanny.Nanny", "$id": 0, "fields": {}}
        with pytest.raises(UnknownGraphTypeError):
            support.structure(doc, Any)
        assert "tabnanny" not in sys.modules

    def test_undeclared_but_importable_type_rejected(self, support):
        doc = {"$type": "pathlib.Path", "$id": 0, "fields": {}}
        with pytest.raises(UnknownGraphTypeError):
            support.structure(doc, Any)

    def test_duplicate_definition(self, support):
        doc = {
            "$type": tn(Holder),
            "$id": 0,
            "fields": {
                "left": node_def(1, "a"),
                "right": node_def(1, "b"),
            },
        }
        with pytest.raises(DuplicateReferenceError) as exc_info:
            support.structure(doc, Holder)
        assert exc_info.value.ref_id == 1

    def test_dangling_reference(self, support):
        with pytest.raises(DanglingReferenceError):
            support.structure({"$ref": 5}, Node)

    def test_forward_reference_is_dangling(self, support):
        doc = {
            "$type": tn(Holder),
            "$id": 0,
            "fields": {
                "left": {"$ref": 1},
                "right": node_def(1, "b"),
            },
        }
        with pytest.raises(DanglingReferenceError):
            support.structure(doc, Holder)

    def test_type_mismatch_class_for_scalar(self, support):
        doc = {
            "$type": tn(Num),
            "$id": 0,
            "fields": {
                "n": node_def(1, "x"),
                "node": None,
            },
        }
        with pytest.raises(GraphTypeMismatchError):
            support.structure(doc, Num)

    def test_type_mismatch_ref_to_list(self, support):
        doc = {
            "$type": tn(Pair),
            "$id": 0,
            "fields": {
                "xs": {"$id": 1, "items": []},
                "n": {"$ref": 1},
            },
        }
        with pytest.raises(GraphTypeMismatchError):
            support.structure(doc, Pair)

    def test_type_mismatch_scalar_for_class(self, support):
        doc = {"$type": tn(Node), "$id": 0, "fields": {"name": "x", "child": 42}}
        with pytest.raises(GraphTypeMismatchError):
            support.structure(doc, Node)

    def test_malformed_nodes(self, support):
        with pytest.raises(GraphConversionError):
            support.structure({"$ref": 1, "extra": 2}, Node)
        with pytest.raises(GraphConversionError):
            support.structure({"$type": tn(Node), "$id": "x", "fields": {}}, Node)


class TestReferenceTableIsolation:
    def test_unstructure_ids_restart(self, support):
        shared = Node("s")
        holder = Holder(shared, shared)

        first = support.unstructure(holder)
        second = support.unstructure(holder)

        assert first == second
        assert support.unstructure(Node("solo"))["$id"] == 0

    def test_structure_tables_not_shared(self, support):
        shared = Node("s")
        doc = support.unstructure(Holder(shared, shared))
        res = support.structure(doc, Holder)
        assert res.left is res.right

        # Id 0 was defined in the previous conversion; it must not leak over.
        with pytest.raises(DanglingReferenceError):
            support.structure({"$ref": 0}, Node)

    def test_conversions_do_not_share_objects(self, support):
        shared = Node("s")
        doc = support.unstructure(Holder(shared, shared))

        first = support.structure(doc, Holder)
        second = support.structure(doc, Holder)

        assert first.left is not second.left
        assert first.left == second.left


class TestConverterInterplay:
    def test_normal_converter_tree_format_unchanged(self, converter):
        shared = Node("s")
        holder = Holder(shared, shared)

        tree = converter.unstructure(holder)
        assert tree == {
            "left": {"name": "s", "child": None},
            "right": {"name": "s", "child": None},
        }

        res = converter.structure(tree, Holder)
        assert res.left is not res.right  # tree semantics copy shared objects

    def test_scalar_hooks_are_used(self, converter):
        class Weird:
            def __init__(self, v):
                self.v = v

            def __eq__(self, other):
                return self.v == other.v

        @define
        class HasWeird:
            w: Weird
            n: int

        converter.register_unstructure_hook(Weird, lambda w: {"v": w.v})
        converter.register_structure_hook(Weird, lambda d, _: Weird(d["v"]))

        support = GraphSupport(converter, allowed_types=(HasWeird,))
        obj = HasWeird(Weird(3), 4)

        doc = support.unstructure(obj)
        assert doc["fields"]["w"] == {"v": 3}

        res = support.structure(doc, HasWeird)
        assert res == obj

    def test_allowed_types_for_any_fields(self, converter):
        obj = AnyHolder(Node("payload"))
        doc = GraphSupport(converter).unstructure(obj)

        with pytest.raises(UnknownGraphTypeError):
            GraphSupport(converter).structure(doc, AnyHolder)

        res = GraphSupport(converter, allowed_types=(Node,)).structure(doc, AnyHolder)
        assert res.payload == Node("payload")
