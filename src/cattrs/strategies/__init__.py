"""High level strategies for converters."""

from ._class_methods import use_class_methods
from ._subclasses import include_subclasses
from ._unions import (
    configure_tagged_union,
    configure_union_passthrough,
    configure_union_probing,
)

__all__ = [
    "configure_tagged_union",
    "configure_union_passthrough",
    "configure_union_probing",
    "include_subclasses",
    "use_class_methods",
]
