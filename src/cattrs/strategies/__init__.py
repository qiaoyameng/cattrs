"""High level strategies for converters."""

from ._class_methods import use_class_methods
from ._subclasses import include_subclasses
from ._unions import configure_tagged_union, configure_union_passthrough
from ._versioned import (
    AmbiguousMigrationPathError,
    DuplicateMigrationError,
    MigrationCycleError,
    MigrationGraphError,
    MigrationRuntimeError,
    MigrationStep,
    MigrationStepError,
    MissingMigrationError,
    UnknownVersionError,
    VersionedPayloadError,
    VersionedRegistry,
    configure_versioned_structure,
    configure_versioned_unstructure,
    export_version,
    register_downgrade,
)

__all__ = [
    "AmbiguousMigrationPathError",
    "DuplicateMigrationError",
    "MigrationCycleError",
    "MigrationGraphError",
    "MigrationRuntimeError",
    "MigrationStep",
    "MigrationStepError",
    "MissingMigrationError",
    "UnknownVersionError",
    "VersionedPayloadError",
    "VersionedRegistry",
    "configure_tagged_union",
    "configure_union_passthrough",
    "configure_versioned_structure",
    "configure_versioned_unstructure",
    "export_version",
    "include_subclasses",
    "register_downgrade",
    "use_class_methods",
]
