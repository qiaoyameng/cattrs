"""Strategy for versioned payloads with explicit, graph-checked migrations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from attrs import Factory, frozen

from .._compat import ExceptionGroup, fields, has as has_class
from ..converters import BaseConverter
from ..errors import AttributeValidationNote, CattrsError

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
    "configure_versioned_structure",
    "configure_versioned_unstructure",
    "export_version",
    "register_downgrade",
]

T = TypeVar("T")
Version = Any
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]
_MISSING = object()


class MigrationGraphError(CattrsError):
    """Base class for errors detected while building a migration graph."""


class DuplicateMigrationError(MigrationGraphError):
    """Raised when two migrations share the same source version for a model."""

    def __init__(self, cl: type, version: Version) -> None:
        self.cl = cl
        self.version = version
        super().__init__(
            f"Duplicate migration registered for {cl.__name__} version {version!r}."
        )


class MigrationCycleError(MigrationGraphError):
    """Raised when the registered migrations form a cycle."""

    def __init__(self, cl: type, cycle: list[Version]) -> None:
        self.cl = cl
        self.cycle = cycle
        super().__init__(
            f"Migration cycle detected for {cl.__name__}: "
            f"{' -> '.join(repr(v) for v in cycle)}."
        )


class AmbiguousMigrationPathError(MigrationGraphError):
    """Raised when more than one migration path reaches the target version."""

    def __init__(
        self, cl: type, version: Version, paths: list[list[MigrationStep]]
    ) -> None:
        self.cl = cl
        self.version = version
        self.paths = paths
        super().__init__(
            f"Ambiguous migration paths for {cl.__name__} version {version!r}: "
            f"{len(paths)} paths reach the target."
        )


class MigrationRuntimeError(CattrsError):
    """Base class for errors detected while applying migrations."""


class VersionedPayloadError(MigrationRuntimeError):
    """Raised when an input payload is not a valid versioned document.

    This covers payloads that are not mappings, are missing the declared
    version field, or carry a value outside the restricted JSON domain.
    """


class UnknownVersionError(MigrationRuntimeError):
    """Raised when a payload version is not part of the registered graph."""

    def __init__(self, cl: type, version: Version, target: Version) -> None:
        self.cl = cl
        self.version = version
        self.target = target
        super().__init__(
            f"Unknown version {version!r} for {cl.__name__}; "
            f"the highest supported version is {target!r}."
        )


class MissingMigrationError(MigrationRuntimeError):
    """Raised when no path connects the payload version to the target version."""

    def __init__(self, cl: type, version: Version, target: Version) -> None:
        self.cl = cl
        self.version = version
        self.target = target
        super().__init__(
            f"No migration path for {cl.__name__} from version "
            f"{version!r} to {target!r}."
        )


class MigrationStepError(MigrationRuntimeError):
    """Raised when a single migration step fails.

    :attr step: the step that failed, including its source and target versions.
    :attr field_path: dotted path of the field the step complained about, or an
        empty string if the failure cannot be attributed to a single field.
    """

    def __init__(
        self, step: MigrationStep, field_path: str, original: Exception
    ) -> None:
        self.step = step
        self.field_path = field_path
        self.original = original
        super().__init__(_render_step_error_message(step, field_path, original))

    def __str__(self) -> str:
        return self.args[0]


@frozen
class MigrationStep:
    """One explicit, one-way migration between two versions.

    :param source: the version this step consumes.
    :param target: the version this step produces.
    :param fn: the migration function. It receives an independent copy of the
        restricted JSON payload and must return the payload for `target`. It
        never receives or mutates the caller's input.
    """

    source: Version
    target: Version
    fn: MigrationFn


MigrationsInput = (
    Mapping[Version, MigrationStep | MigrationFn] | list[MigrationStep] | None
)


def _field_path_from_key_error(exc: KeyError) -> str:
    key = exc.args[0] if exc.args else ""
    return str(key)


@frozen(hash=False)
class VersionedRegistry:
    """The validated migration graph for one model on one converter.

    Instances are produced by the configuration functions; do not construct
    them directly. The graph is checked for duplicate edges, cycles and
    ambiguous paths when migrations are registered, before any payload is
    processed.
    """

    cl: type
    version_field: str
    target_version: Version
    steps: tuple[MigrationStep, ...]
    paths: dict[Version, tuple[MigrationStep, ...]]
    downgrades: dict[Version, Callable[[dict[str, Any]], dict[str, Any]]] = (
        Factory(dict)
    )

    def path_for(self, version: Version) -> tuple[MigrationStep, ...]:
        """Return the unique step sequence that upgrades `version`."""
        if version == self.target_version:
            return ()
        try:
            return self.paths[version]
        except KeyError:
            if self._is_future_version(version):
                raise UnknownVersionError(self.cl, version, self.target_version) from None
            raise MissingMigrationError(
                self.cl, version, self.target_version
            ) from None

    def _is_future_version(self, version: Version) -> bool:
        try:
            return version > self.target_version
        except TypeError:
            return False


def _validate_and_build_paths(
    cl: type,
    target_version: Version,
    steps: list[MigrationStep],
) -> dict[Version, tuple[MigrationStep, ...]]:
    """Validate the migration graph and resolve one unique path per version."""
    _target_has_outgoing_edge(cl, target_version, steps)
    adjacency: dict[Version, list[MigrationStep]] = {}
    for step in steps:
        bucket = adjacency.setdefault(step.source, [])
        if any(existing.target == step.target for existing in bucket):
            raise DuplicateMigrationError(cl, step.source)
        if bucket:
            # The same source version already has an edge to a different
            # target. Re-registering it cannot define a unique path.
            raise AmbiguousMigrationPathError(
                cl,
                step.source,
                [[existing] for existing in bucket] + [[step]],
            )
        bucket.append(step)

    # Iterative DFS cycle detection over the forward graph. The stack holds
    # (node, edges already visited); an edge back to a node on the stack is a
    # cycle.
    state: dict[Version, int] = {}  # 0 = unvisited, 1 = on stack, 2 = done
    for root in list(adjacency):
        if state.get(root, 0) != 0:
            continue
        stack: list[tuple[Version, int]] = [(root, 0)]
        path: list[Version] = [root]
        state[root] = 1
        while stack:
            node, index = stack[-1]
            edges = adjacency.get(node, ())
            if index < len(edges):
                stack[-1] = (node, index + 1)
                nxt = edges[index].target
                if state.get(nxt, 0) == 1:
                    cycle_start = path.index(nxt)
                    raise MigrationCycleError(cl, path[cycle_start:] + [nxt])
                if state.get(nxt, 0) == 0:
                    state[nxt] = 1
                    path.append(nxt)
                    stack.append((nxt, 0))
            else:
                state[node] = 2
                stack.pop()
                path.pop()

    # Resolve every path that reaches the target. Incomplete graphs from
    # incremental registration are allowed: those versions are simply absent
    # from `paths` and will be reported as missing at structure time.
    paths: dict[Version, tuple[MigrationStep, ...]] = {}
    for version in adjacency:
        path = _resolve_path(cl, version, adjacency, target_version)
        if path is not None:
            paths[version] = path

    return paths


def _resolve_path(
    cl: type,
    version: Version,
    adjacency: dict[Version, list[MigrationStep]],
    target_version: Version,
) -> tuple[MigrationStep, ...] | None:
    """Resolve the unique path from `version` to `target_version`.

    Return `None` when the chain is not complete yet, so partially configured
    graphs can be extended incrementally. Cycles and branches are rejected
    eagerly instead.
    """
    if version == target_version:
        return ()

    path: list[MigrationStep] = []
    current = version
    seen: set[Version] = set()
    while current != target_version:
        if current in seen:
            cycle = [*_path_versions(path), current]
            raise MigrationCycleError(cl, cycle)
        seen.add(current)
        edges = adjacency.get(current, ())
        if len(edges) > 1:
            # Branching is detected as ambiguous now; each branch is a valid
            # potential path, but the graph does not define which one wins.
            raise AmbiguousMigrationPathError(
                cl,
                version,
                [
                    _prefix_paths(path, edge, adjacency, target_version)
                    for edge in edges
                ],
            )
        if not edges:
            return None
        step = edges[0]
        path.append(step)
        current = step.target

    return tuple(path)


def _path_versions(path: list[MigrationStep]) -> list[Version]:
    """Return the versions visited by a path, starting at its source."""
    res = [path[0].source] if path else []
    res.extend(step.target for step in path)
    return res


def _prefix_paths(
    prefix: list[MigrationStep],
    edge: MigrationStep,
    adjacency: dict[Version, list[MigrationStep]],
    target_version: Version,
) -> list[MigrationStep]:
    """Best-effort collect one path after a branch, for error reporting."""
    path = [*prefix, edge]
    seen = {*_path_versions(prefix), edge.source}
    current = edge.target
    while current != target_version and current not in seen:
        seen.add(current)
        edges = adjacency.get(current)
        if not edges:
            break
        if len(edges) > 1:
            # Keep the first branch for the report; the error is about the
            # branching, not about enumerating every duplicate path.
            edge = edges[0]
            path.append(edge)
            current = edge.target
            continue
        edge = edges[0]
        path.append(edge)
        current = edge.target
    return path


def _target_has_outgoing_edge(
    cl: type, target_version: Version, steps: list[MigrationStep]
) -> None:
    for step in steps:
        if step.source == target_version:
            raise MigrationGraphError(
                f"{cl.__name__} target version {target_version!r} must be terminal, "
                f"but an edge to {step.target!r} was registered."
            )


def _normalize_migrations(
    migrations: MigrationsInput,
    target_version_to_use: Version,
) -> list[MigrationStep]:
    """Turn the accepted migration input shapes into a list of steps."""
    if migrations is None:
        return []
    if isinstance(migrations, Mapping):
        normalized: list[MigrationStep] = []
        for source, value in migrations.items():
            if isinstance(value, MigrationStep):
                if value.source != source:
                    raise ValueError(
                        f"Migration registered under {source!r} declares source "
                        f"{value.source!r}; these must match."
                    )
                normalized.append(value)
            else:
                normalized.append(MigrationStep(source, target_version_to_use, value))
        return normalized
    normalized = []
    for value in migrations:
        if not isinstance(value, MigrationStep):
            raise TypeError(
                "Migration lists must contain MigrationStep instances, got "
                f"{type(value).__name__}."
            )
        normalized.append(value)
    return normalized


_MAX_DEPTH = 100


def _clone_restricted_json(value: Any, path: str = "") -> Any:
    """Return a deep, independent copy of a restricted JSON value.

    Allowed values are ``None``, booleans, ints, floats, strings, lists and
    string-keyed dicts. Anything else is rejected before a migration function
    can observe it, so migration steps always operate on plain JSON data.
    """
    return _clone_restricted_json_at(value, path, 0)


def _clone_restricted_json_at(value: Any, path: str, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise ValueError(f"JSON value too deeply nested at {path or '<root>'}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [
            _clone_restricted_json_at(item, f"{path}[{index}]", depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        res = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"JSON object keys must be strings at {path or '<root>'}: "
                    f"{key!r}"
                )
            item_path = f"{path}.{key}" if path else key
            res[key] = _clone_restricted_json_at(item, item_path, depth + 1)
        return res
    raise ValueError(
        f"Non-JSON value at {path or '<root>'}: {type(value).__name__}"
    )


def _field_path_from_exception(exc: BaseException) -> str:
    """Extract a useful field path from exceptions raised by migration steps."""
    explicit = getattr(exc, "field_path", None)
    if explicit is not None:
        return str(explicit)
    if isinstance(exc, KeyError):
        return _field_path_from_key_error(exc)
    return ""


def _apply_migrations(
    registry: VersionedRegistry,
    payload: Mapping[str, Any],
    base_path: str = "",
) -> dict[str, Any]:
    """Copy the payload and run its unique migration path exactly once each."""
    if not isinstance(payload, Mapping):
        raise VersionedPayloadError(
            f"Versioned payloads for {registry.cl.__name__} must be mappings, "
            f"not {type(payload).__name__}."
        )

    original_version = payload.get(registry.version_field, _MISSING)
    if original_version is _MISSING:
        raise VersionedPayloadError(
            f"Missing version field {registry.version_field!r} for "
            f"{registry.cl.__name__}."
        )

    steps = registry.path_for(original_version)
    # Clone before reading anything in place; the caller's payload is never
    # modified by the strategy or by migration functions.
    try:
        working = _clone_restricted_json(dict(payload), base_path)
    except ValueError as exc:
        raise VersionedPayloadError(str(exc)) from exc

    for step in steps:
        # Every step gets its own independent copy. A step may mutate its
        # argument freely; neither subsequent steps nor the caller see it.
        step_input = _clone_restricted_json(working, base_path)
        try:
            produced = step.fn(step_input)
        except Exception as exc:
            raise MigrationStepError(
                step, _field_path_from_exception(exc), exc
            ) from exc

        if not isinstance(produced, Mapping):
            raise MigrationStepError(
                step,
                base_path,
                TypeError(
                    "migration functions must return a mapping, "
                    f"got {type(produced).__name__}"
                ),
            )

        # Validate and copy the output too, keeping each version boundary
        # well-defined and preventing shared mutable state between steps.
        try:
            working = _clone_restricted_json(dict(produced), base_path)
        except ValueError as exc:
            raise MigrationStepError(step, "", exc) from exc
        working[registry.version_field] = step.target

    return working


def _render_step_error_message(
    step: MigrationStep, field_path: str, original: Exception
) -> str:
    path_suffix = f" at field {field_path!r}" if field_path else ""
    return (
        f"Migration {step.source!r} -> {step.target!r} failed{path_suffix}: "
        f"{type(original).__name__}: {original}"
    )


def _prefix_nested_migration_paths(exc: ExceptionGroup) -> None:
    """Prepend field names from generated structure notes to nested errors."""
    for subexc in exc.exceptions:
        if isinstance(subexc, MigrationStepError):
            notes = getattr(subexc, "__notes__", ())
            field_names = [
                note.name
                for note in notes
                if isinstance(note, AttributeValidationNote)
            ]
            if field_names:
                prefix = ".".join(field_names)
                subexc.field_path = (
                    f"{prefix}.{subexc.field_path}"
                    if subexc.field_path
                    else prefix
                )
                subexc.args = (
                    _render_step_error_message(
                        subexc.step, subexc.field_path, subexc.original
                    ),
                )
        elif isinstance(subexc, ExceptionGroup):
            _prefix_nested_migration_paths(subexc)


def _has_version_field(cl: type, version_field: str) -> bool:
    return any(field.name == version_field for field in fields(cl))


def configure_versioned_structure(
    cl: type[T],
    converter: BaseConverter,
    *,
    migrations: MigrationsInput = None,
    target_version: Version,
    version_field: str = "version",
) -> VersionedRegistry:
    """Register explicit, graph-checked migrations for structuring `cl`.

    The strategy keeps every version adaptation outside of the model
    constructor: payloads are migrated to `target_version` first, and only
    then go through the converter's normal structure hook for `cl`. This means
    ordinary validation and extra-key checks still apply to the migrated
    payload.

    :param cl: the attrs class or dataclass whose payloads are versioned.
    :param converter: the converter to configure.
    :param migrations: the migrations to add. Either a mapping of source
        versions to migration functions (each wrapped into a step from its key
        to `target_version`), or a list of :class:`MigrationStep` instances
        for full control over intermediate versions. May be omitted to
        (re)declare the target version only. Calling this function again for
        the same class appends steps to the existing graph, so migrations can
        be declared incrementally; the whole graph is re-validated each time.
    :param target_version: the single current version the model represents.
        Payloads already at this version skip migration entirely.
    :param version_field: the payload key holding the version. If `cl` has no
        field with this name, the key is removed after migration so it does
        not trigger extra-key validation.

    The graph is validated eagerly:
      * registering the same source version twice raises
        :class:`DuplicateMigrationError`,
      * cycles raise :class:`MigrationCycleError`,
      * branching that allows more than one route to the target raises
        :class:`AmbiguousMigrationPathError`,
      * an edge leaving the target version raises
        :class:`MigrationGraphError`.

    At structure time, versions above `target_version` (when comparable)
    raise :class:`UnknownVersionError`; versions with no usable path raise
    :class:`MissingMigrationError`; a failing step raises
    :class:`MigrationStepError` carrying the step and the field path.

    Nested versioned models carry their own registries and are migrated
    independently by their own structure hooks. The outer model's version is
    never propagated into nested payloads. Configure nested models before
    their containers: the outer model's structure hook freezes the nested
    hooks that exist at registration time, like the rest of the converter.

    .. versionadded:: NEXT
    """
    if not has_class(cl):
        raise TypeError(
            f"{cl!r} must be an attrs class or dataclass to be versioned."
        )

    new_steps = _normalize_migrations(migrations, target_version)

    existing = _find_registry(converter, cl)
    if existing is not None:
        if (
            existing.version_field != version_field
            or existing.target_version != target_version
        ):
            raise ValueError(
                f"{cl.__name__} is already configured with version field "
                f"{existing.version_field!r} and target version "
                f"{existing.target_version!r}; reconfiguring with different "
                "values is not supported."
            )
        all_steps = [*existing.steps, *new_steps]
    else:
        all_steps = new_steps

    paths = _validate_and_build_paths(cl, target_version, all_steps)
    registry = VersionedRegistry(
        cl=cl,
        version_field=version_field,
        target_version=target_version,
        steps=tuple(all_steps),
        paths=paths,
    )

    # Capture the ordinary structure hook before replacing it with the
    # versioned wrapper. An earlier versioned wrapper marks itself with the
    # base hook, so incremental reconfiguration (and converter copies, which
    # carry the wrapper along) reuse the original hook instead of wrapping
    # the wrapper itself.
    current_hook = converter.get_structure_hook(cl)
    base_structure_hook = getattr(
        current_hook, "_cattrs_versioned_base", current_hook
    )
    keep_version_key = _has_version_field(cl, version_field)

    def structure_versioned(
        payload: Mapping[str, Any],
        _cl: type[T] = cl,
        _registry: VersionedRegistry = registry,
        _base_hook=base_structure_hook,
        _keep_version_key: bool = keep_version_key,
    ) -> T:
        migrated = _apply_migrations(_registry, payload)
        if not _keep_version_key:
            migrated.pop(_registry.version_field, None)
        try:
            return _base_hook(migrated, _cl)
        except ExceptionGroup as exc:
            _prefix_nested_migration_paths(exc)
            raise

    # Expose the underlying normal hook and the registry on the wrapper so a
    # copied converter can recover them instead of treating the versioned
    # wrapper as the new base hook.
    structure_versioned._cattrs_versioned_base = base_structure_hook  # type: ignore[attr-defined]
    structure_versioned._cattrs_versioned_registry = registry  # type: ignore[attr-defined]

    converter.register_structure_hook(cl, structure_versioned)
    _registries.setdefault(converter, {})[cl] = registry
    return registry


# Per-converter registries. Converters are slotted and cannot hold arbitrary
# attributes, so the strategy keeps its own table keyed by converter identity.
_registries: dict[BaseConverter, dict[type, VersionedRegistry]] = {}


def configure_versioned_unstructure(
    cl: type[T],
    converter: BaseConverter,
    *,
    target_version: Version,
    version_field: str = "version",
    include_version: bool = True,
) -> None:
    """Make normal unstructuring emit the current version of `cl`.

    This covers the default case: calling `converter.unstructure(instance)`
    produces the target version, including `version_field` when requested.

    Exporting an explicit older version is a separate capability, provided by
    :func:`export_version`: it requires explicit downgrade functions because
    forward migrations cannot be reversed automatically.

    :param include_version: when false, the version field is not added; this
        is useful for models that keep the version in a separate envelope.

    .. versionadded:: NEXT
    """
    if not has_class(cl):
        raise TypeError(
            f"{cl!r} must be an attrs class or dataclass to be versioned."
        )

    current_hook = converter.get_unstructure_hook(cl)
    base_unstructure_hook = getattr(
        current_hook, "_cattrs_versioned_unstructure_base", current_hook
    )

    def unstructure_versioned(
        instance: T,
        _registry_cl: type[T] = cl,
        _base_hook=base_unstructure_hook,
        _version: Version = target_version,
        _version_field: str = version_field,
        _include: bool = include_version,
    ) -> dict[str, Any]:
        res = _base_hook(instance)
        if not isinstance(res, Mapping):
            raise TypeError(
                "Versioned unstructuring requires a dict-producing "
                f"unstructure strategy for {cl.__name__}, got "
                f"{type(res).__name__}."
            )
        if _include:
            # Stamp the target version explicitly, overriding any stale
            # version stored on the model itself.
            res = {**res, _version_field: _version}
        return res

    unstructure_versioned._cattrs_versioned_unstructure_base = (  # type: ignore[attr-defined]
        base_unstructure_hook
    )

    converter.register_unstructure_hook(cl, unstructure_versioned)


def _find_registry(
    converter: BaseConverter, cl: type
) -> VersionedRegistry | None:
    """Locate the registry, including on a converter copied after setup."""
    registry = _registries.get(converter, {}).get(cl)
    if registry is not None:
        return registry
    hook = converter.get_structure_hook(cl)
    return getattr(hook, "_cattrs_versioned_registry", None)


def register_downgrade(
    cl: type[T],
    converter: BaseConverter,
    version: Version,
    fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Register an explicit downgrade function for legacy export of `cl`.

    Downgrades never run during normal structuring and are never derived from
    forward migrations. Each downgrade receives the current, target-version
    payload (an independent copy) and returns a payload matching `version`.

    .. versionadded:: NEXT
    """
    registry = _find_registry(converter, cl)
    if registry is None:
        raise ValueError(
            f"{cl.__name__} is not configured for versioned handling on this "
            "converter; call configure_versioned_structure first."
        )
    registry.downgrades[version] = fn


def export_version(
    instance: T,
    converter: BaseConverter,
    *,
    cl: type[T],
    version: Version,
) -> dict[str, Any]:
    """Unstructure `instance` as an explicit, possibly older, version.

    The instance is first unstructured at its current target version using the
    converter's normal unstructure hook. If `version` is the target version,
    that payload is returned directly. Otherwise the explicit downgrade
    function registered with :func:`register_downgrade` is applied to an
    independent copy; missing downgrade functions raise `ValueError` instead
    of guessing a reverse migration.

    .. versionadded:: NEXT
    """
    registry = _find_registry(converter, cl)
    if registry is None:
        raise ValueError(
            f"{cl.__name__} is not configured for versioned handling on this "
            "converter; call configure_versioned_structure first."
        )

    current = converter.unstructure(instance, unstructure_as=cl)
    if version == registry.target_version:
        return current

    fn = registry.downgrades.get(version)
    if fn is None:
        raise ValueError(
            f"No explicit downgrade registered for {cl.__name__} to version "
            f"{version!r}. Forward migrations cannot be reversed automatically."
        )

    payload = _clone_restricted_json(
        current, f"downgrade:{cl.__name__}"
    )
    produced = fn(payload)
    if not isinstance(produced, Mapping):
        raise TypeError(
            f"Downgrade for {cl.__name__} version {version!r} must return a "
            f"mapping, got {type(produced).__name__}."
        )
    result = _clone_restricted_json(dict(produced))
    result[registry.version_field] = version
    return result
