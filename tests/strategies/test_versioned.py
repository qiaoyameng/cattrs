"""Tests for the versioned structure strategy with explicit migrations."""

from copy import deepcopy
from dataclasses import dataclass

import pytest
from attrs import define

from cattrs import BaseConverter, Converter
from cattrs.errors import ForbiddenExtraKeysError
from cattrs.strategies import (
    AmbiguousMigrationPathError,
    DuplicateMigrationError,
    MigrationCycleError,
    MigrationGraphError,
    MigrationStep,
    MigrationStepError,
    MissingMigrationError,
    UnknownVersionError,
    VersionedPayloadError,
    configure_versioned_structure,
    configure_versioned_unstructure,
    export_version,
    register_downgrade,
)


@define
class Article:
    title: str
    tags: list[str]


@define
class Nested:
    label: str


@define
class Container:
    title: str
    article: Nested


@pytest.fixture(params=(BaseConverter, Converter))
def converter(request):
    return request.param()


def make_three_step_converter(converter_cl):
    """A v1 -> v2 -> v3 setup with call counters for every step."""
    converter = converter_cl()
    calls = {"v1_to_v2": 0, "v2_to_v3": 0}

    def v1_to_v2(payload):
        calls["v1_to_v2"] += 1
        payload["tags"] = [payload.pop("name")]
        return payload

    def v2_to_v3(payload):
        calls["v2_to_v3"] += 1
        payload["title"] = payload["title"].upper()
        return payload

    registry = configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations=[
            MigrationStep(1, 2, v1_to_v2),
            MigrationStep(2, 3, v2_to_v3),
        ],
    )
    return converter, registry, calls


def test_two_step_migration_runs_each_step_once(converter):
    """v1 input crosses two explicit steps and reaches the target."""
    converter, registry, calls = make_three_step_converter(converter.__class__)

    original = {"version": 1, "name": "cat", "title": "hello"}
    result = converter.structure(original, Article)

    assert result == Article("HELLO", ["cat"])
    assert calls == {"v1_to_v2": 1, "v2_to_v3": 1}
    assert registry.paths[1] == tuple(registry.steps)


def test_successful_steps_are_not_repeated(converter):
    """Every successful step runs exactly once per payload; counts prove it."""
    converter, _, calls = make_three_step_converter(converter.__class__)

    converter.structure({"version": 1, "name": "a", "title": "one"}, Article)
    assert calls == {"v1_to_v2": 1, "v2_to_v3": 1}

    converter.structure({"version": 2, "title": "two", "tags": ["b"]}, Article)
    assert calls == {"v1_to_v2": 1, "v2_to_v3": 2}

    converter.structure({"version": 3, "title": "three", "tags": []}, Article)
    assert calls == {"v1_to_v2": 1, "v2_to_v3": 2}


def test_migration_does_not_mutate_input(converter):
    """The caller's payload must be unchanged after structuring."""
    converter, _, _ = make_three_step_converter(converter.__class__)

    payload = {"version": 1, "name": "cat", "title": "hello", "nested": [1, 2]}
    snapshot = deepcopy(payload)

    converter.structure(payload, Article)

    assert payload == snapshot


def test_each_step_receives_an_independent_copy(converter):
    """Mutations performed by one step cannot leak into the next step."""
    converter = converter.__class__()
    seen = []

    def first(payload):
        payload["from_first"] = True
        payload["tags"] = ["created"]
        return payload

    def second(payload):
        seen.append(deepcopy(payload))
        payload["tags"].append("second")
        return payload

    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations=[MigrationStep(1, 2, first), MigrationStep(2, 3, second)],
    )

    result = converter.structure({"version": 1, "title": "t", "tags": []}, Article)
    assert result == Article("t", ["created", "second"])
    assert seen == [{"version": 2, "title": "t", "tags": ["created"], "from_first": True}]


def test_missing_step_is_rejected_at_structure_time(converter):
    """A version with no path to the target is a missing migration."""
    converter = converter.__class__()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations={1: MigrationStep(1, 3, lambda payload: payload)},
    )

    with pytest.raises(MissingMigrationError) as exc_info:
        converter.structure({"version": 2, "title": "x", "tags": []}, Article)

    assert exc_info.value.version == 2
    assert exc_info.value.target == 3
    assert exc_info.value.cl is Article


def test_incomplete_chain_reports_the_missing_link(converter):
    """Partial graphs configure fine but fail when the gap is hit."""
    converter = converter.__class__()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations=[MigrationStep(1, 2, lambda payload: payload)],
    )

    with pytest.raises(MissingMigrationError):
        converter.structure({"version": 1, "title": "x", "tags": []}, Article)

    # Adding the missing step completes the chain incrementally.
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations=[MigrationStep(2, 3, lambda payload: payload)],
    )
    assert converter.structure(
        {"version": 1, "title": "x", "tags": ["a"]}, Article
    ) == Article("x", ["a"])


def test_cycle_is_rejected_at_registration(converter):
    with pytest.raises(MigrationCycleError) as exc_info:
        configure_versioned_structure(
            Article,
            converter.__class__(),
            target_version=3,
            migrations=[
                MigrationStep(1, 2, lambda payload: payload),
                MigrationStep(2, 1, lambda payload: payload),
            ],
        )
    assert exc_info.value.cycle == [1, 2, 1]


def test_duplicate_migration_is_rejected(converter):
    converter = converter.__class__()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations=[MigrationStep(1, 2, lambda payload: payload)],
    )

    with pytest.raises(DuplicateMigrationError) as exc_info:
        configure_versioned_structure(
            Article,
            converter,
            target_version=3,
            migrations=[MigrationStep(1, 2, lambda payload: payload)],
        )
    assert exc_info.value.version == 1


def test_ambiguous_paths_are_rejected(converter):
    with pytest.raises(AmbiguousMigrationPathError) as exc_info:
        configure_versioned_structure(
            Article,
            converter.__class__(),
            target_version=3,
            migrations=[
                MigrationStep(1, 2, lambda payload: payload),
                MigrationStep(1, 3, lambda payload: payload),
            ],
        )
    assert len(exc_info.value.paths) == 2


def test_target_version_must_be_terminal(converter):
    with pytest.raises(MigrationGraphError):
        configure_versioned_structure(
            Article,
            converter.__class__(),
            target_version=2,
            migrations=[MigrationStep(2, 3, lambda payload: payload)],
        )


def test_unknown_future_version_is_rejected(converter):
    """Comparable versions above the target are explicitly refused."""
    converter, _, _ = make_three_step_converter(converter.__class__)

    with pytest.raises(UnknownVersionError) as exc_info:
        converter.structure({"version": 99, "title": "x", "tags": []}, Article)

    assert exc_info.value.version == 99
    assert exc_info.value.target == 3


def test_non_comparable_unknown_version_is_missing_migration(converter):
    """Versions that cannot be ordered still fail, just not as 'future'."""
    converter, _, _ = make_three_step_converter(converter.__class__)

    with pytest.raises(MissingMigrationError):
        converter.structure({"version": "v9", "title": "x", "tags": []}, Article)


def test_step_failure_carries_step_and_field_path(converter):
    converter = converter.__class__()

    def failing_step(payload):
        return {"title": payload["settings"]["timeout"], "tags": []}

    configure_versioned_structure(
        Article,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, failing_step)},
    )

    with pytest.raises(MigrationStepError) as exc_info:
        converter.structure({"version": 1, "settings": {}}, Article)

    error = exc_info.value
    assert error.step == MigrationStep(1, 2, failing_step)
    assert error.field_path == "timeout"
    assert isinstance(error.original, KeyError)
    assert "1 -> 2" in str(error)


def test_explicit_field_path_is_honored(converter):
    class SettingsError(Exception):
        def __init__(self):
            super().__init__("bad timeout")
            self.field_path = "settings.timeout"

    def failing_step(payload):
        raise SettingsError()

    converter = converter.__class__()
    configure_versioned_structure(
        Article,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, failing_step)},
    )

    with pytest.raises(MigrationStepError) as exc_info:
        converter.structure({"version": 1}, Article)

    assert exc_info.value.field_path == "settings.timeout"


def test_non_json_input_is_rejected(converter):
    converter, _, _ = make_three_step_converter(converter.__class__)

    with pytest.raises(VersionedPayloadError):
        converter.structure(
            {"version": 3, "title": object(), "tags": []}, Article
        )

    with pytest.raises(VersionedPayloadError):
        converter.structure(["not", "a", "mapping"], Article)

    with pytest.raises(VersionedPayloadError):
        converter.structure({"title": "x", "tags": []}, Article)


def test_extra_key_validation_still_runs_after_migration():
    converter = Converter(forbid_extra_keys=True)

    def migrate(payload):
        payload["title"] = payload.pop("headline")
        payload["tags"] = payload.pop("labels")
        return payload

    configure_versioned_structure(
        Article,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, migrate)},
    )

    assert converter.structure(
        {"version": 2, "title": "t", "tags": []}, Article
    ) == Article("t", [])

    with pytest.raises(Exception) as exc_info:
        converter.structure(
            {
                "version": 1,
                "headline": "t",
                "labels": ["x"],
                "obsolete_key": 1,
            },
            Article,
        )
    assert any(
        isinstance(sub, ForbiddenExtraKeysError)
        for sub in exc_info.value.exceptions
    )


def test_version_field_not_on_model_is_removed():
    @define
    class NoVersionField:
        title: str

    converter = Converter(forbid_extra_keys=True)
    configure_versioned_structure(
        NoVersionField,
        converter,
        target_version=2,
        migrations={
            1: MigrationStep(
                1, 2, lambda payload: {"title": payload["title"]}
            )
        },
    )
    assert converter.structure({"version": 2, "title": "t"}, NoVersionField)


def test_nested_models_migrate_independently():
    """A nested model's own version is never overwritten by the outer one."""
    converter = Converter(forbid_extra_keys=True)

    def inner_old(payload):
        payload["label"] = payload.pop("name")
        return payload

    configure_versioned_structure(
        Nested,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, inner_old)},
    )

    def outer_old(payload):
        payload["title"] = payload.pop("headline")
        return payload

    configure_versioned_structure(
        Container,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, outer_old)},
    )

    result = converter.structure(
        {
            "version": 1,
            "headline": "hello",
            "article": {"version": 1, "name": "legacy"},
        },
        Container,
    )
    assert result == Container("hello", Nested("legacy"))


def test_nested_model_keeps_its_own_version_field():
    """Both levels use 'version', with different targets; neither leaks."""
    converter = Converter(forbid_extra_keys=True)
    calls = {"inner": 0, "outer": 0}

    def inner_step(payload):
        calls["inner"] += 1
        payload["label"] = payload.pop("name")
        return payload

    def outer_step(payload):
        calls["outer"] += 1
        payload["title"] = payload.pop("headline")
        return payload

    configure_versioned_structure(
        Nested,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, inner_step)},
    )
    configure_versioned_structure(
        Container,
        converter,
        target_version=3,
        migrations=[
            MigrationStep(1, 2, outer_step),
            MigrationStep(2, 3, lambda payload: payload),
        ],
    )

    # The outer payload is v1, the nested one is also v1 but has a shorter
    # chain. The outer v2 -> v3 step must not force the nested one anywhere.
    result = converter.structure(
        {
            "version": 1,
            "headline": "hello",
            "article": {"version": 1, "name": "legacy"},
        },
        Container,
    )
    assert result == Container("hello", Nested("legacy"))
    assert calls == {"inner": 1, "outer": 1}

    # Inner at its target, outer still old: only the outer step runs.
    calls["inner"] = calls["outer"] = 0
    result = converter.structure(
        {
            "version": 1,
            "headline": "world",
            "article": {"version": 2, "label": "current"},
        },
        Container,
    )
    assert result == Container("world", Nested("current"))
    assert calls == {"inner": 0, "outer": 1}


def test_nested_unknown_future_version_is_reported_for_the_nested_model():
    converter = Converter()
    configure_versioned_structure(
        Nested,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, lambda payload: payload)},
    )
    configure_versioned_structure(
        Container,
        converter,
        target_version=3,
        migrations={1: MigrationStep(1, 3, lambda payload: payload)},
    )

    with pytest.raises(Exception) as exc_info:
        converter.structure(
            {
                "version": 3,
                "title": "outer",
                "article": {"version": 9, "label": "x"},
            },
            Container,
        )

    error = exc_info.value
    nested_errors = [
        exc for exc in error.exceptions if isinstance(exc, UnknownVersionError)
    ]
    assert nested_errors
    assert nested_errors[0].cl is Nested
    assert nested_errors[0].target == 2


def test_nested_step_error_carries_full_field_path():
    """The outer field name is prefixed onto a nested migration error path."""
    converter = Converter()
    configure_versioned_structure(
        Nested,
        converter,
        target_version=2,
        migrations={
            1: MigrationStep(1, 2, lambda payload: {"label": payload["missing"]})
        },
    )
    configure_versioned_structure(
        Container,
        converter,
        target_version=2,
        migrations={1: MigrationStep(1, 2, lambda payload: payload)},
    )

    with pytest.raises(Exception) as exc_info:
        converter.structure(
            {
                "version": 1,
                "title": "outer",
                "article": {"version": 1, "x": 1},
            },
            Container,
        )

    step_errors = [
        exc
        for exc in exc_info.value.exceptions
        if isinstance(exc, MigrationStepError)
    ]
    assert step_errors
    assert step_errors[0].field_path == "article.missing"


def test_dataclass_is_supported():
    @dataclass
    class DataclassArticle:
        title: str

    converter = Converter()
    configure_versioned_structure(
        DataclassArticle,
        converter,
        target_version=2,
        migrations={
            1: MigrationStep(
                1, 2, lambda payload: {"title": payload.pop("headline")}
            )
        },
    )
    assert converter.structure(
        {"version": 1, "headline": "hello"}, DataclassArticle
    ) == DataclassArticle("hello")


def test_default_unstructure_emits_target_version():
    converter = Converter()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations={1: MigrationStep(1, 3, lambda payload: payload)},
    )
    configure_versioned_unstructure(Article, converter, target_version=3)

    assert converter.unstructure(Article("t", ["a"])) == {
        "title": "t",
        "tags": ["a"],
        "version": 3,
    }


def test_default_unstructure_can_omit_version_field():
    converter = Converter()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations={1: MigrationStep(1, 3, lambda payload: payload)},
    )
    configure_versioned_unstructure(
        Article, converter, target_version=3, include_version=False
    )

    assert converter.unstructure(Article("t", ["a"])) == {
        "title": "t",
        "tags": ["a"],
    }


def test_export_legacy_version_requires_explicit_downgrade():
    converter = Converter()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations={1: MigrationStep(1, 3, lambda payload: payload)},
    )
    configure_versioned_unstructure(Article, converter, target_version=3)

    # Exporting the current version does not need a downgrade.
    current = export_version(Article("t", ["a"]), converter, cl=Article, version=3)
    assert current == {"title": "t", "tags": ["a"], "version": 3}

    # An older version cannot be guessed by reversing forward migrations.
    with pytest.raises(ValueError, match="cannot be reversed"):
        export_version(Article("t", ["a"]), converter, cl=Article, version=1)

    def to_v1(payload):
        return {
            "name": payload["tags"][0] if payload["tags"] else payload["title"],
            "title": payload["title"],
        }

    register_downgrade(Article, converter, 1, to_v1)

    legacy = export_version(Article("t", ["a"]), converter, cl=Article, version=1)
    assert legacy == {"name": "a", "title": "t", "version": 1}


def test_export_does_not_mutate_the_instance_or_run_forward_migrations():
    converter = Converter()
    calls = {"forward": 0}

    def forward(payload):
        calls["forward"] += 1
        return payload

    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations={1: MigrationStep(1, 3, forward)},
    )
    configure_versioned_unstructure(Article, converter, target_version=3)
    register_downgrade(
        Article,
        converter,
        1,
        lambda payload: {**payload, "name": payload["title"]},
    )

    instance = Article("t", ["a"])
    snapshot_title = instance.title
    export_version(instance, converter, cl=Article, version=1)
    assert instance.title == snapshot_title
    # Downgrading never touches the forward migration graph.
    assert calls == {"forward": 0}


def test_reconfiguring_with_different_target_is_rejected(converter):
    converter = converter.__class__()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations=[MigrationStep(1, 3, lambda payload: payload)],
    )

    with pytest.raises(ValueError, match="already configured"):
        configure_versioned_structure(Article, converter, target_version=4)


def test_invalid_class_is_rejected(converter):
    class NotModel:
        pass

    with pytest.raises(TypeError):
        configure_versioned_structure(
            NotModel,
            converter.__class__(),
            target_version=1,
        )


def test_copied_converter_keeps_versioning_and_can_grow_independently():
    converter = Converter()
    configure_versioned_structure(
        Article,
        converter,
        target_version=3,
        migrations=[
            MigrationStep(1, 2, lambda payload: payload),
            MigrationStep(2, 3, lambda payload: payload),
        ],
    )
    configure_versioned_unstructure(Article, converter, target_version=3)
    register_downgrade(
        Article, converter, 1, lambda payload: {**payload, "legacy": True}
    )

    copied = converter.copy()
    assert copied.structure({"version": 1, "title": "t", "tags": []}, Article)
    assert copied.unstructure(Article("t", []))["version"] == 3
    assert export_version(Article("t", []), copied, cl=Article, version=1)[
        "legacy"
    ] is True

    # Adding a step to the copy must not mutate the original converter.
    configure_versioned_structure(
        Article,
        copied,
        target_version=3,
        migrations=[MigrationStep(0, 1, lambda payload: payload)],
    )
    assert copied.structure({"version": 0, "title": "t", "tags": []}, Article)
    with pytest.raises(MissingMigrationError):
        converter.structure({"version": 0, "title": "t", "tags": []}, Article)
