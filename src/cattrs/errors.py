from collections.abc import Sequence
from typing import Any, Optional, Union

from typing_extensions import Self

from cattrs._compat import ExceptionGroup


class CattrsError(Exception):
    """Base ``cattrs`` exception."""


class StructureHandlerNotFoundError(CattrsError):
    """
    Error raised when structuring cannot find a handler for converting inputs into
    :attr:`type_`.
    """

    def __init__(self, message: str, type_: type) -> None:
        super().__init__(message, type_)
        self.message = message
        self.type_ = type_

    def __str__(self) -> str:
        return self.message


class BaseValidationError(ExceptionGroup, CattrsError):
    cl: type

    def __new__(cls, message: str, excs: Sequence[Exception], cl: type) -> Self:
        obj = super().__new__(cls, message, excs)
        obj.cl = cl
        return obj

    def derive(self, excs: Sequence[Exception]) -> Self:
        return self.__class__(self.message, excs, self.cl)


class IterableValidationNote(str):
    """Attached as a note to an exception when an iterable element fails structuring."""

    index: Union[int, str]  # Ints for list indices, strs for dict keys
    type: Any

    def __new__(cls, string: str, index: Union[int, str], type: Any) -> Self:
        instance = str.__new__(cls, string)
        instance.index = index
        instance.type = type
        return instance

    def __getnewargs__(self) -> tuple[str, Union[int, str], Any]:
        return (str(self), self.index, self.type)


class IterableValidationError(BaseValidationError):
    """Raised when structuring an iterable."""

    def group_exceptions(
        self,
    ) -> tuple[list[tuple[Exception, IterableValidationNote]], list[Exception]]:
        """Split the exceptions into two groups: with and without validation notes."""
        excs_with_notes = []
        other_excs = []
        for subexc in self.exceptions:
            if hasattr(subexc, "__notes__"):
                for note in subexc.__notes__:
                    if note.__class__ is IterableValidationNote:
                        excs_with_notes.append((subexc, note))
                        break
                else:
                    other_excs.append(subexc)
            else:
                other_excs.append(subexc)

        return excs_with_notes, other_excs


class AttributeValidationNote(str):
    """Attached as a note to an exception when an attribute fails structuring."""

    name: str
    type: Any

    def __new__(cls, string: str, name: str, type: Any) -> Self:
        instance = str.__new__(cls, string)
        instance.name = name
        instance.type = type
        return instance

    def __getnewargs__(self) -> tuple[str, str, Any]:
        return (str(self), self.name, self.type)


class ClassValidationError(BaseValidationError):
    """Raised when validating a class if any attributes are invalid."""

    def group_exceptions(
        self,
    ) -> tuple[list[tuple[Exception, AttributeValidationNote]], list[Exception]]:
        """Split the exceptions into two groups: with and without validation notes."""
        excs_with_notes = []
        other_excs = []
        for subexc in self.exceptions:
            if hasattr(subexc, "__notes__"):
                for note in subexc.__notes__:
                    if note.__class__ is AttributeValidationNote:
                        excs_with_notes.append((subexc, note))
                        break
                else:
                    other_excs.append(subexc)
            else:
                other_excs.append(subexc)

        return excs_with_notes, other_excs


class ForbiddenExtraKeysError(CattrsError):
    """
    Raised when `forbid_extra_keys` is activated and such extra keys are detected
    during structuring.

    The attribute `extra_fields` is a sequence of those extra keys, which were the
    cause of this error, and `cl` is the class which was structured with those extra
    keys.
    """

    def __init__(
        self, message: Optional[str], cl: type, extra_fields: set[str]
    ) -> None:
        self.message = message
        self.cl = cl
        self.extra_fields = extra_fields

        super().__init__(message, cl, extra_fields)

    def __str__(self) -> str:
        return (
            self.message
            or f"Extra fields in constructor for {self.cl.__name__}: "
            f"{', '.join(sorted(self.extra_fields))}"
        )


class GraphConversionError(CattrsError):
    """Base error for un/structuring object graphs via the graph strategy.

    The `path` attribute locates the offending position within the graph
    document, for example ``$.items[2].child``.
    """

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path

    def __str__(self) -> str:
        if self.path:
            return f"{self.message} (at {self.path})"
        return self.message


class UnknownGraphTypeError(GraphConversionError):
    """A ``$type`` name in the payload is not one of the declared, allowed types.

    Type names are never imported or resolved dynamically; they must be
    reachable from the target type or registered as allowed types.
    """

    def __init__(self, type_name: str, path: str | None = None) -> None:
        super().__init__(f"Unknown graph type {type_name!r}", path)
        self.type_name = type_name


class DuplicateReferenceError(GraphConversionError):
    """Two definitions in the payload use the same reference ``$id``."""

    def __init__(self, ref_id: int, path: str | None = None) -> None:
        super().__init__(f"Duplicate definition of reference id {ref_id}", path)
        self.ref_id = ref_id


class DanglingReferenceError(GraphConversionError):
    """A ``$ref`` points to an id with no preceding definition in this document."""

    def __init__(self, ref_id: int, path: str | None = None) -> None:
        super().__init__(f"Dangling reference to undefined id {ref_id}", path)
        self.ref_id = ref_id


class GraphTypeMismatchError(GraphConversionError):
    """A graph node or reference does not match the declared type at its position."""


class GraphCycleError(GraphConversionError):
    """A reference cycle cannot be reconstructed without bypassing initialization.

    Raised when a cycle passes through a frozen class or a class requiring a
    custom constructor, neither of which can be safely rebuilt in two phases.
    """
