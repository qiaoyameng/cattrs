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


class ColumnValidationNote(str):
    """Attached to an exception when a column cell fails structuring."""

    row_index: int
    field: str
    column: str
    type: Any

    def __new__(
        cls,
        string: str,
        row_index: int,
        field: str,
        column: str,
        type: Any,
    ) -> Self:
        instance = str.__new__(cls, string)
        instance.row_index = row_index
        instance.field = field
        instance.column = column
        instance.type = type
        return instance

    def __getnewargs__(self) -> tuple[str, int, str, str, Any]:
        return (str(self), self.row_index, self.field, self.column, self.type)


class ColumnRowValidationNote(str):
    """Attached to an exception raised while constructing a columnar row."""

    row_index: int

    def __new__(cls, string: str, row_index: int) -> Self:
        instance = str.__new__(cls, string)
        instance.row_index = row_index
        return instance

    def __getnewargs__(self) -> tuple[str, int]:
        return str(self), self.row_index


class ColumnarValidationError(BaseValidationError):
    """Raised when one or more columnar rows cannot be structured."""

    def group_exceptions(
        self,
    ) -> tuple[
        list[tuple[Exception, ColumnValidationNote | ColumnRowValidationNote]],
        list[Exception],
    ]:
        """Split the exceptions into cell errors and other column errors."""
        excs_with_notes = []
        other_excs = []
        for subexc in self.exceptions:
            if hasattr(subexc, "__notes__"):
                for note in subexc.__notes__:
                    if note.__class__ in (
                        ColumnValidationNote,
                        ColumnRowValidationNote,
                    ):
                        excs_with_notes.append((subexc, note))
                        break
                else:
                    other_excs.append(subexc)
            else:
                other_excs.append(subexc)

        return excs_with_notes, other_excs


class MissingColumnError(CattrsError):
    """Raised when a required column is absent from columnar input."""

    def __init__(
        self, cl: type, field_name: str, column_name: str | None = None
    ) -> None:
        self.cl = cl
        self.field_name = field_name
        self.column_name = column_name or field_name
        super().__init__(cl, field_name, self.column_name)

    def __str__(self) -> str:
        return (
            f"Missing required column {self.column_name!r} "
            f"for field {self.field_name!r}"
        )


class ColumnLengthError(CattrsError):
    """Raised when columns do not contain the same number of rows."""

    def __init__(
        self,
        column_name: str,
        expected_length: int,
        actual_length: int,
    ) -> None:
        self.column_name = column_name
        self.expected_length = expected_length
        self.actual_length = actual_length
        super().__init__(column_name, expected_length, actual_length)

    def __str__(self) -> str:
        return (
            f"Column {self.column_name!r} has length {self.actual_length}, "
            f"expected {self.expected_length}"
        )


class InvalidColumnError(CattrsError):
    """Raised when a column cannot be indexed by row."""

    def __init__(self, column_name: str) -> None:
        self.column_name = column_name
        super().__init__(column_name)

    def __str__(self) -> str:
        return f"Column {self.column_name!r} must be a sized, indexable sequence"


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
