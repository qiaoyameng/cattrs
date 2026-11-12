"""Per-conversion resource budgets.

A budget caps the work a single ``structure`` or ``unstructure`` call is
allowed to do. The same budget instance is shared by every hook participating
in one conversion, including hooks generated for nested collections, generics
and user-registered hooks; crossing a collection or hook boundary never
resets the counters.

Work performed entirely *inside* a user hook is outside cattrs' control. Such
work is only metered when it flows back through the converter's hooks, so a
user hook which loops over its input without calling back into the converter
is an opaque unit of work to the budget.
"""

from __future__ import annotations

from attrs import define

from .errors import ConversionBudgetExceeded

__all__ = ["ConversionBudget"]


@define
class ConversionBudget:
    """A budget for a single conversion.

    The first four fields are the configured limits; ``None`` means unlimited.
    The remaining fields are live counters, exposed for tests and diagnostics.
    A budget is single-use: install it with
    :meth:`BaseConverter.configure_budget` for one call and replace it for the
    next call.

    :param node_limit: Maximum number of hook node visits.
    :param depth_limit: Maximum nesting depth of hook calls.
    :param container_limit: Maximum number of generated output container
        elements.
    :param error_limit: Maximum number of ordinary errors collected by
        detailed validation.

    .. versionadded:: NEXT
    """

    node_limit: int | None
    depth_limit: int | None
    container_limit: int | None
    error_limit: int | None

    nodes: int = 0
    depth: int = 0
    max_depth: int = 0
    container_elements: int = 0
    errors: int = 0
    exhausted: bool = False

    @classmethod
    def of(
        cls,
        node_limit: int | None = None,
        depth_limit: int | None = None,
        container_limit: int | None = None,
        error_limit: int | None = None,
    ) -> ConversionBudget:
        """Create a budget from keyword-style limits."""
        for name, value in (
            ("node_limit", node_limit),
            ("depth_limit", depth_limit),
            ("container_limit", container_limit),
            ("error_limit", error_limit),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")
        return cls(node_limit, depth_limit, container_limit, error_limit)

    @property
    def active(self) -> bool:
        """Whether any limit is configured."""
        return (
            self.node_limit is not None
            or self.depth_limit is not None
            or self.container_limit is not None
            or self.error_limit is not None
        )

    def _exceed(self, resource: str, limit: int, value: int) -> None:
        self.exhausted = True
        raise ConversionBudgetExceeded(resource, limit, value, self)

    def enter_node(self) -> None:
        """Account for one hook node visit and one level of nesting."""
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            self._exceed("nodes", self.node_limit, self.nodes)
        self.depth += 1
        if self.depth > self.max_depth:
            self.max_depth = self.depth
        if self.depth_limit is not None and self.depth > self.depth_limit:
            # This visit never gets an ``exit_node``; roll the depth back so
            # unwinding the outer visits restores the counter to zero.
            self.depth -= 1
            self._exceed("depth", self.depth_limit, self.depth + 1)

    def exit_node(self) -> None:
        """Leave a hook node, restoring the nesting depth.

        Always safe to call, even after a limit terminated the conversion, so
        nested entry points can balance their ``enter_node`` in a ``finally``.
        """
        self.depth -= 1

    def add_container_elements(self, count: int) -> None:
        """Account for ``count`` elements about to join an output container."""
        if count:
            self.container_elements += count
            if (
                self.container_limit is not None
                and self.container_elements > self.container_limit
            ):
                self._exceed(
                    "containers", self.container_limit, self.container_elements
                )

    def add_error(self) -> None:
        """Account for one ordinary error collected by detailed validation."""
        self.errors += 1
        if self.error_limit is not None and self.errors > self.error_limit:
            self._exceed("errors", self.error_limit, self.errors)
