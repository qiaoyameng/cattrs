# Validation

_cattrs_ has a detailed validation mode since version 22.1.0, and this mode is enabled by default.
When running under detailed validation, the structuring hooks are slightly slower but produce richer and more precise error messages.
Unstructuring hooks are not affected.

## Detailed Validation

```{versionadded} 22.1.0

```
In detailed validation mode, any structuring errors will be grouped and raised together as a {class}`cattrs.BaseValidationError`, which is a [PEP 654 ExceptionGroup](https://www.python.org/dev/peps/pep-0654/).
ExceptionGroups are special exceptions which contain lists of other exceptions, which may themselves be other ExceptionGroups.
In essence, ExceptionGroups are trees of exceptions.

When structuring a class, _cattrs_ will gather any exceptions on a field-by-field basis and raise them as a {class}`cattrs.ClassValidationError`, which is a subclass of {class}`BaseValidationError <cattrs.BaseValidationError>`.

When structuring sequences and mappings, _cattrs_ will gather any exceptions on a key- or index-basis and raise them as a {class}`cattrs.IterableValidationError`, which is a subclass of {class}`BaseValidationError <cattrs.BaseValidationError>`.

The exceptions will also have their `__notes__` attributes set, as per [PEP 678](https://www.python.org/dev/peps/pep-0678/), showing the field, key or index for each inner exception.

A simple example involving a class containing a list and a dictionary:

```python
@define
class Class:
    a_list: list[int]
    a_dict: dict[str, int]

>>> structure({"a_list": ["a"], "a_dict": {"str": "a"}}, Class)
  + Exception Group Traceback (most recent call last):
  |   File "<stdin>", line 1, in <module>
  |   File "/Users/tintvrtkovic/pg/cattrs/src/cattr/converters.py", line 276, in structure
  |     return self._structure_func.dispatch(cl)(obj, cl)
  |   File "<cattrs generated structure __main__.Class>", line 14, in structure_Class
  |     if errors: raise __c_cve('While structuring Class', errors, __cl)
  | cattrs.errors.ClassValidationError: While structuring Class
  +-+---------------- 1 ----------------
    | Exception Group Traceback (most recent call last):
    |   File "<cattrs generated structure __main__.Class>", line 5, in structure_Class
    |     res['a_list'] = __c_structure_a_list(o['a_list'], __c_type_a_list)
    |   File "/Users/tintvrtkovic/pg/cattrs/src/cattr/converters.py", line 457, in _structure_list
    |     raise IterableValidationError(
    | cattrs.errors.IterableValidationError: While structuring list[int]
    | Structuring class Class @ attribute a_list
    +-+---------------- 1 ----------------
      | Traceback (most recent call last):
      |   File "/Users/tintvrtkovic/pg/cattrs/src/cattr/converters.py", line 450, in _structure_list
      |     res.append(handler(e, elem_type))
      |   File "/Users/tintvrtkovic/pg/cattrs/src/cattr/converters.py", line 375, in _structure_call
      |     return cl(obj)
      | ValueError: invalid literal for int() with base 10: 'a'
      | Structuring list[int] @ index 0
      +------------------------------------
    +---------------- 2 ----------------
    | Exception Group Traceback (most recent call last):
    |   File "<cattrs generated structure __main__.Class>", line 10, in structure_Class
    |     res['a_dict'] = __c_structure_a_dict(o['a_dict'], __c_type_a_dict)
    |   File "", line 17, in structure_mapping
    | cattrs.errors.IterableValidationError: While structuring dict
    | Structuring class Class @ attribute a_dict
    +-+---------------- 1 ----------------
      | Traceback (most recent call last):
      |   File "", line 5, in structure_mapping
      | ValueError: invalid literal for int() with base 10: 'a'
      | Structuring mapping value @ key 'str'
      +------------------------------------
```

### Transforming Exceptions into Error Messages

```{versionadded} 23.1.0

```

ExceptionGroup stack traces are useful while developing, but sometimes a more compact representation of validation errors is required.
_cattrs_ provides a helper function, {func}`cattrs.transform_error`, which transforms validation errors into lists of error messages.

The example from the previous paragraph produces the following error messages:

```{testsetup} class
@define
class Class:
    a_list: list[int]
    a_dict: dict[str, int]
```

```{doctest} class

>>> from cattrs import structure, transform_error

>>> try:
...     structure({"a_list": ["a"], "a_dict": {"str": "a"}}, Class)
... except Exception as exc:
...     print(transform_error(exc))
['invalid value for type, expected int @ $.a_list[0]', "invalid value for type, expected int @ $.a_dict['str']"]
```

A small number of built-in exceptions are converted into error messages automatically.
This can be further customized by providing {func}`cattrs.transform_error` with a function that it can use to turn individual, non-ExceptionGroup exceptions into error messages.
A useful pattern is wrapping the default, {func}`cattrs.v.format_exception` function.

```
>>> from cattrs.v import format_exception

>>> def my_exception_formatter(exc: BaseException, type) -> str:
...     if isinstance(exc, MyInterestingException):
...         return "My error message"
...     return format_exception(exc, type)

>>> try:
...     structure(..., Class)
... except Exception as exc:
...     print(transform_error(exc, format_exception=my_exception_formatter))
```

If even more customization is required, {func}`cattrs.transform_error` can be copied over into your codebase and adjusted as needed.

## Non-detailed Validation

Non-detailed validation can be enabled by initializing any of the converters with `detailed_validation=False`.
In this mode, any errors during un/structuring will bubble up directly as soon as they happen.

## Conversion Budgets

```{versionadded} NEXT

```

Untrusted input can request an unbounded amount of work: very deeply nested models, very large collections or inputs crafted to provoke many detailed validation errors.
A {class}`cattrs.ConversionBudget` caps the resources a single {meth}`BaseConverter.structure` or {meth}`BaseConverter.unstructure` call may use.

Four independent limits are supported; leaving a limit as `None` makes it unlimited:

- `node_limit` counts hook visits, including hooks for nested collections, generic classes and user-registered hooks.
- `depth_limit` counts the current nesting depth of those hook visits.
- `container_limit` counts the elements of target containers, both when structuring and unstructuring. The charge is applied *before* the target container is allocated, so an over-limit container is never built. Sized iterables are charged their length; an unsized iterable is charged as one element, since its length cannot be known without consuming it.
- `error_limit` counts ordinary errors collected by detailed validation. When it is exceeded the conversion stops instead of aggregating yet another error.

The same budget instance is shared by every hook taking part in one conversion.
Crossing a collection or hook boundary never resets the counters.
Budgets are single-use: install a fresh one (or clear it with `None`) before the next conversion.

```python
from cattrs import (
    ConversionBudget,
    ConversionBudgetExceeded,
    Converter,
)

converter = Converter()
budget = ConversionBudget.of(node_limit=1000, depth_limit=20, container_limit=10_000, error_limit=50)
converter.configure_budget(budget)

try:
    converter.structure(untrusted_data, Model)
except Exception as exc:
    # A ConversionBudgetExceeded is terminal: detailed validation never groups
    # it with ordinary field errors and never continues past it.
    if isinstance(exc, ConversionBudgetExceeded):
        print(exc.resource, exc.limit, exc.value)
```

`conversion_budget` is also available as a context manager, and a budget may be passed to the converter constructor.
Copied converters do not inherit the source converter's budget.

After a conversion, the budget's counters (`nodes`, `max_depth`, `container_elements`, `errors`) describe the work performed.
Once a limit terminates a conversion the budget is marked `exhausted`.
Clearing or replacing the budget leaves no counters behind on the converter, so the same converter can be reused immediately.

### Measurement Boundaries

_cattrs_ meters work that flows through its own dispatch and hook machinery.
Work performed *inside* a user hook is only visible to the budget when the hook calls back into the converter.

For example, a user hook that builds a list directly and never calls the converter is a single opaque unit of work: its internal loop is not charged per element, and its output container is not charged to `container_limit`.
If that same hook calls `converter.structure` for each element, every call is metered and the shared budget is enforced.

The budget cannot interrupt arbitrary Python code inside a user hook, so a hook that loops on its own input should apply its own limits if it needs this protection.
