"""Provides a utility for getting better pydantic differences with pytest.

Instructions for use:
1. Create a conftest.py file at the root of your project's testing directory.
2. Put in it the following content:

```
from e84_geoai_common.llm.tests.pydantic_compare import custom_assertrepr_compare

# Register custom assertion for pydantic models
pytest_assertrepr_compare = custom_assertrepr_compare
```

"""

import difflib
from typing import Any

from pydantic import BaseModel


def generate_pydantic_diff(before: BaseModel, after: BaseModel) -> list[str]:
    """Generates a diff of pydantic models that compares their JSON representations."""

    def _dump_for_diff(m: BaseModel) -> list[str]:
        return m.model_dump_json(indent=2).splitlines(keepends=True)

    return list(difflib.unified_diff(_dump_for_diff(before), _dump_for_diff(after)))


def custom_assertrepr_compare(op: str, left: Any, right: Any) -> list[str] | None:  # noqa: ANN401
    """Implements a custom compare for pytest.

    Use this by assigning to pytest_assertrepr_compare in a conftest.py file at the root of your
    test hierarchy.
    """
    if op == "==" and isinstance(left, BaseModel) and isinstance(right, BaseModel):
        return generate_pydantic_diff(right, left)
    return None
