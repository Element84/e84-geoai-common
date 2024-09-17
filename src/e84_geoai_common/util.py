import time
import os
import textwrap
from typing import Any, Callable, TypeVar


def get_env_var(name: str, default: str | None = None) -> str:
    value = os.getenv(name) or default

    if value is None:
        raise Exception(f"Env var {name} must be set")
    return value


def dedent(text: str) -> str:
    return textwrap.dedent(text).strip()


T = TypeVar("T", bound=Callable[..., Any])


def timed_function(func: T) -> T:
    """
    A decorator for timing a function call.

    This decorator will print the execution time of the decorated function after it runs.

    Parameters:
    func (Callable): The function to be timed.

    Returns:
    Callable: The decorated function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()  # capture the start time before executing
        result = func(*args, **kwargs)  # execute the function
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper  # type: ignore
