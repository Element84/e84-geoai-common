import logging
import os
import textwrap
from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

log = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable[..., Any])


def get_env_var(name: str, default: str | None = None) -> str:
    """Retrieve the value of an environment variable."""
    value = os.getenv(name) or default
    if value is None:
        msg = f"Env var {name} must be set"
        raise ValueError(msg)
    return value


def dedent(text: str) -> str:
    """Remove common leading whitespace from every line in a multi-line string.

    Args:
        text (str): The multi-line string with potentially uneven indentation.

    Returns:
        str: The modified string with common leading whitespace removed from
        every line.

    Example:
        text = '''
            Lorem ipsum dolor sit amet,
            consectetur adipiscing elit,
            sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        '''
        result = dedent(text)
        print(result)
        # Output:
        # 'Lorem ipsum dolor sit amet,
        # consectetur adipiscing elit,
        # sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'
    """
    return textwrap.dedent(text).strip()


def singleline(text: str) -> str:
    """Remove common leading whitespace from every line in a multi-line string.

    Args:
        text (str): The multi-line string with potentially uneven indentation.

    Returns:
        str: The modified string with common leading whitespace removed from
            every line and converted into a single line.

    Example:
        text = '''
            Lorem ipsum dolor sit amet,
            consectetur adipiscing elit,
            sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        '''
        result = singleline(text)
        print(result)
        # Output:
        # 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.'
    """  # noqa: E501
    return dedent(text).replace("\n", " ")


def timed_function(func: T) -> T:
    """Decorate a function to log execution time.

    This decorator will print the execution time of the decorated function
    after it finishes executing.

    Args:
        func (Callable): The function to be timed.

    Returns:
        Callable: The decorated function.
    """

    def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        diff = end_time - start_time
        log.info("%s took %f seconds to run.", func.__name__, diff)
        return result

    return wrapper  # type: ignore[reportReturnType]
