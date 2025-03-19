import functools
import logging
import os
import textwrap
from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar, cast, overload

log = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])


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


@overload
def timed_function(arg: F) -> F: ...


@overload
def timed_function(arg: logging.Logger | None = None) -> Callable[[F], F]: ...


def timed_function(
    arg: Callable[..., Any] | logging.Logger | None = None,
) -> Callable[..., Any] | Callable[[F], F]:
    """A decorator that times function execution.

    Can be used with or without arguments:

    @timed_function
    def func(): ...

    @timed_function(logger)
    def func(): ...
    """
    # If called without arguments, arg will be the function itself
    if callable(arg):
        func = cast(F, arg)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            # Use default logger in this case
            log.info("%s took %.4f seconds to execute", func.__name__, end_time - start_time)
            return result

        return cast(F, wrapper)

    # If called with arguments (or None), return a decorator that will be called with the function
    custom_logger = arg  # This is the logger passed in or None

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            # Use custom logger if provided, otherwise create a default one
            logger_to_use = custom_logger or log
            logger_to_use.info(
                "%s took %.4f seconds to execute", func.__name__, end_time - start_time
            )
            return result

        return cast(F, wrapper)

    return decorator
