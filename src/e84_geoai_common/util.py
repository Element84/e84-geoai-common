import time
import os
import textwrap
from typing import Any, Callable, TypeVar


def get_env_var(name: str, default: str | None = None) -> str:
    """
    Retrieves the value of an environment variable.
    """
    value = os.getenv(name) or default

    if value is None:
        raise Exception(f"Env var {name} must be set")
    return value


def dedent(text: str) -> str:
    """
    Remove common leading whitespace from every line in a multi-line string.

    Parameters:
    text (str): The multi-line string with potentially uneven indentation.

    Returns:
    str: The modified string with common leading whitespace removed from every line.

    Raises:
    None

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
    """
    Remove common leading whitespace from every line in a multi-line string and convert it into a single line.

    Parameters:
    text (str): The multi-line string with potentially uneven indentation.

    Returns:
    str: The modified string with common leading whitespace removed from every line and converted into a single line.

    Raises:
    None

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
    """
    return dedent(text).replace("\n", " ")


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
