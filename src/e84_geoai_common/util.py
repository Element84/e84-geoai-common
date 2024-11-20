from dataclasses import dataclass, field
from time import time

import os
import textwrap
from typing import Any, Callable, TypeVar

import humanize


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
        start_time = time()  # capture the start time before executing
        result = func(*args, **kwargs)  # execute the function
        end_time = time()
        print(f"{func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper  # type: ignore


@dataclass
class ProcessTracker:
    total: int
    start_time: float = field(default_factory=lambda: time())
    completed: int = 0

    def increment_completed(self, num_completed: int = 1) -> None:
        self.completed = self.completed + num_completed

    @property
    def elapsed_secs(self) -> float:
        return time() - self.start_time

    @property
    def elapsed_ms(self) -> int:
        return int(self.elapsed_secs * 1000)

    @property
    def completed_per_sec(self) -> float:
        return self.completed / self.elapsed_secs

    @property
    def num_left(self) -> int:
        return self.total - self.completed

    @property
    def secs_left(self) -> int:
        return int(self.num_left / self.completed_per_sec)

    @property
    def completed_pct(self) -> float:
        return self.completed / self.total

    def report(self) -> None:
        pct = int(self.completed_pct * 100)
        if self.completed == 0:
            rate = "Unknown"
            time_left = "Unknown"
        else:
            rate = round(self.completed_per_sec, 1)
            time_left = humanize.naturaldelta(self.secs_left)
        print(
            f"Completed: {pct}% ({self.completed} out of {self.total}) Rate: {rate} per sec Time Left: {time_left} "
        )
