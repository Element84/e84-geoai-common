from dataclasses import dataclass, field
from time import time

import os
import textwrap
from typing import Any, Callable, Sequence, TypeVar

import humanize

T = TypeVar("T")
K = TypeVar("K")


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


TimedFn = TypeVar("TimedFn", bound=Callable[..., Any])


def timed_function(func: TimedFn) -> TimedFn:
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


def identity(item: T) -> T:
    return item


def group_by(items: Sequence[T], fn: Callable[[T], K] = identity) -> dict[K, list[T]]:
    """
    Groups items in a sequence by a key function.

    Parameters:
    items (Sequence[T]): A sequence of items to be grouped.
    fn (Callable[[T], K]): A function that takes an item and returns the key to group by.

    Returns:
    dict[K, list[T]]: A dictionary where keys are the results of the key function,
                      and values are lists of items that produced that key.

    Example:
    # Group numbers by their remainder when divided by 3
    numbers = [1, 2, 3, 4, 5, 6]
    grouped = group_by(numbers, lambda x: x % 3)
    # Result: {1: [1, 4], 2: [2, 5], 0: [3, 6]}

    # Group strings by their length
    words = ["cat", "dog", "mouse", "rat", "elephant"]
    grouped = group_by(words, len)
    # Result: {3: ["cat", "dog", "rat"], 5: ["mouse"], 8: ["elephant"]}
    """
    groups: dict[K, list[T]] = {}

    for item in items:
        key = fn(item)
        if key in groups:
            groups[key].append(item)
        else:
            groups[key] = [item]
    return groups


def count_by(items: Sequence[T], fn: Callable[[T], K] = identity) -> dict[K, int]:
    groups: dict[K, int] = {}

    for item in items:
        key = fn(item)
        if key in groups:
            groups[key] += 1
        else:
            groups[key] = 1
    return groups


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
