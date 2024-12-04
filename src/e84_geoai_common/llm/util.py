import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

log = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable[..., Any])


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
