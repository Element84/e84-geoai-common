import functools
import logging
from collections.abc import Callable
from time import perf_counter
from typing import Any, cast, overload

log = logging.getLogger(__name__)

langfuse = None
try:
    from langfuse import get_client

    langfuse = get_client()
except ImportError:
    pass


@overload
def timed_function[F: Callable[..., Any]](arg: F) -> F: ...


@overload
def timed_function[F: Callable[..., Any]](
    arg: logging.Logger | None = None,
) -> Callable[[F], F]: ...


def timed_function[F: Callable[..., Any]](
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
        func = cast("F", arg)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            # Use default logger in this case
            log.info("%s took %.4f seconds to execute", func.__name__, end_time - start_time)
            return result

        return cast("F", wrapper)

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

        return cast("F", wrapper)

    return decorator


@overload
def observe[F: Callable[..., Any]](func: F) -> F: ...


@overload
def observe[F: Callable[..., Any]](
    func: None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Callable[[F], F]: ...


def observe[F: Callable[..., Any]](
    func: F | None = None,
    **kwargs: Any,
) -> F | Callable[[F], F]:
    """A decorator that adds Langfuse observability if available.

    Attempts to import and use Langfuse's observe decorator. If Langfuse is not
    available, the function is returned unchanged.
    """
    try:
        from langfuse import observe  # pyright: ignore[reportUnknownVariableType]  # noqa: PLC0415

        def decorator(func: F) -> F:
            return observe(func, **kwargs)

        if func is None:
            return decorator
        return decorator(func)
    except ImportError:

        def decorator(func: F) -> F:
            return func

        if func is None:
            return decorator
        return func


def update_current_generation(**kwargs: Any) -> None:  # noqa: ANN401
    """Wrapper for langfuse update_current_generation."""
    if langfuse:
        langfuse.update_current_generation(**kwargs)
