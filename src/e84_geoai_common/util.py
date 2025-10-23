import functools
import logging
import os
import textwrap
from collections.abc import Callable, Generator, Iterable
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast, overload

from botocore.exceptions import ClientError
from mypy_boto3_s3 import S3Client

if TYPE_CHECKING:
    from mypy_boto3_s3.literals import BucketLocationConstraintType

log = logging.getLogger(__name__)


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


def unique_by[T, K](
    items: Iterable[T],
    *,
    key_fn: Callable[[T], K] = lambda x: x,
    duplicate_handler_fn: Callable[[T, K], None] | None = None,
) -> Generator[T, None, None]:
    """Filter an iterator to yield only items with unique keys.

    This function takes an iterator and yields only the first item encountered for each unique key,
    filtering out subsequent items with duplicate keys. Optionally handles duplicates with a custom
    function.

    Args:
        items (Iterable[T]): The input iterator containing items to be filtered for uniqueness.
        key_fn (Callable[[T], K]): A function that extracts the key to determine uniqueness.
            Defaults to the identity function (using the item itself as its key).
        duplicate_handler_fn (Callable[[T, K], None] | None): Optional function to call when
            encountering duplicate items. Receives the duplicate item and its key. Defaults to None.

    Yields:
        T: Items from the original iterator with unique keys.

    Example:
        >>> data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, {"id": 1, "name": "Alex"}]
        >>> list(unique_by(data, key_fn=lambda x: x["id"]))
        [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    """
    keys: set[K] = set()

    for item in items:
        key = key_fn(item)
        if key not in keys:
            keys.add(key)
            yield item
        elif duplicate_handler_fn:
            duplicate_handler_fn(item, key)


def ensure_bucket_exists(
    s3_client: S3Client, bucket: str, *, create_buckets_if_missing: bool = False
) -> None:
    """Ensure that the specified bucket exists and creates it if it doesn't and flag is true."""
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as e:
        if create_buckets_if_missing:
            region = s3_client.meta.region_name
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket)
            else:
                region = cast("BucketLocationConstraintType", region)
                s3_client.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": region},
                )
        else:
            msg = (
                f"Bucket {bucket} does not exist or you do not have permissions,"
                f" and create_buckets_if_missing flag is False."
            )
            raise ValueError(msg) from e


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
        if func is None:
            raise ValueError("Expected a value for func.") from None
        return func
