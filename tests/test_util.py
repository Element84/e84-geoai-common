from e84_geoai_common.util import chunk_items, unique_by


def test_chunk_items() -> None:
    result = chunk_items(range(4), 1)
    assert list(result) == [[0], [1], [2], [3]]
    result = chunk_items(range(8), 2)
    assert list(result) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    result = chunk_items(range(8), 3)
    assert list(result) == [[0, 1, 2], [3, 4, 5], [6, 7]]
    result = chunk_items(range(8), 6)
    assert list(result) == [[0, 1, 2, 3, 4, 5], [6, 7]]
    result = chunk_items(range(8), 10)
    assert list(result) == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_unique_by() -> None:
    # tests uniqueness with default key function
    output = unique_by([1, 2, 3, 1, 2, 3, 0, 1, 1])
    assert list(output) == [1, 2, 3, 0]

    # Tests uniqueness with a key function
    words = ["foo", "rock", "snow", "a", "longer", "b"]
    output = unique_by(words, key_fn=lambda s: len(s))
    assert list(output) == ["foo", "rock", "a", "longer"]

    # Verifies the duplicate handler function is called with correct args
    dups: list[tuple[str, int]] = []

    output = unique_by(
        words,
        key_fn=lambda s: len(s),
        duplicate_handler_fn=lambda s, length: dups.append((s, length)),
    )
    assert list(output) == ["foo", "rock", "a", "longer"]
    assert dups == [("snow", 4), ("b", 1)]
