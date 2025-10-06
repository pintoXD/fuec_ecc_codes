import pytest

from fuec_encoder_decoder import enumerate_bursts


def tuple_list(values):
    """Helper to normalize list of tuples for comparison (order-insensitive)."""
    return sorted(values)


def test_enumerate_bursts_basic_multiple_runs():
    # indices = [0, 1, 2, 5, 6]
    indices = range(0,8)
    bursts = enumerate_bursts(indices, length=3, exact=True)
    # Expected from runs (0..2) and (5..6): lengths 2 and 3 from first run, only length 2 from second run
    # expected = [(0, 1), (1, 2), (0, 1, 2), (5, 6)]
    print(f"bursts: {tuple_list(bursts)}")
    # assert tuple_list(bursts) == tuple_list(expected)


def test_enumerate_bursts_exact_length():
    indices = [0, 1, 2, 5, 6]
    bursts = enumerate_bursts(indices, length=2, exact=True)
    # Only contiguous windows of length exactly 2
    expected = [(0, 1), (1, 2), (5, 6)]
    assert bursts == expected  # order is deterministic: run order then sliding start


def test_enumerate_bursts_empty_indices():
    assert enumerate_bursts([], length=4) == []


def test_enumerate_bursts_length_less_than_two():
    # length=1 still produces bursts of length 2 (floor at 2) when available
    indices = [10, 11, 12]
    bursts = enumerate_bursts(indices, length=1, exact=False)
    # lengths considered become [2]; windows: (10,11),(11,12)
    assert bursts == [(10, 11), (11, 12)]


def test_enumerate_bursts_large_length_truncation():
    indices = [0, 1, 2]
    bursts = enumerate_bursts(indices, length=5, exact=False)
    # lengths considered [2,3,4,5]; only 2 and 3 fit run_len=3
    expected = [(0, 1), (1, 2), (0, 1, 2)]
    assert tuple_list(bursts) == tuple_list(expected)


def test_enumerate_bursts_with_duplicates_and_gap():
    indices = [2, 2, 3, 4, 10]
    bursts = enumerate_bursts(indices, length=3, exact=False)
    # Runs: (2..4) len=3 -> lengths 2 & 3: (2,3),(3,4),(2,3,4); solitary 10 ignored
    expected = [(2, 3), (3, 4), (2, 3, 4)]
    assert tuple_list(bursts) == tuple_list(expected)


def test_enumerate_bursts_exact_excludes_too_short_runs():
    indices = [5, 6, 9]  # runs (5..6) length 2; (9..9) length 1
    bursts = enumerate_bursts(indices, length=3, exact=True)
    # No run long enough for exact length 3
    assert bursts == []
