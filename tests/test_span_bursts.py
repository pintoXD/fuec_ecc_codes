import pytest
from fuec_encoder_decoder import enumerate_burst_spans


def test_span_burst_length_3_includes_101():
    indices = list(range(0, 10))
    patterns = enumerate_burst_spans(indices, length=3, exact=True)
    # Extract only those within a single 3-span starting at 0 for simplicity
    # Expect (0,2) with optional middle 1 -> patterns (0,2) and (0,1,2)
    assert (0, 2) in patterns  # 101 pattern (endpoints only)
    assert (0, 1, 2) in patterns  # 111 solid burst


def test_span_burst_upto_3_accumulates():
    indices = [0,1,2,3]
    pats = enumerate_burst_spans(indices, length=3, exact=False)
    # Spans length 2 windows: (0,1),(1,2),(2,3)
    # Span length 3 windows: start 0 -> endpoints 0,2 with interior variants (0,2),(0,1,2); start 1 -> (1,3),(1,2,3)
    for ev in [(0,1),(1,2),(2,3),(0,2),(0,1,2),(1,3),(1,2,3)]:
        assert ev in pats


def test_span_burst_length_4_variants():
    indices = range(0,16)
    pats = enumerate_burst_spans(indices, length=4, exact=True)
    # Length 4 span endpoints 5,8 interior {6,7} -> 2^(2)=4 variants
    expected = { (5,8), (5,6,8), (5,7,8), (5,6,7,8) }
    print(f"patterns: {pats}")
    print(f"len patterns: {len(pats)}")
    assert expected.issubset(set(pats))

