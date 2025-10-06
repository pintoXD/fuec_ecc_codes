import random
import itertools
import pytest

from fuec_encoder_decoder import Area, ControlSpec, FUECBuilder, enumerate_bursts


def flip_bits(cw, indices):
    out = list(cw)
    for i in indices:
        out[i] ^= 1
    return out


@pytest.fixture(scope="module")
def code_48_32():
    """Build once: k=32, r fixed to 16, span bursts up to length 4 correctable over full 48-bit space."""
    k = 32
    r = 16
    area = Area("A", tuple(range(0, k + r)))
    specs = [
        ControlSpec(
            area=area, correct=["single", "span_burst<=L"], detect=[], params={"L": 4}
        )
    ]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1209))
    code = builder.build(min_r=r, max_r=r, max_attempts_per_r=100000)
    rnd = random.Random(123)
    data = [rnd.randint(0, 1) for _ in range(k)]
    cw = code.encode(data)
    assert code.syndrome(cw) == 0
    return code, area, data, cw


def test_48_32_code_properties(code_48_32):
    code, area, data, cw = code_48_32
    assert code.k == 32
    assert code.r == 16
    assert len(cw) == code.n == 48


def test_48_32_single_errors_corrected(code_48_32):
    code, area, data, cw = code_48_32
    for i in area.indices:
        rcv = flip_bits(cw, [i])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert ev is not None
        assert corrected == cw


def test_48_32_adjacent_doubles_corrected(code_48_32):
    code, area, data, cw = code_48_32
    indices_set = set(area.indices)
    for i in area.indices:
        if i + 1 not in indices_set:
            continue
        rcv = flip_bits(cw, [i, i + 1])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert ev is not None
        assert corrected == cw


def _span3_patterns(start):
    # windows [start, start+1, start+2]
    return [
        (start, start + 1, start + 2),  # 111
        (start, start + 2),
    ]  # 101 endpoints only


@pytest.mark.parametrize(
    "start", list(range(0, 48 - 2))[:12]
)  # limit to first 12 for runtime
def test_48_32_span3_patterns_corrected(code_48_32, start):
    code, area, data, cw = code_48_32
    indices_set = set(area.indices)
    if not {start, start + 2}.issubset(indices_set):
        pytest.skip("window exceeds area")
    for ev in _span3_patterns(start):
        rcv = flip_bits(cw, list(ev))
        corrected, ok, matched = code.decode(rcv)
        assert ok, f"Failed to correct pattern {ev}"
        assert matched is not None
        assert corrected == cw


def _span4_patterns(start):
    # Endpoints start, start+3; interior: start+1, start+2 optional => 4 variants
    a, b, c, d = start, start + 1, start + 2, start + 3
    return [(a, d), (a, b, d), (a, c, d), (a, b, c, d)]


@pytest.mark.parametrize(
    "start", list(range(0, 48 - 3))[:10]
)  # limit to first 10 starts
def test_48_32_span4_patterns_corrected(code_48_32, start):
    code, area, data, cw = code_48_32
    indices_set = set(area.indices)
    if not {start, start + 3}.issubset(indices_set):
        pytest.skip("window exceeds area")
    for ev in _span4_patterns(start):
        rcv = flip_bits(cw, list(ev))
        corrected, ok, matched = code.decode(rcv)
        assert ok, f"Failed to correct pattern {ev}"
        assert matched is not None
        assert corrected == cw


def test_48_32_random_subset_of_span4_windows(code_48_32):
    code, area, data, cw = code_48_32
    rng = random.Random(999)
    starts = rng.sample(range(0, 48 - 3), 8)
    for s in starts:
        for ev in _span4_patterns(s):
            rcv = flip_bits(cw, list(ev))
            corrected, ok, matched = code.decode(rcv)
            assert ok
            assert matched is not None
            assert corrected == cw
