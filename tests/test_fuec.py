import random
import itertools
import pytest

from fuec_encoder_decoder import Area, ControlSpec, FUECBuilder, enumerate_bursts


def flip_bits(cw, indices):
    out = list(cw)
    for i in indices:
        out[i] ^= 1
    return out


def test_example_build_and_basic_decode():
    k = 16
    area_a = Area("A", tuple(range(0, 8)))
    area_b = Area("B", tuple(range(8, 16)))
    specs = [
        ControlSpec(area=area_a, correct=["single", "double_adjacent"], detect=[]),
        ControlSpec(area=area_b, correct=[], detect=["single"]),
    ]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1))
    code = builder.build(max_r=12, max_attempts_per_r=4000)
    assert code.k == k
    assert code.r >= 1

    # Random data
    rnd = random.Random(123)
    data = [rnd.randint(0, 1) for _ in range(k)]
    cw = code.encode(data)
    assert len(cw) == code.n
    # No error syndrome is zero
    assert code.syndrome(cw) == 0

    # All singles in area A must be corrected
    for i in area_a.indices:
        rcv = flip_bits(cw, [i])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert code.syndrome(corrected) == 0

    # All adjacent doubles in area A must be corrected
    for i in range(min(area_a.indices), max(area_a.indices)):
        if i + 1 not in set(area_a.indices):
            continue
        rcv = flip_bits(cw, [i, i + 1])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert code.syndrome(corrected) == 0

    # Singles in area B must be detected but not corrected
    for i in area_b.indices:
        rcv = flip_bits(cw, [i])
        corrected, ok, ev = code.decode(rcv)
        assert not ok


def test_burst_correction_small():
    # Small code, two areas; require burst<=3 correction in area 0..5
    k = 8
    area_a = Area("A", tuple(range(0, 6)))
    area_b = Area("B", tuple(range(6, 8)))
    specs = [
        ControlSpec(area=area_a, correct=["burst<=L"], detect=[], params={"L": 3}),
        ControlSpec(area=area_b, correct=[], detect=["single"]),
    ]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(2))
    code = builder.build(max_r=14, max_attempts_per_r=8000)
    rnd = random.Random(456)
    data = [rnd.randint(0, 1) for _ in range(k)]
    cw = code.encode(data)
    # All bursts up to 3 inside area A are correctable
    bursts = enumerate_bursts(area_a.indices, 3, exact=False)
    assert bursts, "Expected non-empty set of bursts"
    for ev in bursts[:20]:  # sample to keep test time bounded
        rcv = flip_bits(cw, list(ev))
        corrected, ok, _ = code.decode(rcv)
        assert ok
        assert code.syndrome(corrected) == 0


def test_variable_redundancy_bounds():
    # Same spec, verify builder picks minimal feasible r within max_r
    k = 10
    area = Area("A", tuple(range(0, 10)))
    specs = [ControlSpec(area=area, correct=["single"], detect=[])]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(3))
    code = builder.build(max_r=8, max_attempts_per_r=2000)
    assert 1 <= code.r <= 8
    # Check all single errors across the whole word are corrected
    rnd = random.Random(789)
    data = [rnd.randint(0, 1) for _ in range(k)]
    cw = code.encode(data)
    for i in range(code.n):
        rcv = flip_bits(cw, [i])
        corrected, ok, _ = code.decode(rcv)
        # Parity-bit single flips are detect-only by default in builder; here area covers only data positions,
        # so data singles must be corrected, parity singles detected-not-corrected.
        if i in area.indices:
            assert ok
        else:
            assert not ok


def test_example_build_and_basic_decode_23_16():
    k = 12
    r_bits = 6
    area_a = Area("A", tuple(range(0, 4)))
    area_b = Area("B", tuple(range(4, 8)))
    area_c = Area("C", tuple(range(8, 16)))
    specs = [
        ControlSpec(
            area=area_a,
            correct=["single", "double_adjacent"],
            detect=["burst==L", "double"],
            params={"L": 3},
        ),
        ControlSpec(
            area=area_b,
            correct=["single", "double_adjacent"],
            detect=[]
        ),
        ControlSpec(
            area=area_c,
            correct=["single"],
            detect=["double_adjacent"]
        ),
    ]

    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1))
    code = builder.build(max_r=r_bits, max_attempts_per_r=100000)
    assert code.k == k
    assert code.r >= 1

    # Random data
    rnd = random.Random(123)
    data = [rnd.randint(0, 1) for _ in range(k)]
    cw = code.encode(data)
    assert len(cw) == code.n
    # No error syndrome is zero
    assert code.syndrome(cw) == 0

    # All singles in area A must be corrected
    for i in area_a.indices:
        rcv = flip_bits(cw, [i])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert code.syndrome(corrected) == 0

    # All adjacent doubles in area A must be corrected
    for i in range(min(area_a.indices), max(area_a.indices)):
        if i + 1 not in set(area_a.indices):
            continue
        rcv = flip_bits(cw, [i, i + 1])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert code.syndrome(corrected) == 0

    # All 3-bit burst errors in area A must be detected but not corrected.
    for i in range(min(area_a.indices), max(area_a.indices)):
        if i + 1 not in set(area_a.indices):
            continue
        rcv = flip_bits(cw, [i, i + 2])
        corrected, ok, ev = code.decode(rcv)
        assert not ok
        assert None is ev
        assert corrected == rcv  # No correction

    # Singles in area B must be detected and corrected
    for i in area_b.indices:
        rcv = flip_bits(cw, [i])
        corrected, ok, ev = code.decode(rcv)
        assert ok

    # All adjacent doubles in area B must be corrected
    for i in range(min(area_b.indices), max(area_b.indices)):
        if i + 1 not in set(area_b.indices):
            continue
        rcv = flip_bits(cw, [i, i + 1])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert code.syndrome(corrected) == 0

    # Singles in area C must be detected and corrected
    for i in area_c.indices:
        rcv = flip_bits(cw, [i])
        corrected, ok, ev = code.decode(rcv)
        assert ok

    # All adjacent doubles in area C must be detected but not corrected.
    for i in range(min(area_c.indices), max(area_c.indices)):
        if i + 1 not in set(area_c.indices):
            continue
        rcv = flip_bits(cw, [i, i + 1])
        corrected, ok, ev = code.decode(rcv)
        assert not ok
        assert ev is None
        assert corrected == rcv  # No correction


def test_example_build_and_basic_decode_12_8():
    k = 8
    area_a = Area("A", tuple(range(0, 12)))
    specs = [
        ControlSpec(area=area_a, correct=["single"], detect=["double"], params={}),
        # ControlSpec(area=area_a, correct=["single"], detect=["burst==L"], params={"L": 2}),
    ]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1))
    code = builder.build(max_r=5, max_attempts_per_r=100000)
    assert code.k == k
    print(f"Achieved r = {code.r}") 
    assert code.r >= 1

    # Random data
    rnd = random.Random(123)
    data = [rnd.randint(0, 1) for _ in range(k)]
    cw = code.encode(data)
    assert len(cw) == code.n
    # No error syndrome is zero
    assert code.syndrome(cw) == 0

    # All singles in area A must be corrected
    for i in area_a.indices:
        rcv = flip_bits(cw, [i])
        corrected, ok, ev = code.decode(rcv)
        assert ok
        assert code.syndrome(corrected) == 0
        # print(f"Output of ev: {ev}")

    # All adjacent doubles in area A must be detected but not corrected.
    for i in range(min(area_a.indices), max(area_a.indices)):
        if i + 1 not in set(area_a.indices):
            continue
        rcv = flip_bits(cw, [i, i + 1])
        corrected, ok, ev = code.decode(rcv)
        
        # print(f"Error positions: {[i, i + 1]}")
        # print(f"data:      {data}")
        # print(f"rcv:       {rcv}")
        # print(f"corrected: {corrected}")
        # print(f"ok:        {ok}")
        # print(f"ev:        {ev}")
        
        
        assert not ok
        assert ev is None
        assert corrected == rcv  # No correction
        

    # All adjacent random errors in area A must not be detected and not be corrected.
    for i in range(min(area_a.indices), max(area_a.indices)):

        bit_error_offset = random.randint(1, code.n - 1)

        if i + bit_error_offset not in set(area_a.indices):
            continue
        rcv = flip_bits(cw, [i, i + bit_error_offset])
        corrected, ok, ev = code.decode(rcv)
        # assert ok
        # assert ev is not None
        print(f"Error positions: {[i, i + bit_error_offset]}")
        print(f"data:      {data}")
        print(f"rcv:       {rcv}")
        print(f"corrected: {corrected}")
        print(f"ok: {ok}")
        print(f"Output of ev: {ev}")
        assert not ok
        assert ev is None
        assert corrected == rcv  # No correction

        # assert corrected == rcv  # No correction
