import random
import os
from fuec_encoder_decoder import Area, ControlSpec, FUECBuilder, FUECCode

def build_sample_code():
    k = 8
    r = 6
    area = Area("A", tuple(range(k + r)))
    spec = ControlSpec(area=area, correct=["single"], detect=[], params=None)
    builder = FUECBuilder(k=k, specs=[spec], rng=random.Random(42))
    code = builder.build(min_r=r, max_r=r, max_attempts_per_r=2000)
    return code


def test_save_load_roundtrip(tmp_path):
    code = build_sample_code()
    path = tmp_path / "code.json"
    code.save_json(str(path))
    assert path.exists()

    loaded = FUECCode.load_json(str(path))
    # Basic structural equality checks
    assert loaded.n == code.n
    assert loaded.k == code.k
    assert loaded.r == code.r
    assert loaded.data_positions == code.data_positions
    assert loaded.parity_positions == code.parity_positions
    assert loaded.columns == code.columns
    assert loaded.correctable_map == code.correctable_map

    # Functional encoding/decoding roundtrip
    data_bits = [1 if i % 2 else 0 for i in range(code.k)]
    cw = code.encode(data_bits)
    cw_loaded = loaded.encode(data_bits)
    assert cw == cw_loaded

    # Inject single errors and ensure both decode identically
    for i in range(code.n):
        rcv = list(cw)
        rcv[i] ^= 1
        corr1, ok1, ev1 = code.decode(rcv)
        corr2, ok2, ev2 = loaded.decode(rcv)
        assert corr1 == corr2
        assert ok1 == ok2
        assert ev1 == ev2


def test_encode_int_decode_to_int(tmp_path):
    code = build_sample_code()
    for data_val in [0, 1, 0xAA >> 0, (1 << (code.k - 1)) - 1]:
        cw = code.encode_int(data_val)
        val, ok, ev = code.decode_to_int(cw)
        assert ok
        assert val == data_val
