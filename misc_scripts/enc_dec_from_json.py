from fuec_encoder_decoder import Area, ControlSpec, FUECBuilder, FUECCode
import random

def encode_and_persist():
    # Build once
    k, r = 32, 16
    area = Area("A", tuple(range(k + r)))
    specs = [
        ControlSpec(
            area=area, correct=["single", "span_burst<=L"], detect=[], params={"L": 4}
        )
    ]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1209))
    code = builder.build(min_r=r, max_r=r)

    # Save
    code.save_json("fuec_48_32.json")

def read_and_decode():
    # Later / another process
    loaded = FUECCode.load_json("fuec_48_32.json")

    # Encode/decode via list of bits
    data_bits = [1 if i % 3 == 0 else 0 for i in range(k)]
    cw = loaded.encode(data_bits)
    corrected, ok, ev = loaded.decode(cw)

    # Integer interface (LSB = d0)
    value = 0xDEADBEEF & ((1 << k) - 1)
    cw2 = loaded.encode_int(value)
    recovered, ok2, ev2 = loaded.decode_to_int(cw2)
    assert ok2 and recovered == value

if __name__ == "__main__":
    print("OIt")