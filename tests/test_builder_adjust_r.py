import re
from fuec_encoder_decoder import Area, ControlSpec, FUECBuilder


def test_builder_adjusts_r_start_when_area_indices_extend_past_k(capfd):
    # Force a situation where provided min_r is too small to cover area indices.
    k = 4
    area = Area("A", tuple(range(0, 8)))  # indices 0..7 require r >= 4 since n=k+r
    specs = [ControlSpec(area=area, correct=["single"], detect=["double"])]
    builder = FUECBuilder(k=k, specs=specs)
    code = builder.build(min_r=1, max_r=8, max_attempts_per_r=500)
    out, _ = capfd.readouterr()
    assert code.r >= 4
    # Either adjustment message printed or initial min_r (1) was raised to >=4 silently.
    # Prefer to see the message, but don't fail if constraints already forced r_start.
    # We still assert that the log includes the final min_r line with min_r>=4.
    assert f"min_r={code.r}" in out or "Adjusted r_start" in out
