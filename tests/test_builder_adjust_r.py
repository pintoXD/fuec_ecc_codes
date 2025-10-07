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
    # Ensure the starting redundancy actually used (r_start) was >= required threshold.
    # Builder log prints: '... r_start=<value> ...'. Accept either an explicit adjustment line
    # or simply the presence of r_start with adequate value.
    import re
    m = re.search(r"r_start=(\d+)", out)
    assert m, f"Did not find r_start in builder log: {out}"
    used_r_start = int(m.group(1))
    assert used_r_start >= 4
