# FUEC (Flexible Unequal Error Control) – Python Implementation

This repo includes a small, practical implementation of FUEC codes: a generator that builds a parity-check matrix H satisfying different error control levels in distinct codeword areas, plus a fast encoder/decoder and unit tests.

Highlights:
- Build codes with multiple areas and per-area control like: correct singles, correct double-adjacent, correct bursts up to L, detect-only, etc.
- Systematic layout: data bits followed by r parity bits by default; encoder is a few XORs.
- Decoder uses syndrome lookup; non-matching non-zero syndromes are detected but not corrected.
- Searches for minimal redundancy r (within bounds) that satisfies the constraints.

## Files
- `fuec_encoder_decoder.py` – FUEC builder, `FUECCode` encoder/decoder, and a small CLI demo.
- `tests/test_fuec.py` – Unit tests covering build, encode/decode, and different specs.

## Quick start

- Run the example demo (uses the existing virtualenv):

```zsh
/home/otto/Documentos/ims_bordeaux/source_codes/fuec_code_002/fuec_002_env/bin/python fuec_encoder_decoder.py --example
```

- Run tests:

```zsh
/home/otto/Documentos/ims_bordeaux/source_codes/fuec_code_002/fuec_002_env/bin/python -m pytest -q
```

## API sketch

- Define areas and specs:

```python
from fuec_encoder_decoder import Area, ControlSpec, FUECBuilder

area_a = Area("A", tuple(range(0, 8)))      # absolute bit positions in the codeword
area_b = Area("B", tuple(range(8, 16)))

specs = [
    ControlSpec(area=area_a, correct=["single", "double_adjacent"], detect=[]),
    ControlSpec(area=area_b, correct=[], detect=["single"]),
]

builder = FUECBuilder(k=16, specs=specs)
code = builder.build(max_r=12, max_attempts_per_r=5000)

# Encode data bits (LSB-first order per index)
data = [0,1,1,0, 1,0,0,1, 1,1,0,0, 1,0,1,0]
cw = code.encode(data)

# Decode and correct
decoded, ok, ev = code.decode(cw)
```

Supported control models per area:
- `"single"` – correct all single random errors in the area
- `"double"` – correct all random double errors in the area (use with care; combinatorial growth)
- `"double_adjacent"` – correct all adjacent pairs (i, i+1) within the area
- `"burst<=L"` – correct any contiguous burst with length 2..L; pass `params={"L": L}`
- `"burst==L"` – correct any contiguous burst with exact length L; pass `params={"L": L}`
- Detect-only errors are listed in `detect=[...]` and won’t be miscorrected.

## Redundancy optimization

The builder computes a lower bound r ≥ ceil(log2(1 + |E_plus|)) and then searches from that r upward until a valid code is found (within `max_r`). Increase `max_attempts_per_r` for tighter specs.

Notes:
- Single-correction over D data positions plus detect-only on parity bits typically needs r ≥ ceil(log2(1 + D)).
- Adding double-adjacent or bursts increases |E_plus| accordingly.

## Limitations and tips
- Large random-double or long-burst specs grow fast; keep areas small or reduce scope.
- This is a constructive randomized search. For hard specs, try a different RNG seed or raise `max_attempts_per_r`/`max_r`.
- Data/parity positions default to systematic `[0..k-1]` for data and `[k..k+r-1]` for parity. Custom layouts are possible via the builder.

## License
This code is provided for research and prototyping. Adapt as needed for production use.
