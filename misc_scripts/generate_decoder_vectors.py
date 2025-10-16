#!/usr/bin/env python3
"""Generate exhaustive test vectors for the generated fuec_decoder_48_32.v module.

Parses the decoder Verilog file for lines containing syndrome selector comments of the form:
    // e=(i, j, ...)
and extracts each error pattern. For each pattern it builds a bitmask (48 bits) with ones at the
error positions. It then emits a SystemVerilog testbench that:

  - Instantiates the decoder DUT.
  - Applies each error mask as an input codeword r (assuming original codeword is all zeros so
    the error pattern equals the received word).
  - Checks that for non-zero patterns: corrected=1, uncorrectable=0, r_fix == 0.
  - Checks that for the zero vector: no_error=1, corrected=0, uncorrectable=0, r_fix == 0.

This assumes every listed pattern is correctable (i.e. appears in the correctable_map) and that
the underlying code is systematic with all-zero data producing an all-zero codeword (true for
the current generator because parity of all-zero data is zero).

Outputs:
  sv_testbenchs/tb_fuec_decoder_48_32_vectors.sv
"""
from __future__ import annotations
import re
from pathlib import Path

DECODER_PATH = Path("verilog_out/fuec_decoder_48_32.v")
OUT_TB_PATH = Path("sv_testbenchs/tb_fuec_decoder_48_32_vectors.sv")
N = 48
R = 16  # hard-coded for naming clarity; not strictly needed beyond cosmetics

pattern_re = re.compile(r"//\s*e=\(([^)]*)\)")

def parse_patterns() -> list[tuple[int, ...]]:
    if not DECODER_PATH.exists():
        raise SystemExit(f"Decoder file not found: {DECODER_PATH}")
    patterns: set[tuple[int, ...]] = set()
    with DECODER_PATH.open() as f:
        for line in f:
            m = pattern_re.search(line)
            if not m:
                continue
            content = m.group(1).strip()
            if not content:
                patterns.add(())
                continue
            idxs = [int(x.strip()) for x in content.split(',') if x.strip()]
            patterns.add(tuple(sorted(idxs)))
    # Sort deterministic: by length then lexicographically
    return sorted(patterns, key=lambda ev: (len(ev), ev))

def mask_from_pattern(ev: tuple[int, ...]) -> int:
    m = 0
    for i in ev:
        if i < 0 or i >= N:
            raise ValueError(f"Index {i} out of range for N={N}")
        m |= (1 << i)
    return m

def to_hex(width: int, value: int) -> str:
    hex_digits = (width + 3) // 4
    return f"{width}'h{value:0{hex_digits}x}"  # lower-case hex

def generate_tb(patterns: list[tuple[int, ...]]) -> str:
    lines: list[str] = []
    lines.append("`timescale 1ns/1ps")
    lines.append("// Auto-generated exhaustive decoder testbench")
    # Module header
    lines.append("module tb_fuec_decoder_48_32_vectors;")
    lines.append(f"  localparam int N = {N};")
    lines.append(f"  localparam int R = {R};")
    lines.append(f"  localparam int NUM_PATTERNS = {len(patterns)};")
    lines.append("")
    lines.append("  // DUT signals")
    lines.append("  reg  [N-1:0] r;")
    lines.append("  wire [R-1:0] s;")
    lines.append("  wire [N-1:0] r_fix;")
    lines.append("  wire no_error, corrected, uncorrectable;")
    lines.append("")
    lines.append("  fuec_decoder_48_32 dut(.r(r), .s(s), .r_fix(r_fix), .no_error(no_error), .corrected(corrected), .uncorrectable(uncorrectable));")
    lines.append("")
    lines.append("  // Patterns (bitmasks) derived from selector comments in decoder")
    # patterns array sized exactly to NUM_PATTERNS entries
    lines.append(f"  logic [N-1:0] patterns [0:NUM_PATTERNS-1];")
    lines.append("  initial begin")
    for idx, ev in enumerate(patterns):
        mask = mask_from_pattern(ev)
        ev_str = ",".join(str(i) for i in ev) if ev else "(empty)"
        lines.append(f"    patterns[{idx}] = {to_hex(N, mask)}; // e={ev_str}")
    lines.append("  end")
    lines.append("")
    lines.append("  integer i; integer errors = 0;")
    lines.append("  task check_pattern(input [N-1:0] mask);")
    lines.append("    begin")
    lines.append("      r = mask; #1;  // combinational settle")
    lines.append("      if (mask == '0) begin")
    lines.append("        if (!(no_error && !corrected && !uncorrectable && r_fix == '0)) begin")
    lines.append("          $display(\"[FAIL][zero] r=%b no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%b\", r, no_error, corrected, uncorrectable, r_fix);")
    lines.append("          errors = errors + 1;")
    lines.append("        end")
    lines.append("      end else begin")
    lines.append("        // Expect correction: corrected=1, r_fix back to zero, no_error=0")
    lines.append("        if (!(corrected && !no_error && !uncorrectable && r_fix == '0)) begin")
    lines.append("          $display(\"[FAIL][mask=%b] no_error=%0d corrected=%0d uncorrectable=%0d r_fix=%b\", mask, no_error, corrected, uncorrectable, r_fix);")
    lines.append("          errors = errors + 1;")
    lines.append("        end")
    lines.append("      end")
    lines.append("    end")
    lines.append("  endtask")
    lines.append("")
    lines.append("  initial begin")
    lines.append("    // Zero vector sanity")
    lines.append("    check_pattern('0);")
    lines.append("    // All listed correctable patterns")
    lines.append("    for (i = 0; i < NUM_PATTERNS; i = i + 1) begin")
    lines.append("      check_pattern(patterns[i]);")
    lines.append("    end")
    lines.append("    if (errors == 0) $display(\"All %0d patterns PASSED.\", NUM_PATTERNS); else $display(\"%0d patterns FAILED.\", errors);")
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"

def main():
    patterns = parse_patterns()
    print(f"Parsed {len(patterns)} patterns from {DECODER_PATH}")
    tb = generate_tb(patterns)
    OUT_TB_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_TB_PATH.write_text(tb)
    print(f"Wrote testbench: {OUT_TB_PATH}")

if __name__ == "__main__":
    main()
