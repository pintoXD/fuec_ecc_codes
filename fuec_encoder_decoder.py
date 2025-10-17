"""
FUEC (Flexible Unequal Error Control) code generator, encoder and decoder.

Summary
-------
This module builds linear block codes (binary) in a FUEC fashion: you can specify
different control areas of the codeword and demand distinct error control levels
for each: e.g., correct all single errors in area A, correct all 2-adjacent errors
in area B, detect singles in area C, or correct all bursts up to length L within an
area. The generator will search for a parity-check matrix H that satisfies the
syndrome-uniqueness constraints for the specified correctable error set while
ensuring detection (no miscorrection) for the required detectable set.

The design uses a systematic form with H = [H_d | I_r]. Encoding is p = H_d * d^T.
Decoding uses a lookup table mapping syndromes to correctable error patterns; all
non-zero syndromes not in the table are reported as detected-but-uncorrectable.

Notes and scope
---------------
- This implementation targets practical, small-to-medium n, and common control
  requirements: single-error correction, double-adjacent correction, burst<=L
  correction, and detection of singles. General t-random correction is supported
  for t=2 (random doubles) but beware of combinatorial growth.
- The builder performs randomized search over data-column labels of H for a
  chosen number of redundancy bits r and increases r as needed to satisfy the
  constraints. For tough specs, increase max_attempts per r.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import random
import csv


BitIndex = int
ErrorPattern = Tuple[BitIndex, ...]  # tuple of bit indices that are in error


@dataclass(frozen=True)
class Area:
    name: str
    indices: Tuple[int, ...]  # absolute positions within the n-bit codeword


@dataclass
class ControlSpec:
    """Per-area control requirement.

    Fields:
      - correct: list of models to correct in this area
      - detect: list of models to detect (must not miscorrect)
    Models supported:
      'single'                  -> single random error within the area
      'double'                  -> 2 random errors within the area (all pairs)
      'double_adjacent'         -> 2 adjacent errors (i and i+1 within the area)
      'burst<=L'                -> any contiguous burst of length 2..L within the area
      'burst==L'                -> any contiguous burst of exactly length L within the area
            'span_burst<=L'           -> any error pattern whose span length is 2..L (first & last bits in span flipped; interior bits arbitrary)
            'span_burst==L'           -> any error pattern whose span length is exactly L (first & last bits flipped; interior bits arbitrary)
    Provide parameters via the params dict for the burst length L.
    """

    area: Area
    correct: List[str]
    detect: List[str]
    params: Dict[str, int] | None = None


def _as_set(seq: Sequence[int]) -> Set[int]:
    return set(int(x) for x in seq)


def _contiguous_runs(sorted_indices: List[int]) -> List[Tuple[int, int]]:
    """Given a sorted list of indices, return list of (start,end) inclusive runs where
    indices are consecutive integers.
    """
    if not sorted_indices:
        return []
    runs: List[Tuple[int, int]] = []
    s = e = sorted_indices[0]
    for x in sorted_indices[1:]:
        if x == e + 1:
            e = x
        else:
            runs.append((s, e))
            s = e = x
    runs.append((s, e))
    return runs


def enumerate_single_errors(indices: Iterable[int]) -> List[ErrorPattern]:
    return [(i,) for i in indices]


def enumerate_double_errors(indices: Iterable[int]) -> List[ErrorPattern]:
    idx = sorted(_as_set(indices))
    out: List[ErrorPattern] = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            out.append((idx[i], idx[j]))
    return out


def enumerate_double_adjacent(indices: Iterable[int]) -> List[ErrorPattern]:
    s = sorted(_as_set(indices))
    sset = set(s)
    out: List[ErrorPattern] = []
    for i in s:
        j = i + 1
        if j in sset:
            out.append((i, j))
    return out


def enumerate_bursts(indices: Iterable[int], length: int, exact: bool = False) -> List[ErrorPattern]:
    """Enumerate bursts within the area as contiguous windows on the absolute index line.
    If exact=False, generate lengths in [2, length]; else only bursts of 'length'.
    """
    s = sorted(_as_set(indices))
    runs = _contiguous_runs(s)
    out: List[ErrorPattern] = []
    lengths = [length] if exact else list(range(2, max(2, length) + 1))
    for (a, b) in runs:
        run_len = b - a + 1
        for L in lengths:
            if L > run_len:
                continue
            for start in range(a, b - L + 2):
                out.append(tuple(range(start, start + L)))
    return out


def enumerate_burst_spans(indices: Iterable[int], length: int, exact: bool = False) -> List[ErrorPattern]:
    """Enumerate burst *spans* (classic burst definition) over given indices.

    A burst of span L is any error pattern whose leftmost and rightmost erroneous bits
    are exactly L-1 positions apart (i.e. they occupy a window of length L) and both
    endpoints are in error; interior bits (if any) may be either error or not. For L=2
    this degenerates to a solid adjacent double. For L>2, all 2^(L-2) interior
    combinations are produced.

    If exact=False, include all spans with L' in [2, length]; else only spans of L.
    Returned patterns are tuples of absolute bit indices (sorted increasing).
    """
    s = sorted(_as_set(indices))
    runs = _contiguous_runs(s)
    patterns: Set[ErrorPattern] = set()
    span_lengths = [length] if exact else list(range(2, max(2, length) + 1))
    for (a, b) in runs:
        run_len = b - a + 1
        for L in span_lengths:
            if L > run_len:
                continue
            for start in range(a, b - L + 2):
                window = list(range(start, start + L))
                if L == 2:
                    patterns.add(tuple(window))  # only solid 2-burst
                    continue
                interior = window[1:-1]
                interior_count = L - 2
                # Enumerate all subsets of interior positions (0 .. 2^(L-2)-1)
                for mask in range(1 << interior_count):
                    pat = [window[0]]
                    for j, idx in enumerate(interior):
                        if mask & (1 << j):
                            pat.append(idx)
                    pat.append(window[-1])
                    patterns.add(tuple(pat))
    # Return in deterministic order: sort by length then lexicographically
    return sorted(patterns, key=lambda ev: (len(ev), ev))


def enumerate_error_patterns(spec: ControlSpec, model: str) -> List[ErrorPattern]:
    if model == "single":
        return enumerate_single_errors(spec.area.indices)
    if model == "double":
        return enumerate_double_errors(spec.area.indices)
    if model == "double_adjacent":
        return enumerate_double_adjacent(spec.area.indices)
    if model.startswith("burst"):
        if spec.params is None or "L" not in spec.params:
            raise ValueError("burst model requires params={'L': int}")
        L = int(spec.params["L"])
        if model == "burst<=L":
            return enumerate_bursts(spec.area.indices, L, exact=False)
        elif model == "burst==L":
            return enumerate_bursts(spec.area.indices, L, exact=True)
    if model.startswith("span_burst"):
        if spec.params is None or "L" not in spec.params:
            raise ValueError("span_burst model requires params={'L': int}")
        L = int(spec.params["L"])  # Span length definition
        if model == "span_burst<=L":
            return enumerate_burst_spans(spec.area.indices, L, exact=False)
        elif model == "span_burst==L":
            return enumerate_burst_spans(spec.area.indices, L, exact=True)
    raise ValueError(f"Unsupported model '{model}'")


@dataclass
class FUECCode:
    n: int
    k: int
    r: int
    data_positions: Tuple[int, ...]  # positions of data bits in codeword order
    parity_positions: Tuple[int, ...]  # positions of parity bits (systematic, default: last r)
    columns: Tuple[int, ...]  # length n, each an r-bit int column of H
    correctable_map: Dict[int, ErrorPattern]  # syndrome -> error vector (correctable)

    def encode(self, data_bits: Sequence[int], verbose: bool = False) -> List[int]:
        if len(data_bits) != self.k:
            raise ValueError(f"Expected {self.k} data bits, got {len(data_bits)}")
        # Build full codeword initialized with data bits in their positions.
        cw = [0] * self.n
        for pos, bit in zip(self.data_positions, data_bits):
            b = 1 if bit else 0
            cw[pos] = b

        # Compute parity vector p = H_d * d^T, with H_p = I_r.
        # That parity vector equals the syndrome of the current data-only word.
        s = 0
        for i, bit in enumerate(cw):
            if bit:
                s ^= self.columns[i]
        # Place parity bits: parity_positions are associated to unit vectors e_j.
        # s currently equals H * b^T for b without parity. To achieve Hb^T=0, we need to
        # set p such that H_p * p^T = s, knowing H_p = I, thus p = s.
        # Optional: print parity vector p (bits j=0..r-1)
        if verbose or getattr(self, "_verbose_encode", False):
            print("p:", [(s >> j) & 1 for j in range(self.r)])
        for j, pos in enumerate(self.parity_positions):
            cw[pos] = (s >> j) & 1
        return cw

    def syndrome(self, word: Sequence[int]) -> int:
        if len(word) != self.n:
            raise ValueError(f"Expected codeword length {self.n}, got {len(word)}")
        s = 0
        for i, bit in enumerate(word):
            if bit:
                s ^= self.columns[i]
        return s

    def decode(self, received: Sequence[int]) -> Tuple[List[int], bool, Optional[ErrorPattern]]:
        """Decode using syndrome look-up.

        Returns (corrected_codeword, is_valid, corrected_error_pattern)
          - is_valid=True if no error or corrected; False if detected-uncorrectable.
          - corrected_error_pattern is None if no change, else the applied error vector.
        """
        s = self.syndrome(received)
        if s == 0:
            return list(received), True, None
        ev = self.correctable_map.get(s)
        if ev is None:
            # detected but not correctable per the spec
            return list(received), False, None
        # Correct: flip bits at ev positions
        corrected = list(received)
        for i in ev:
            corrected[i] ^= 1
        # check
        if self.syndrome(corrected) != 0:
            # Should not happen; indicates inconsistency
            return corrected, False, ev
        return corrected, True, ev

    # --- H matrix helpers ---
    def H_matrix(self) -> List[List[int]]:
        """Return H as a list of rows (r x n), where rows are ordered by bit significance
        of the integer columns: row 0 is LSB, row r-1 is MSB of column labels.
        """
        rows: List[List[int]] = []
        for j in range(self.r):
            row = [ (col >> j) & 1 for col in self.columns ]
            rows.append(row)
        return rows

    def H_as_str(self, sep: str = " ") -> str:
        rows = self.H_matrix()
        # Dynamic column width based on max column index (so indices 0..9..99 align)
        col_w = max(1, len(str(self.n - 1)))
        # Dynamic row index width to keep r00, r01 ... or r000 if needed
        row_idx_w = max(2, len(str(self.r - 1)))
        row_label = f"r{{:0{row_idx_w}d}}: "
        header_prefix = row_label.format(0)
        header = " " * len(header_prefix) + sep.join(f"{i:>{col_w}d}" for i in range(self.n))
        lines = [header]
        for j, row in enumerate(rows):
            lines.append(row_label.format(j) + sep.join(f"{b:>{col_w}d}" for b in row))
        return "\n".join(lines)

    # --- correctable_map pretty printing as XOR equations ---
    def correctable_map_as_xor(self, per_bit: bool = True) -> str:
        """Return the correctable_map (syndrome -> error pattern) as XOR equations.

        If per_bit=True, show each syndrome bit equation: s_j = ⊕ H[j,i] for i in error pattern, with the resulting bit value.
        Else, show compact vector equation: h[i1] ⊕ h[i2] ⊕ ... = s (binary).
        """
        def bin_s(val: int) -> str:
            return format(val, f"0{self.r}b")

        lines: List[str] = []
        # sort by syndrome integer ascending for stable order
        for s in sorted(self.correctable_map.keys()):
            ev = self.correctable_map[s]
            indices = ",".join(str(i) for i in ev)
            lines.append(f"s={bin_s(s)} -> e=({indices})")
            if per_bit:
                for j in range(self.r):
                    rhs_terms = " ⊕ ".join(f"H[{j},{i}]" for i in ev) if ev else "0"
                    val = (s >> j) & 1
                    lines.append(f"  s{j} = {rhs_terms} = {val}")
            else:
                rhs_terms = " ⊕ ".join(f"h[{i}]" for i in ev) if ev else "0"
                lines.append(f"  {rhs_terms} = {bin_s(s)}")
        return "\n".join(lines)

    def logic_selectors_and_flips(self, style: str = "verilog", only_involved_bits: bool = True) -> str:
        """Generate Boolean equations for syndrome selectors and bit flips.

        - For each correctable error pattern e with constant syndrome c_e, define a selector:
            sel_e = ∏_j XNOR(s_j, c_e,j) = ∏_j (s_j) if c_e,j=1 else (¬s_j)
          (XNOR with 1 is s_j; with 0 is ¬s_j.)
        - For each bit i, the flip equation is XOR of all selectors of patterns that include i:
            flip_i = ⊕_{e: i∈e} sel_e

        style: 'verilog' for Verilog-like syntax.
        only_involved_bits: if True, emit flip_i only for bits that appear in some correctable pattern.
        """
        if style != "verilog":
            raise ValueError("Only 'verilog' style is currently supported")

        def sel_name(ev: ErrorPattern) -> str:
            if not ev:
                return "sel_e_empty"
            return "sel_e_" + "_".join(str(i) for i in ev)

        # Build mapping from bit index -> list of selectors that include it
        involved_bits: Dict[int, List[str]] = {}
        lines: List[str] = []
        lines.append("// Syndrome signals: s0..s{r-1}")
        # Emit selectors
        for s_val in sorted(self.correctable_map.keys()):
            ev = self.correctable_map[s_val]
            name = sel_name(ev)
            # For each row j, term is s_j if c bit is 1, else ~s_j
            terms: List[str] = []
            for j in range(self.r):
                bit = (s_val >> j) & 1
                terms.append(f"s{j}" if bit == 1 else f"~s{j}")
            conj = " & ".join(terms) if terms else "1'b1"
            lines.append(f"// s = {format(s_val, f'0{self.r}b')}  e = {ev}")
            lines.append(f"wire {name} = {conj};")
            for i in ev:
                involved_bits.setdefault(i, []).append(name)

        # Emit flip equations
        target_bits = sorted(involved_bits.keys()) if only_involved_bits else list(range(self.n))
        lines.append("")
        lines.append("// Flip signals per bit (XOR of matching selectors)")
        for i in target_bits:
            sels = involved_bits.get(i, [])
            if not sels:
                expr = "1'b0"
            else:
                expr = " ^ ".join(sels)
            lines.append(f"wire flip_{i} = {expr};  // corrected bit: r[{i}] ^ flip_{i}")
        return "\n".join(lines)

    # --- Persistence helpers ---
    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary describing this code.

        Fields:
          version: format version for forward compatibility.
          n,k,r,data_positions,parity_positions,columns
          correctable_map: mapping of syndrome (as decimal string) -> list of bit indices
        """
        return {
            "version": 1,
            "n": self.n,
            "k": self.k,
            "r": self.r,
            "data_positions": list(self.data_positions),
            "parity_positions": list(self.parity_positions),
            "columns": list(self.columns),
            "correctable_map": {str(s): list(ev) for s, ev in self.correctable_map.items()},
        }

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "FUECCode":
        """Reconstruct a FUECCode previously produced by to_dict()."""
        version = d.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported persisted code version: {version}")
        correctable_map_raw = d["correctable_map"]  # type: ignore[index]
        if not isinstance(correctable_map_raw, dict):
            raise TypeError("correctable_map must be a dict in persisted data")
        correctable_map: Dict[int, ErrorPattern] = {}
        for k_str, ev_list in correctable_map_raw.items():  # type: ignore[assignment]
            s_val = int(k_str)
            if not isinstance(ev_list, list):
                raise TypeError("error pattern must be a list")
            correctable_map[s_val] = tuple(int(x) for x in ev_list)
        return FUECCode(
            n=int(d["n"]),
            k=int(d["k"]),
            r=int(d["r"]),
            data_positions=tuple(int(x) for x in d["data_positions"]),
            parity_positions=tuple(int(x) for x in d["parity_positions"]),
            columns=tuple(int(x) for x in d["columns"]),
            correctable_map=correctable_map,
        )

    def save_json(self, path: str) -> None:
        """Persist this code as JSON (portable, human-readable)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load_json(path: str) -> "FUECCode":
        """Load a previously saved JSON code description."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return FUECCode.from_dict(d)

    # Convenience helpers for integer-based IO ---------------------------------
    def encode_int(self, data_value: int) -> List[int]:
        """Encode a data word supplied as integer with bit 0 = d0 (LSB of list).

        Returns codeword as list of bits length n.
        """
        bits = [ (data_value >> i) & 1 for i in range(self.k) ]
        return self.encode(bits)

    def decode_to_int(self, received: Sequence[int]) -> Tuple[int, bool, Optional[ErrorPattern]]:
        """Decode codeword bits; on success return (data_value, is_valid, error_pattern).

        data_value is reconstructed assuming data_positions order maps to d0..d{k-1} with bit i = d_i.
        """
        corrected, ok, ev = self.decode(received)
        # Extract data bits back to integer
        pos_to_didx: Dict[int, int] = {pos: di for di, pos in enumerate(self.data_positions)}
        data_bits: List[int] = [0]*self.k
        for pos in self.data_positions:
            data_bits[pos_to_didx[pos]] = corrected[pos]
        val = 0
        for i, b in enumerate(data_bits):
            if b:
                val |= (1 << i)
        return val, ok, ev

    # --- Encoder parity equations ---
    def encoder_equations(self, as_data_symbols: bool = True, symbol_prefix: str = "d") -> str:
        """Return parity equations p_j as XORs of data bits per current H and layout.

        - If as_data_symbols=True, use symbols d0..d{k-1} in data order (data_positions order).
        - Else, use codeword signals b[i] for absolute bit positions.

        Note: Parity bit j resides at absolute position parity_positions[j]. Equations consider only data bits.
        """
        rows = self.H_matrix()  # r x n
        # Map absolute position -> data index for nicer d0..d{k-1} naming
        pos_to_didx: Dict[int, int] = {pos: di for di, pos in enumerate(self.data_positions)}
        lines: List[str] = []
        for j in range(self.r):
            terms: List[str] = []
            for pos in self.data_positions:
                if rows[j][pos] == 1:
                    if as_data_symbols:
                        didx = pos_to_didx[pos]
                        terms.append(f"{symbol_prefix}{didx}")
                    else:
                        terms.append(f"b[{pos}]")
            rhs = " ^ ".join(terms) if terms else "1'b0"
            p_pos = self.parity_positions[j]
            if as_data_symbols:
                lines.append(f"// parity bit p{j} at cw[{p_pos}]")
                lines.append(f"p{j} = {rhs}")
            else:
                lines.append(f"// parity bit at cw[{p_pos}]")
                lines.append(f"b[{p_pos}] = {rhs}")
        return "\n".join(lines)

    def encoder_equations_verilog(self, mode: str = "dp", d_bus: str = "d", p_bus: str = "p", b_bus: str = "cw") -> str:
        """Return Verilog assigns for encoder parity equations.

        mode:
          - 'dp': assign p[j] = ^ of d[indexes in data order]
          - 'cw': assign cw[parity_pos] = ^ of cw[data_positions where H row is 1]
        Buses can be renamed via d_bus, p_bus, b_bus.
        """
        rows = self.H_matrix()
        pos_to_didx: Dict[int, int] = {pos: di for di, pos in enumerate(self.data_positions)}
        lines: List[str] = []
        if mode == "dp":
            for j in range(self.r):
                terms: List[str] = []
                for pos in self.data_positions:
                    if rows[j][pos] == 1:
                        terms.append(f"{d_bus}[{pos_to_didx[pos]}]")
                rhs = " ^ ".join(terms) if terms else "1'b0"
                lines.append(f"// parity bit p{j} at {b_bus}[{self.parity_positions[j]}]")
                lines.append(f"assign {p_bus}[{j}] = {rhs};")
        elif mode == "cw":
            for j in range(self.r):
                terms: List[str] = []
                for pos in self.data_positions:
                    if rows[j][pos] == 1:
                        terms.append(f"{b_bus}[{pos}]")
                rhs = " ^ ".join(terms) if terms else "1'b0"
                ppos = self.parity_positions[j]
                lines.append(f"// parity bit for row {j}")
                lines.append(f"assign {b_bus}[{ppos}] = {rhs};")
        else:
            raise ValueError("mode must be 'dp' or 'cw'")
        return "\n".join(lines)

    # --- Verilog encoder module ---
    def verilog_encoder_module(self, module_name: Optional[str] = None, d_bus: str = "d", p_bus: str = "p", cw_bus: str = "cw") -> str:
        """Generate a complete combinational Verilog encoder module using current H.

        Ports:
          - input  [K-1:0] d  (data in data order)
          - output [R-1:0] p  (parity in row order)
          - output [N-1:0] cw (full codeword with data placed at data_positions and parity at parity_positions)
        """
        K, R, N = self.k, self.r, self.n
        name = module_name or f"fuec_encoder_{N}_{K}"

        # Build assigns for p from d
        p_assigns = self.encoder_equations_verilog(mode="dp", d_bus=d_bus, p_bus=p_bus, b_bus=cw_bus).splitlines()

        # Build assigns for cw: data positions from d, parity positions from p
        cw_assigns: List[str] = []
        # Data placements
        pos_to_didx: Dict[int, int] = {pos: di for di, pos in enumerate(self.data_positions)}
        for pos in self.data_positions:
            cw_assigns.append(f"assign {cw_bus}[{pos}] = {d_bus}[{pos_to_didx[pos]}];")
        # Parity placements
        for j, ppos in enumerate(self.parity_positions):
            cw_assigns.append(f"assign {cw_bus}[{ppos}] = {p_bus}[{j}];")

        lines: List[str] = []
        lines.append("// Generated by FUECCode.verilog_encoder_module")
        lines.append(f"module {name} (")
        lines.append(f"    input  wire [{K-1}:0] {d_bus},")
        lines.append(f"    output wire [{R-1}:0] {p_bus},")
        lines.append(f"    output wire [{N-1}:0] {cw_bus}")
        lines.append(");\n")
        lines.append("    // Parity equations")
        for ln in p_assigns:
            lines.append("    " + ln)
        lines.append("")
        lines.append("    // Codeword assembly")
        for ln in cw_assigns:
            lines.append("    " + ln)
        lines.append("")
        lines.append("endmodule")
        return "\n".join(lines)

    def to_verilog_encoder(self, path: str, module_name: Optional[str] = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.verilog_encoder_module(module_name=module_name))

    def verilog_encoder_testbench(self, module_name: Optional[str] = None, tb_name: Optional[str] = None,
                                  vectors: Optional[List[List[int]]] = None) -> str:
        """Generate a simple self-checking testbench for the encoder module.

        - Embeds a set of data vectors and expected p and cw, checks outputs match.
        """
        K, R, N = self.k, self.r, self.n
        mod_name = module_name or f"fuec_encoder_{N}_{K}"
        tname = tb_name or f"tb_{mod_name}"

        # Prepare vectors: include a few deterministic cases and optional randoms
        vecs: List[List[int]] = []
        vecs.append([0]*K)
        if K > 0:
            vecs.append([1] + [0]*(K-1))
        if K > 1:
            vecs.append([0,1] + [0]*(K-2))
        # Add more randoms if not provided
        rng = random.Random(123)
        while len(vecs) < 16:
            vecs.append([rng.randint(0,1) for _ in range(K)])
        if vectors is not None:
            vecs = vectors

        # Compute expected p and cw for each vector
        p_list: List[int] = []
        cw_list: List[int] = []
        for d in vecs:
            cw = self.encode(d)
            # Extract parity bits in row order
            p_bits = [(cw[self.parity_positions[j]] & 1) for j in range(R)]
            # Pack bits lsb-first to an integer for Verilog literal
            p_val = 0
            for j, b in enumerate(p_bits):
                if b:
                    p_val |= (1 << j)
            # Pack full cw to integer (lsb = bit 0)
            cw_val = 0
            for i, b in enumerate(cw):
                if b:
                    cw_val |= (1 << i)
            p_list.append(p_val)
            cw_list.append(cw_val)

        def to_verilog_literal(width: int, value: int) -> str:
            return f"{width}'h{value:0{(width+3)//4}x}"

        lines: List[str] = []
        lines.append("`timescale 1ns/1ps")
        lines.append(f"module {tname};")
        lines.append(f"  reg  [{K-1}:0] d;")
        lines.append(f"  wire [{R-1}:0] p;")
        lines.append(f"  wire [{N-1}:0] cw;")
        lines.append("")
        lines.append(f"  {mod_name} uut(.d(d), .p(p), .cw(cw));")
        lines.append("")
        lines.append(f"  integer i; integer errors = 0;")
        lines.append(f"  reg [{K-1}:0] D [0:{len(vecs)-1}];")
        lines.append(f"  reg [{R-1}:0] Pexp [0:{len(vecs)-1}];")
        lines.append(f"  reg [{N-1}:0] CWexp [0:{len(vecs)-1}];")
        lines.append("")
        lines.append("  initial begin")
        # Initialize arrays
        for idx, dvec in enumerate(vecs):
            d_val = 0
            for j, b in enumerate(dvec):
                if b:
                    d_val |= (1 << j)
            lines.append(f"    D[{idx}] = {to_verilog_literal(K, d_val)};")
            lines.append(f"    Pexp[{idx}] = {to_verilog_literal(R, p_list[idx])};")
            lines.append(f"    CWexp[{idx}] = {to_verilog_literal(N, cw_list[idx])};")
        lines.append("    #1;")
        lines.append("    for (i = 0; i < $size(D); i = i + 1) begin")
        lines.append("      d = D[i]; #1;")
        lines.append("      if (p !== Pexp[i]) begin")
        lines.append("        $display(\"[FAIL] i=%0d d=%b p=%b exp=%b\", i, d, p, Pexp[i]); errors = errors + 1;")
        lines.append("      end")
        lines.append("      if (cw !== CWexp[i]) begin")
        lines.append("        $display(\"[FAIL] i=%0d d=%b cw=%b exp=%b\", i, d, cw, CWexp[i]); errors = errors + 1;")
        lines.append("      end")
        lines.append("    end")
        lines.append("    if (errors == 0) $display(\"All %0d tests PASSED.\", $size(D)); else $display(\"%0d tests FAILED.\", errors);")
        lines.append("    $finish;")
        lines.append("  end")
        lines.append("endmodule")
        return "\n".join(lines)

    def to_verilog_encoder_tb(self, path: str, module_name: Optional[str] = None, tb_name: Optional[str] = None,
                               vectors: Optional[List[List[int]]] = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.verilog_encoder_testbench(module_name=module_name, tb_name=tb_name, vectors=vectors))

    # --- H export helpers ---
    def H_to_csv(self, path: str, include_header: bool = True, delimiter: str = ",") -> None:
        rows = self.H_matrix()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            if include_header:
                writer.writerow([f"c{i}" for i in range(self.n)])
            for row in rows:
                writer.writerow(row)

    def H_numpy(self, dtype: Optional["np.dtype"] = None):  # type: ignore[name-defined]
        try:
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover - runtime environment
            raise RuntimeError("NumPy is required for H_numpy; please install numpy") from e
        return np.array(self.H_matrix(), dtype=dtype if dtype is not None else np.uint8)

    def H_to_npy(self, path: str, dtype: Optional["np.dtype"] = None) -> None:  # type: ignore[name-defined]
        try:
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover - runtime environment
            raise RuntimeError("NumPy is required for H_to_npy; please install numpy") from e
        arr = self.H_numpy(dtype=dtype)
        np.save(path, arr)

    def to_lut_csv(self, path: str, include_all: bool = False, include_header: bool = True, delimiter: str = ",") -> None:
        """Dump the syndrome -> flip-mask lookup table as CSV.

        Columns: syndrome_dec, syndrome_hex, syndrome_bin, flip_mask_bin, weight, indices
        - If include_all=True, emit all 2^r syndromes (non-correctable map to zero mask).
        - Else, emit only the correctable syndromes present in correctable_map.
        """
        N, R = self.n, self.r

        def flip_mask_from_ev(ev: Optional[ErrorPattern]) -> int:
            if not ev:
                return 0
            m = 0
            for i in ev:
                m |= (1 << i)
            return m

        syndromes = list(range(1 << R)) if include_all else sorted(self.correctable_map.keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=delimiter)
            if include_header:
                writer.writerow(["syndrome_dec", "syndrome_hex", "syndrome_bin", "flip_mask_bin", "weight", "indices"])
            for s_val in syndromes:
                ev = self.correctable_map.get(s_val)
                mask_val = flip_mask_from_ev(ev)
                s_hex = f"0x{s_val:0{(R+3)//4}X}"
                s_bin = format(s_val, f"0{R}b")
                mask_bin = format(mask_val, f"0{N}b")
                indices_str = "" if ev is None else ";".join(str(i) for i in ev)
                weight = 0 if ev is None else len(ev)
                writer.writerow([s_val, s_hex, s_bin, mask_bin, weight, indices_str])

    # --- Verilog export ---
    def verilog_module(self, module_name: Optional[str] = None) -> str:
        """Generate a complete combinational Verilog decoder module using current H and correctable_map.

        The module computes s = H * r^T (GF(2)), compares s to each correctable syndrome, flips the bits
        indicated by the matched error pattern, and outputs flags (no_error/corrected/uncorrectable).
        """
        N = self.n
        R = self.r
        K = self.k
        name = module_name or f"fuec_decoder_{N}_{K}"
        rows = self.H_matrix()  # r x n, row j corresponds to s[j]

        def bin_row_msb_first(row: List[int]) -> str:
            # Verilog literal MSB..LSB. Our columns are 0..N-1 left-to-right, so reverse to MSB..LSB
            return ''.join('1' if row[i] else '0' for i in reversed(range(N)))

        # Build selectors and flip mapping
        items = sorted(self.correctable_map.items(), key=lambda kv: (kv[0], kv[1]))
        selector_names: List[str] = []
        per_bit_selectors: Dict[int, List[str]] = {}

        def sel_name(ev: ErrorPattern) -> str:
            return "sel_e_" + "_".join(str(i) for i in ev) if ev else "sel_e_empty"

        # Header
        lines: List[str] = []
        lines.append("// Generated by FUECCode.verilog_module")
        lines.append(f"module {name} (")
        lines.append("    input  wire [%d:0] r," % (N - 1))
        lines.append("    output wire [%d:0] s," % (R - 1))
        lines.append("    output wire [%d:0] r_fix," % (N - 1))
        lines.append("    output wire        no_error,")
        lines.append("    output wire        corrected,")
        lines.append("    output wire        uncorrectable")
        lines.append(");\n")

        # H rows
        lines.append("    // Parity-check matrix rows HROWj (bit i true means r[i] participates in s[j])")
        for j, row in enumerate(rows):
            lines.append(f"    localparam [{N-1}:0] HROW{j} = {N}'b{bin_row_msb_first(row)};")
        lines.append("")

        # Syndrome
        lines.append("    // Syndrome computation s[j] = ^(r & HROWj)")
        for j in range(R):
            lines.append(f"    assign s[{j}] = ^(r & HROW{j});")
        lines.append("")

        # Selectors
        lines.append("    // Selectors for correctable patterns (XNOR equality: &(s ^~ SVAL))")
        for s_val, ev in items:
            sval_bin = format(s_val, f"0{R}b")
            sparam = f"SVAL_{sel_name(ev)}"
            sel = sel_name(ev)
            lines.append(f"    localparam [{R-1}:0] {sparam} = {R}'b{sval_bin};  wire {sel} = &(s ^~ {sparam});  // e={ev}")
            selector_names.append(sel)
            for i in ev:
                per_bit_selectors.setdefault(i, []).append(sel)
        lines.append("")

        # Flip wires and r_fix
        lines.append("    // Flip signals per bit and corrected output")
        for i in range(N):
            sels = per_bit_selectors.get(i, [])
            expr = " ^ ".join(sels) if sels else "1'b0"
            lines.append(f"    wire flip_{i} = {expr};")
        for i in range(N):
            lines.append(f"    assign r_fix[{i}] = r[{i}] ^ flip_{i};")
        lines.append("")

        # Status
        any_sel_expr = " | ".join(selector_names) if selector_names else "1'b0"
        lines.append("    // Status flags")
        lines.append("    assign no_error = ~(|s);")
        lines.append(f"    wire any_selector = {any_sel_expr};")
        lines.append("    assign corrected = any_selector;")
        lines.append("    assign uncorrectable = (|s) & ~any_selector;")
        lines.append("")
        lines.append("endmodule")
        return "\n".join(lines)

    def to_verilog(self, path: str, module_name: Optional[str] = None) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.verilog_module(module_name=module_name))

    # --- COE export for Vivado ROM ---
    def to_coe_file(self, path: str) -> None:
        """Generate a Vivado Memory Initialization File (.coe) for a ROM.

        The ROM maps every possible syndrome (address) to its corresponding
        n-bit correction pattern (data).
        """
        total_addresses = 1 << self.r
        lines = []
        lines.append("memory_initialization_radix=2;")
        lines.append("memory_initialization_vector=")

        for addr in range(total_addresses):
            # Default correction pattern is all zeros (for no error or uncorrectable)
            flip_pattern_val = 0

            # Check if this syndrome is in our map of correctable errors
            if addr in self.correctable_map:
                error_pattern = self.correctable_map[addr]
                # Create the integer value for the flip pattern
                for bit_index in error_pattern:
                    flip_pattern_val |= 1 << bit_index

            # Format as an n-bit binary string
            binary_string = format(flip_pattern_val, f"0{self.n}b")

            # Add comma or semicolon
            separator = "," if addr < (total_addresses - 1) else ";"
            lines.append(f"{binary_string}{separator}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def verilog_decoder_module_rom(self, module_name: Optional[str] = None, rom_ip_name: str = "ecc_rom_ip") -> str:
        """Generate a pipelined Verilog decoder module that uses a ROM for the lookup table.
        
        This module is designed to instantiate a Vivado Block Memory Generator IP.
        The ROM should be configured as a Single Port ROM with a depth of 2^r and a
        width of n, initialized with the .coe file generated by to_coe_file().
        """
        N, K, R = self.n, self.k, self.r
        name = module_name or f"fuec_decoder_{N}_{K}_rom"
        
        lines = [
            f"// Generated by FUECCode.verilog_decoder_module_rom",
            f"// Pipelined FUEC Decoder ({N}, {K}) using a Block RAM ROM",
            f"// Latency: 3 clock cycles (1 for syndrome, 1 for ROM lookup, 1 for correction)",
            f"module {name} (",
            "    input  wire         clk,",
            "    input  wire         rst,",
            f"   input  wire [{N-1}:0] r_in,",
            f"   output wire [{R-1}:0] s_out,",
            f"   output wire [{N-1}:0] r_fix_out,",
            f"   output wire [{N-1}:0] pos_error_out,",
            "    output wire         no_error_out,",
            "    output wire         corrected_out,",
            "    output wire         uncorrectable_out",
            ");",
            "",
            "    // --- Pipeline Stage 1: Syndrome Calculation ---",
            f"   reg  [{N-1}:0] r_s1;",
            f"   wire [{R-1}:0] s_wire;",
            "",
            "    // --- Pipeline Stage 2: ROM Lookup ---",
            f"   reg [{N-1}:0] r_s2;",
            f"   reg [{R-1}:0] s_s2; // Registered syndrome, used as ROM address",
            f"   wire [{N-1}:0] flip_bits_from_rom; // Output of the ROM",
            "",
            "    // --- Pipeline Stage 3: Apply Correction & Status ---",
            f"   reg [{N-1}:0] r_s3;",
            f"   reg [{N-1}:0] flip_bits_s3;",
            f"   reg [{R-1}:0] s_s3;",
            "",
            "    // Parity-check matrix rows for syndrome calculation",
        ]

        # Add H-Matrix for syndrome calculation
        for j, row in enumerate(self.H_matrix()):
            row_str = "".join(map(str, reversed(row)))
            lines.append(f"    localparam [{N-1}:0] HROW{j} = {N}'b{row_str};")
        
        lines.append("")
        for j in range(R):
            lines.append(f"    assign s_wire[{j}] = ^(r_s1 & HROW{j});")

        lines.extend([
            "",
            "    // Instantiate the Block Memory Generator IP (ROM) here.",
            "    // - Configure as a Single Port ROM",
            f"   // - Depth: {1 << R} (2^{R})",
            f"   // - Width: {N}",
            "    // - Load the .coe file generated by the python script.",
            f"   {rom_ip_name} ecc_correction_rom (",
            "        .clka(clk),",
            "        .addra(s_s2),",
            "        .douta(flip_bits_from_rom)",
            "    );",
            "",
            "    // Pipeline Register Logic",
            "    always @(posedge clk) begin",
            "        if (rst) begin",
            "            r_s1 <= '0; r_s2 <= '0; r_s3 <= '0;",
            "            s_s2 <= '0; s_s3 <= '0;",
            "            flip_bits_s3 <= '0;",
            "        end else begin",
            "            // Stage 1 -> 2",
            "            r_s1 <= r_in;",
            "            s_s2 <= s_wire;",
            "",
            "            // Stage 2 -> 3",
            "            r_s2 <= r_s1;",
            "            s_s3 <= s_s2;",
            "            flip_bits_s3 <= flip_bits_from_rom;",
            "",
            "            // Stage 3 -> Output Register (not strictly needed, but good practice)",
            "            r_s3 <= r_s2;",
            "        end",
            "    end",
            "",
            "    // Final Output Assignments (combinational from last pipeline stage)",
            "    assign s_out = s_s3;",
            "    assign pos_error_out = flip_bits_s3;",
            "    assign r_fix_out = r_s3 ^ flip_bits_s3;",
            "",
            "    // Status flags are based on the state in the final pipeline stage",
            "    assign no_error_out = (s_s3 == 0);",
            "    assign corrected_out = (|flip_bits_s3);",
            "    assign uncorrectable_out = ((|s_s3) && !(|flip_bits_s3));",
            "",
            "endmodule"
        ])
        return "\n".join(lines)


class FUECBuilder:
    def __init__(self, k: int, specs: List[ControlSpec], data_positions: Optional[Sequence[int]] = None,
                 parity_positions: Optional[Sequence[int]] = None, rng: Optional[random.Random] = None) -> None:
        self.k = int(k)
        self.specs = specs
        self.rng = rng or random.Random()
        # Codeword positions: default systematic layout [data...parity]
        if data_positions is None:
            data_positions = list(range(self.k))
        if parity_positions is None:
            # We'll decide r later; placeholder
            self._parity_positions_template = "tail"  # will be k..k+r-1
        else:
            self._parity_positions_template = tuple(parity_positions)
        self.data_positions = tuple(int(x) for x in data_positions)

    def _enumerate_required_sets(self, _n: int, parity_positions: Sequence[int]) -> Tuple[Set[ErrorPattern], Set[ErrorPattern]]:
        """Return (E_plus, Delta) sets as sets of error patterns (tuples of indices).
        Add single parity-bit errors to detection set to avoid miscorrection on parity bit flips.
        """
        E_plus: Set[ErrorPattern] = set()
        Delta: Set[ErrorPattern] = set()
        for spec in self.specs:
            for m in spec.correct:
                for ev in enumerate_error_patterns(spec, m):
                    E_plus.add(tuple(sorted(ev)))
            for m in spec.detect:
                for ev in enumerate_error_patterns(spec, m):
                    Delta.add(tuple(sorted(ev)))
        # parity single errors detected-only by default
        for p in parity_positions:
            Delta.add((p,))
        # Ensure no intersection by design intent (if user specified both, correction wins)
        Delta -= E_plus
        return E_plus, Delta

    @staticmethod
    def _min_r_required(E_plus_count: int) -> int:
        # Need at least 1 (zero syndrome) + |E_plus| distinct syndromes
        need = 1 + max(0, E_plus_count)
        r = 0
        cap = 1
        while cap < need:
            r += 1
            cap <<= 1
        return r

    def _try_build_for_r(self, r: int, max_attempts: int, parity_positions: Sequence[int]) -> Optional[FUECCode]:
        n = self.k + r if self._parity_positions_template == "tail" else max(self.data_positions + tuple(parity_positions)) + 1
        # Assemble parity positions (systematic tail unless provided)
        if self._parity_positions_template == "tail":
            parity_positions = tuple(range(self.k, self.k + r))
        else:
            parity_positions = tuple(int(x) for x in parity_positions)
        n = max(list(self.data_positions) + list(parity_positions)) + 1

        # Precompute required sets
        E_plus, Delta = self._enumerate_required_sets(n, parity_positions)

        # Columns array of length n containing r-bit labels.
        # Parity columns are the identity basis e_j.
        parity_cols = [1 << j for j in range(r)]
        parity_colset = set(parity_cols)

        data_positions = list(self.data_positions)
        # If user-provided data_positions aren't unique, raise
        if len(set(data_positions)) != len(data_positions):
            raise ValueError("data_positions must be unique")
        # Ensure columns array indexes cover all positions
        columns: List[int] = [0] * n
        for j, pos in enumerate(parity_positions):
            columns[pos] = 1 << j

        # Allowed pool for data columns: non-zero, exclude parity basis.
        # Additionally, avoid columns that equal XOR of any two parity basis? Not required.
        allowed_labels = [x for x in range(1, 1 << r) if x not in parity_colset]

        # Prepare list of error patterns for constraints
        E_plus_list = sorted(list(E_plus))
        Delta_list = sorted(list(Delta))

        def check_constraints() -> bool:
            # Build syndrome map for E_plus
            syndromes: Dict[int, ErrorPattern] = {}
            # Helper: syndrome of an error pattern
            def syn(ev: ErrorPattern) -> int:
                s = 0
                # print(f"Columns length: {len(columns)}")
                # print(f"ev length: {len(ev)}")
                for i in ev:
                    s ^= columns[i]
                return s
            # Correctable must map to non-zero and be unique and not collide with parity single syndromes
            for ev in E_plus_list:
                s = syn(ev)
                if s == 0:
                    return False
                # unique
                if s in syndromes:
                    return False
                syndromes[s] = ev
            # Detect-only must not be zero and must not collide with any correctable syndrome
            for ev in Delta_list:
                s = syn(ev)
                if s == 0:
                    return False
                if s in syndromes:
                    return False
            return True

        # Random attempts
        for _ in range(max_attempts):
            # Assign random labels to all data positions
            # We can enforce that data columns for single-correct areas are unique to ease search
            self.rng.shuffle(allowed_labels)
            # Simple draw without replacement; if more data bits than allowed labels, impossible
            if len(allowed_labels) < len(data_positions):
                return None
            for pos, label in zip(data_positions, allowed_labels):
                columns[pos] = label
            if check_constraints():
                # Build correctable map
                corr_map: Dict[int, ErrorPattern] = {}
                for ev in E_plus_list:
                    s = 0
                    for i in ev:
                        s ^= columns[i]
                    corr_map[s] = ev
                return FUECCode(
                    n=n,
                    k=self.k,
                    r=r,
                    data_positions=tuple(data_positions),
                    parity_positions=tuple(parity_positions),
                    columns=tuple(columns),
                    correctable_map=corr_map,
                )
        return None

    def build(self, min_r: Optional[int] = None, max_r: int = 16, max_attempts_per_r: int = 2000) -> FUECCode:
        # Determine parity positions for unknown template
        # Start with lower bound from |E_plus|
        # Need a provisional r to compute E_plus when parity positions are required.
        # We'll iterate r anyway; start at provided min_r or computed lower bound.
        tmp_r = 1 if min_r is None else min_r
        if self._parity_positions_template == "tail":
            parity_positions = tuple(range(self.k, self.k + tmp_r))
        else:
            parity_positions = tuple(self._parity_positions_template)  # type: ignore[arg-type]
        E_plus, _ = self._enumerate_required_sets(self.k + tmp_r, parity_positions)
        lb = self._min_r_required(len(E_plus))
        r_start = max(min_r or 0, lb)
        orig_r_start = r_start

        # Ensure r is large enough so that all user-declared area indices fit inside n.
        # For the common 'tail' parity layout, n = k + r. If any area references an
        # index >= k + r_start, we must raise r_start; otherwise enumeration of error
        # patterns will include out-of-range indices causing IndexError during column
        # access. This situation occurs when the user declares areas that include
        # (anticipated) parity bit positions while also allowing the builder to start
        # the search at too small an r.
        if self._parity_positions_template == "tail":
            max_area_index = -1
            for spec in self.specs:
                for idx in spec.area.indices:
                    if idx < 0:
                        raise ValueError(f"Area index {idx} must be non-negative")
                    if idx > max_area_index:
                        max_area_index = idx
            if max_area_index >= 0:
                required_r_for_indices = max(0, (max_area_index + 1) - self.k)
                if required_r_for_indices > r_start:
                    r_start = required_r_for_indices
                    # Emit an informational note so users understand why r increased.
                    if orig_r_start != r_start:
                        print(
                            f"[FUECBuilder] Adjusted r_start from {orig_r_start} to {r_start} to cover area index {max_area_index} (n = k + r must exceed {max_area_index})."
                        )
        # (If a custom parity_positions layout was provided we assume caller ensured
        # indices are in range of that layout.)
        print(f"FUECBuilder: k={self.k}, |E+|={len(E_plus)}, min_r={min_r}, max_r={max_r}, r_start={r_start} , data_positions={self.data_positions}, parity_positions_template={self._parity_positions_template}")
        for r in range(r_start, max_r + 1):
            code = self._try_build_for_r(r, max_attempts_per_r, parity_positions)
            if code is not None:
                return code
        raise RuntimeError(
            f"Failed to build FUEC code for k={self.k} within r <= {max_r}. Try increasing max_r or attempts."
        )


def make_example_code() -> FUECCode:
    """Convenience factory: k=16 data bits
    - Area A: bits [0..7] correct single and 2-adjacent errors
    - Area B: bits [8..15] detect singles
    Parity bits at tail. r chosen automatically.
    """
    # k = 16
    # r_bits = 12
    # area_a = Area("A", tuple(range(0, 8)))
    # area_b = Area("B", tuple(range(8, 16)))
    # specs = [
    #     ControlSpec(area=area_a, correct=["single", "double_adjacent"], detect=[]),
    #     ControlSpec(area=area_b, correct=[], detect=["single"]),
    # ]
    # builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1234))
    
    example_no = 4
    if example_no == 1:
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
    elif example_no == 2:
        k = 8
        r_bits = 5
        area_a = Area("A", tuple(range(0, 13)))
        specs = [
            ControlSpec(
                area=area_a,
                correct=["single"],
                detect=["double"],
            ),
        ]
    elif example_no == 3:
        k = 8
        r_bits = 4
        area_a = Area("A", tuple(range(0, 8)))
        specs = [
            ControlSpec(
                area=area_a,
                correct=["single"],
                detect=["double_adjacent"],
            ),
        ]
    elif example_no == 4:
        k = 32
        r_bits = 16
        area_a = Area("A", tuple(range(0, k + r_bits)))
        specs = [
            ControlSpec(
                area=area_a, correct=["single", "span_burst<=L"], detect=[], params={"L": 4}
            )
        ]
    elif example_no == 5:
        k = 32
        r_bits = 16
        area_a = Area("A", tuple(range(0, k + r_bits)))
        specs = [
            ControlSpec(
                area=area_a, correct=["single"], detect=[], params={}
            )
        ]
    else:
        k = 16
        r_bits = 12
        area_a = Area("A", tuple(range(0, 8)))
        area_b = Area("B", tuple(range(8, 16)))
        specs = [
            ControlSpec(area=area_a, correct=["single", "double_adjacent"], detect=[]),
            ControlSpec(area=area_b, correct=[], detect=["single"]),
        ]
    
    
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1209))

    return builder.build(min_r=r_bits, max_r=r_bits, max_attempts_per_r=1000000)

def make_quick_code() -> FUECCode:
    """Build a small, easy spec quickly for demos and dumps.
    - k=8, Area A: all data bits, correct singles; parity bits at tail; modest search bounds.
    """
    k = 8
    area_a = Area("A", tuple(range(0, k)))
    specs = [ControlSpec(area=area_a, correct=["single"], detect=[])]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(42))
    return builder.build(max_r=8, max_attempts_per_r=3000)


def _bits_from_int(x: int, nbits: int) -> List[int]:
    return [(x >> i) & 1 for i in range(nbits)]


def _int_from_bits(bits: Sequence[int]) -> int:
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= (1 << i)
    return v


def main_cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="FUEC code generator and demo")
    parser.add_argument("--k", type=int, default=16, help="Number of data bits")
    parser.add_argument("--example", action="store_true", help="Build example spec and demo encode/decode")
    parser.add_argument("--max_r", type=int, default=12, help="Max redundancy bits to try")
    parser.add_argument("--print-H", dest="print_H", action="store_true", help="Print the H matrix after building the code")
    parser.add_argument("--dump-H-csv", dest="dump_H_csv", type=str, default=None, help="Path to write H as CSV (rows are parity-check rows)")
    parser.add_argument("--dump-H-npy", dest="dump_H_npy", type=str, default=None, help="Path to write H as NumPy .npy (requires numpy)")
    parser.add_argument("--print-lut-xor", dest="print_lut_xor", action="store_true", help="Print the correctable_map as XOR equations")
    parser.add_argument("--print-logic", dest="print_logic", action="store_true", help="Print Boolean selector and flip equations (Verilog-style)")
    parser.add_argument("--dump-verilog", dest="dump_verilog", type=str, default=None, help="Path to write a complete Verilog decoder module (.v)")
    parser.add_argument("--print-p", dest="print_p", action="store_true", help="When encoding in demo, also print the parity vector p")
    parser.add_argument("--print-encoder-eq", dest="print_encoder_eq", action="store_true", help="Print encoder parity equations p_j in terms of d0..d{k-1}")
    parser.add_argument("--print-encoder-verilog", dest="print_encoder_verilog", action="store_true", help="Print encoder parity equations as Verilog assigns")
    parser.add_argument("--encoder-verilog-mode", dest="encoder_verilog_mode", choices=["dp","cw"], default="dp", help="dp: p[j]=^d[i]; cw: cw[pos]=^cw[pos]")
    parser.add_argument("--dump-encoder-verilog", dest="dump_encoder_verilog", type=str, default=None, help="Path to write a Verilog encoder module (.v)")
    parser.add_argument("--dump-encoder-tb", dest="dump_encoder_tb", type=str, default=None, help="Path to write a Verilog encoder testbench (.v)")
    parser.add_argument("--tb-vectors", dest="tb_vectors", type=int, default=16, help="Number of test vectors to embed in the encoder TB (approximate)")
    parser.add_argument("--dump-coe", type=str, default=None, help="Path to write the ROM initialization .coe file")
    parser.add_argument("--dump-verilog-rom", type=str, default=None, help="Path to write a pipelined, ROM-based Verilog decoder")
    parser.add_argument("--dump-lut-csv", type=str, default=None, help="Path to write the syndrome->flip LUT as CSV")
    parser.add_argument("--lut-csv-all", action="store_true", help="Include all 2^r syndromes in CSV (default: only correctable)")
    args = parser.parse_args()

    if args.example:
        code = make_example_code()
        print(f"Built example code: n={code.n}, k={code.k}, r={code.r}")
        if args.print_H:
            print(f"H matrix (rows r0..r{code.r-1}, columns 0..{code.n-1}):")
            print(code.H_as_str())
        if args.print_lut_xor:
            print("Correctable map as XOR equations:")
            print(code.correctable_map_as_xor(per_bit=True))
        if args.print_encoder_eq:
            print("Encoder parity equations (p_j in terms of d0..d{k-1}):")
            print(code.encoder_equations(as_data_symbols=True))
        if args.print_encoder_verilog:
            print("Encoder parity equations (Verilog assigns):")
            print(code.encoder_equations_verilog(mode=args.encoder_verilog_mode))
        if args.dump_H_csv:
            code.H_to_csv(args.dump_H_csv)
            print(f"Wrote H to CSV: {args.dump_H_csv}")
        if args.dump_H_npy:
            code.H_to_npy(args.dump_H_npy)
            print(f"Wrote H to NPY: {args.dump_H_npy}")
        if args.dump_verilog:
            code.to_verilog(args.dump_verilog)
            print(f"Wrote Verilog module: {args.dump_verilog}")
        if args.dump_encoder_verilog:
            code.to_verilog_encoder(args.dump_encoder_verilog)
            print(f"Wrote Verilog encoder module: {args.dump_encoder_verilog}")
        if args.dump_encoder_tb:
            # Build a set of vectors (approximate size)
            rng = random.Random(777)
            vecs = [[rng.randint(0,1) for _ in range(code.k)] for _ in range(max(1, args.tb_vectors))]
            # Ensure some edge cases included
            if vecs:
                vecs[0] = [0]*code.k
            if len(vecs) > 1:
                vecs[1] = [1] + [0]*(code.k-1)
            code.to_verilog_encoder_tb(args.dump_encoder_tb, vectors=vecs)
            print(f"Wrote Verilog encoder testbench: {args.dump_encoder_tb}")
        if args.dump_coe:
            code.to_coe_file(args.dump_coe)
            print(f"Wrote ROM initialization file to: {args.dump_coe}")
        if args.dump_verilog_rom:
            with open(args.dump_verilog_rom, "w", encoding="utf-8") as f:
                f.write(code.verilog_decoder_module_rom())
            print(f"Wrote ROM-based Verilog decoder to: {args.dump_verilog_rom}")
        if args.dump_lut_csv:
            code.to_lut_csv(args.dump_lut_csv, include_all=args.lut_csv_all)
            print(f"Wrote LUT CSV to: {args.dump_lut_csv}")
        # Demo encode/decode
        data = [random.randint(0, 1) for _ in range(code.k)]
        cw = code.encode(data, verbose=args.print_p)
        print("Data:", data)
        print("Codeword:", cw)
        # Inject an error in area A (single)
        err_pos = 3
        rcv = list(cw)
        rcv[err_pos] ^= 1
        _, ok, ev = code.decode(rcv)
        print(f"Injected single at {err_pos}; decoded ok={ok}, ev={ev}")
        # Inject a double-adjacent error in area A
        rcv2 = list(cw)
        rcv2[5] ^= 1
        rcv2[6] ^= 1
        _, ok2, ev2 = code.decode(rcv2)
        print(f"Injected adj(5,6); decoded ok={ok2}, ev={ev2}")
        # Inject a single error in area B (detect only)
        rcv3 = list(cw)
        rcv3[7] ^= 1
        _, ok3, ev3 = code.decode(rcv3)
        print(f"Injected single at 7; decoded ok={ok3}, ev={ev3}")
    else:
        print("Run with --example for a quick demo.")
        code = make_quick_code()
        print(f"Built quick demo code: n={code.n}, k={code.k}, r={code.r}")
        if args.print_H:
            print(f"H matrix (rows r0..r{code.r-1}, columns 0..{code.n-1}):")
            print(code.H_as_str())
        if args.print_lut_xor:
            print("Correctable map as XOR equations:")
            print(code.correctable_map_as_xor())
        if args.print_logic:
            print("Boolean selector and flip equations (Verilog-style):")
            print(code.logic_selectors_and_flips())
        if args.print_encoder_eq:
            print("Encoder parity equations (p_j in terms of d0..d{k-1}):")
            print(code.encoder_equations(as_data_symbols=True))
        if args.print_encoder_verilog:
            print("Encoder parity equations (Verilog assigns):")
            print(code.encoder_equations_verilog(mode=args.encoder_verilog_mode))
        if hasattr(args, "dump_H_csv") and args.dump_H_csv:
            code.H_to_csv(args.dump_H_csv)
            print(f"Wrote H to CSV: {args.dump_H_csv}")
        if hasattr(args, "dump_H_npy") and args.dump_H_npy:
            code.H_to_npy(args.dump_H_npy)
            print(f"Wrote H to NPY: {args.dump_H_npy}")
        if hasattr(args, "dump_verilog") and args.dump_verilog:
            code.to_verilog(args.dump_verilog)
            print(f"Wrote Verilog module: {args.dump_verilog}")
        if hasattr(args, "dump_encoder_verilog") and args.dump_encoder_verilog:
            code.to_verilog_encoder(args.dump_encoder_verilog)
            print(f"Wrote Verilog encoder module: {args.dump_encoder_verilog}")
        if hasattr(args, "dump_encoder_tb") and args.dump_encoder_tb:
            rng = random.Random(888)
            vecs = [[rng.randint(0,1) for _ in range(code.k)] for _ in range(max(1, getattr(args, 'tb_vectors', 16)))]
            if vecs:
                vecs[0] = [0]*code.k
            if len(vecs) > 1:
                vecs[1] = [1] + [0]*(code.k-1)
            code.to_verilog_encoder_tb(args.dump_encoder_tb, vectors=vecs)
            print(f"Wrote Verilog encoder testbench: {args.dump_encoder_tb}")
        if hasattr(args, "dump_verilog_rom") and args.dump_verilog_rom:
            with open(args.dump_verilog_rom, "w", encoding="utf-8") as f:
                f.write(code.verilog_decoder_module_rom())
            print(f"Wrote ROM-based Verilog decoder to: {args.dump_verilog_rom}")
        if hasattr(args, "dump_lut_csv") and args.dump_lut_csv:
            code.to_lut_csv(args.dump_lut_csv, include_all=args.lut_csv_all)
            print(f"Wrote LUT CSV to: {args.dump_lut_csv}")
        # If the user explicitly asked to print p, run a tiny encode to display it
        if args.print_p:
            data = [random.randint(0, 1) for _ in range(code.k)]
            cw = code.encode(data, verbose=True)
            print("Data:", data)
            print("Codeword:", cw)


if __name__ == "__main__":
    main_cli()
