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

    def encode(self, data_bits: Sequence[int]) -> List[int]:
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

    # --- H export helpers ---
    def H_to_csv(self, path: str, include_header: bool = True, delimiter: str = ",") -> None:
        rows = self.H_matrix()
        with open(path, "w", newline="") as f:
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

    def _enumerate_required_sets(self, n: int, parity_positions: Sequence[int]) -> Tuple[Set[ErrorPattern], Set[ErrorPattern]]:
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
        for attempt in range(max_attempts):
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
    
    # k = 12
    # r_bits = 6
    # area_a = Area("A", tuple(range(0, 4)))
    # area_b = Area("B", tuple(range(4, 8)))
    # area_c = Area("C", tuple(range(8, 16)))
    # specs = [
    #     ControlSpec(
    #         area=area_a,
    #         correct=["single", "double_adjacent"],
    #         detect=["burst==L", "double"],
    #         params={"L": 3},
    #     ),
    #     ControlSpec(
    #         area=area_b,
    #         correct=["single", "double_adjacent"],
    #         detect=[]
    #     ),
    #     ControlSpec(
    #         area=area_c,
    #         correct=["single"],
    #         detect=["double_adjacent"]
    #     ),
    # ]
    
    k = 8
    r_bits = 4
    area_a = Area("A", tuple(range(0, 8)))
    specs = [
        ControlSpec(
            area=area_a,
            correct=["single"],
            detect=["double_adjacent", "burst<=L"],
            params={"L": 3},
        ),
    ]
    builder = FUECBuilder(k=k, specs=specs, rng=random.Random(1234))

    return builder.build(max_r=r_bits, max_attempts_per_r=100000)

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
    args = parser.parse_args()

    if args.example:
        code = make_example_code()
        print(f"Built example code: n={code.n}, k={code.k}, r={code.r}")
        if args.print_H:
            print(f"H matrix (rows r0..r{code.r-1}, columns 0..{code.n-1}):")
            print(code.H_as_str())
        if args.dump_H_csv:
            code.H_to_csv(args.dump_H_csv)
            print(f"Wrote H to CSV: {args.dump_H_csv}")
        if args.dump_H_npy:
            code.H_to_npy(args.dump_H_npy)
            print(f"Wrote H to NPY: {args.dump_H_npy}")
        data = [random.randint(0, 1) for _ in range(code.k)]
        cw = code.encode(data)
        print("Data:", data)
        print("Codeword:", cw)
        # Inject an error in area A (single)
        err_pos = 3
        rcv = list(cw)
        rcv[err_pos] ^= 1
        corrected, ok, ev = code.decode(rcv)
        print(f"Injected single at {err_pos}; decoded ok={ok}, ev={ev}")
        # Inject a double-adjacent error in area A
        rcv2 = list(cw)
        rcv2[5] ^= 1
        rcv2[6] ^= 1
        corrected2, ok2, ev2 = code.decode(rcv2)
        print(f"Injected adj(5,6); decoded ok={ok2}, ev={ev2}")
        # Inject a single error in area B (detect only)
        rcv3 = list(cw)
        rcv3[7] ^= 1
        corrected3, ok3, ev3 = code.decode(rcv3)
        print(f"Injected single at 7; decoded ok={ok3}, ev={ev3}")
    else:
        print("Run with --example for a quick demo.")
        # Provide a quick demo path to enable testing exports without heavy search
        code = make_quick_code()
        print(f"Built quick demo code: n={code.n}, k={code.k}, r={code.r}")
        if args.print_H:
            print(f"H matrix (rows r0..r{code.r-1}, columns 0..{code.n-1}):")
            print(code.H_as_str())
        if hasattr(args, "dump_H_csv") and args.dump_H_csv:
            code.H_to_csv(args.dump_H_csv)
            print(f"Wrote H to CSV: {args.dump_H_csv}")
        if hasattr(args, "dump_H_npy") and args.dump_H_npy:
            code.H_to_npy(args.dump_H_npy)
            print(f"Wrote H to NPY: {args.dump_H_npy}")


if __name__ == "__main__":
    main_cli()
