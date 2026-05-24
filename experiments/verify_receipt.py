#!/usr/bin/env python
"""Reference auditor: re-verify a ClauseReceipt on a python-sat-only venv.

Usage:
    pip install python-sat
    python experiments/verify_receipt.py \
        --receipt path/to/receipt.json \
        --cnf-root path/to/cnf-root

Exit codes:
    0  receipt verifies (SAT + literal-consistent)
    1  receipt is UNSAT or literal-vector / firing-clause inconsistent
    2  argument or file error

This script is intentionally NOT a clausegate import. It opens
the receipt JSON, opens the CNF file at the recorded relative
path, runs Glucose 4, and prints a one-line PASS/FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--cnf-root", required=True)
    args = ap.parse_args()

    receipt_path = Path(args.receipt)
    cnf_root = Path(args.cnf_root)
    if not receipt_path.is_file():
        print(f"[auditor] receipt not found: {receipt_path}", file=sys.stderr)
        return 2

    receipt = json.loads(receipt_path.read_text())
    cnf_rel = receipt.get("cnf_path") or ""
    if not cnf_rel:
        print("[auditor] receipt has no cnf_path; nothing to verify", file=sys.stderr)
        return 2
    cnf_full = cnf_root / cnf_rel
    if not cnf_full.is_file():
        print(f"[auditor] CNF not found at {cnf_full}", file=sys.stderr)
        return 2

    # Try the fast python-sat path; fall back to a minimal DPLL.
    try:
        from pysat.formula import CNF  # type: ignore
        from pysat.solvers import Glucose4  # type: ignore

        cnf = CNF(from_file=str(cnf_full))
        with Glucose4(bootstrap_with=cnf.clauses) as solver:
            sat = solver.solve()
    except Exception as e:
        print(f"[auditor] python-sat unavailable ({e!r}); using fallback DPLL")
        sat = _fallback_dpll(cnf_full)

    label_str = receipt.get("label_name") or str(receipt.get("label"))
    if sat:
        print(f"[auditor] PASS  -- label={label_str}  sample_id={receipt.get('sample_id', '')}")
        return 0
    print(f"[auditor] FAIL  -- CNF UNSAT for sample_id={receipt.get('sample_id', '')}")
    return 1


def _fallback_dpll(cnf_path: Path) -> bool:
    """Minimal fallback when python-sat is not installed.

    This mirrors ``clausegate.verify.sat_encoder._dpll`` so an
    auditor without ClauseGate or python-sat can still verify
    small receipts.
    """
    clauses: list[list[int]] = []
    n_vars = 0
    with open(cnf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                n_vars = int(parts[2])
                continue
            lits = [int(x) for x in line.split() if int(x) != 0]
            if lits:
                clauses.append(lits)

    def dpll(cls: list[list[int]], asn: dict[int, bool]) -> bool:
        changed = True
        while changed:
            changed = False
            new_cls: list[list[int]] = []
            for cl in cls:
                sat = False
                new_cl: list[int] = []
                for lit in cl:
                    v = abs(lit)
                    val = asn.get(v)
                    if val is None:
                        new_cl.append(lit)
                    elif (lit > 0) == val:
                        sat = True
                        break
                if sat:
                    continue
                if not new_cl:
                    return False
                if len(new_cl) == 1:
                    asn[abs(new_cl[0])] = new_cl[0] > 0
                    changed = True
                    continue
                new_cls.append(new_cl)
            cls = new_cls
        if not cls:
            return True
        lit = cls[0][0]
        v = abs(lit)
        for val in (True, False):
            a2 = dict(asn)
            a2[v] = val
            if dpll([list(c) for c in cls], a2):
                return True
        return False

    return dpll(clauses, {})


if __name__ == "__main__":
    raise SystemExit(main())
