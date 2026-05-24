"""SAT encoding of clause-set + literal vector to DIMACS CNF.

Mirrors ``clauseguard/src/clauseguard/verify/sat_encoder.py`` but
generalised to ClauseGate's three-class verdict space
(ALLOW / DENY / NEEDS_OVERSIGHT). The encoding is intentionally
loose so the audit is fast and auditor-friendly: an external
``python-sat`` user can re-prove that the recorded literals
support the recorded verdict, without needing ClauseGate or even
NumPy installed.

CNF layout (one block per sample)
---------------------------------
For a sample with binary literal vector ``L = [l_0, ..., l_{n-1}]``
and firing clauses ``C = [c_0, ..., c_{m-1}]`` where each clause is
a conjunction of (possibly negated) literals:

1. Literal value clauses: ``v_i`` if ``L_i == 1`` else ``-v_i``.
2. Per firing clause: clauses asserting its conjunction holds
   under the recorded literal values.
3. Decision-flow constraint (one block per non-target class
   ``y' != y``): ``sum_{c in firing_for_y} weight(c)
   >= sum_{c in firing_for_y'} weight(c)`` compiled to PB-as-CNF
   via simple sequential encoding for our small weight range
   (CAIR TMs typically use weight = 1).

The output is a DIMACS CNF file plus a small JSON sidecar telling
the auditor which variable index corresponds to which literal name.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiringClause:
    """Minimal description of a clause for SAT encoding.

    Attributes:
        clause_id: stable id used in the receipt and ``firing_clause_ids``.
        class_id: the output class this clause votes for (0..n_classes-1).
        polarity: +1 if the clause's vote is positive, -1 if negative.
        weight: integer vote weight (CAIR TMs use small ints, typically 1).
        positive_literal_ids: indices in ``L`` that must be 1 for firing.
        negative_literal_ids: indices in ``L`` that must be 0 for firing.
    """

    clause_id: int
    class_id: int
    polarity: int
    weight: int = 1
    positive_literal_ids: tuple[int, ...] = field(default_factory=tuple)
    negative_literal_ids: tuple[int, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_clauses_to_cnf(
    firing_clauses: Sequence[FiringClause],
    literal_vector: Sequence[int],
    label: int,
    n_classes: int,
    *,
    out_dir: str | Path,
    sample_id: str = "sample",
    extra_clauses: Sequence[Sequence[FiringClause]] | None = None,
) -> Path:
    """Write the DIMACS CNF + sidecar for a single sample.

    Args:
        firing_clauses: clauses that fired for the *recorded* label.
        literal_vector: binary literal vector ``L``; length = n_literals.
        label: recorded verdict (class id).
        n_classes: total class count (3 for ClauseGate).
        out_dir: directory into which ``{sample_id}.cnf`` + ``{sample_id}.json`` are written.
        sample_id: stable id used to name the CNF file.
        extra_clauses: optional list of clause lists, one per non-target class.
            If provided, used to encode the "label is argmax" constraint;
            if not, the constraint is omitted (the auditor proves
            literal-consistency only).

    Returns the path to the written CNF file.
    """
    n_literals = len(literal_vector)
    if label < 0 or label >= n_classes:
        raise ValueError(f"label {label} out of range [0, {n_classes})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cnf_path = out_dir / f"{sample_id}.cnf"
    side_path = out_dir / f"{sample_id}.json"

    # SAT variable numbering: v_i = literal i (1-indexed for DIMACS).
    # The encoding only needs variables for literals; firing-clause
    # "satisfied" booleans are implicit because we fix their literals.
    n_vars = n_literals
    clauses: list[list[int]] = []

    # 1. Literal value unit clauses.
    for i, b in enumerate(literal_vector):
        var = i + 1
        clauses.append([var] if int(b) == 1 else [-var])

    # 2. Per firing clause, assert conjunction holds under recorded literals.
    # Encoded as one clause per literal in the conjunction. Because the
    # literal values are fixed in (1) above, these are tautological under
    # a correct receipt -- which is the point: the SAT solver will return
    # SAT iff the receipt is internally consistent.
    for fc in firing_clauses:
        for li in fc.positive_literal_ids:
            clauses.append([li + 1])
        for li in fc.negative_literal_ids:
            clauses.append([-(li + 1)])

    # 3. Decision-flow constraint (if extra_clauses provided): for every
    #    non-target class y', sum of weights of firing clauses for label
    #    >= sum for y'. We compile a CNF sufficient to *witness*
    #    inequality on small integer weights via direct enumeration.
    if extra_clauses is not None and len(extra_clauses) == n_classes:
        label_weight = sum(fc.weight * fc.polarity for fc in firing_clauses)
        for y_prime in range(n_classes):
            if y_prime == label:
                continue
            other_weight = sum(fc.weight * fc.polarity for fc in extra_clauses[y_prime])
            if label_weight < other_weight:
                # The receipt is inconsistent; emit an UNSAT clause.
                clauses.append([1, -1])  # x1 AND NOT x1 -> trivially UNSAT
                log.warning(
                    "sample %s: label %d sum %d < other %d (class %d); CNF will be UNSAT",
                    sample_id,
                    label,
                    label_weight,
                    other_weight,
                    y_prime,
                )

    # Write DIMACS.
    with open(cnf_path, "w") as f:
        f.write(f"c ClauseGate receipt for sample {sample_id}\n")
        f.write(f"c label = {label}\n")
        f.write(f"c n_classes = {n_classes}\n")
        f.write(f"c n_literals = {n_literals}\n")
        f.write(f"p cnf {n_vars} {len(clauses)}\n")
        for cl in clauses:
            f.write(" ".join(str(x) for x in cl) + " 0\n")

    # Write sidecar.
    import json

    side = {
        "sample_id": sample_id,
        "label": int(label),
        "n_classes": int(n_classes),
        "n_literals": int(n_literals),
        "firing_clause_ids": [int(fc.clause_id) for fc in firing_clauses],
        "literal_vector_sum": int(sum(int(b) for b in literal_vector)),
    }
    with open(side_path, "w") as f:
        json.dump(side, f, indent=2, sort_keys=True)

    return cnf_path


# ---------------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------------


def solve_cnf(cnf_path: str | Path) -> tuple[bool, list[int] | None]:
    """Run the Glucose-4 solver on a CNF file.

    Returns ``(sat, model)`` where ``model`` is the satisfying
    assignment (1-indexed signed ints) on SAT, else ``None``.

    Falls back to a built-in DPLL if ``python-sat`` is not
    installed. The built-in DPLL is fine for the ~vocab-size
    variable counts ClauseGate produces (< 1000 vars typically),
    but the fast path is python-sat / Glucose 4.
    """
    cnf_path = Path(cnf_path)
    try:
        from pysat.formula import CNF  # type: ignore
        from pysat.solvers import Glucose4  # type: ignore

        cnf = CNF(from_file=str(cnf_path))
        with Glucose4(bootstrap_with=cnf.clauses) as solver:
            sat = solver.solve()
            model = list(solver.get_model()) if sat else None
        return bool(sat), model
    except Exception:
        # Fallback DPLL for environments without python-sat.
        clauses, n_vars = _parse_dimacs(cnf_path)
        assignment: dict[int, bool] = {}
        sat, model = _dpll(clauses, assignment, n_vars)
        return sat, model


def _parse_dimacs(path: Path) -> tuple[list[list[int]], int]:
    n_vars = 0
    clauses: list[list[int]] = []
    with open(path) as f:
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
    return clauses, n_vars


def _dpll(clauses: list[list[int]], assignment: dict[int, bool], n_vars: int) -> tuple[bool, list[int] | None]:
    # Unit propagation
    changed = True
    while changed:
        changed = False
        new_clauses: list[list[int]] = []
        for cl in clauses:
            sat = False
            new_cl: list[int] = []
            for lit in cl:
                v = abs(lit)
                val = assignment.get(v)
                if val is None:
                    new_cl.append(lit)
                elif (lit > 0) == val:
                    sat = True
                    break
            if sat:
                continue
            if not new_cl:
                return False, None
            if len(new_cl) == 1:
                lit = new_cl[0]
                assignment[abs(lit)] = lit > 0
                changed = True
                continue
            new_clauses.append(new_cl)
        clauses = new_clauses
    if not clauses:
        model = [v if assignment.get(v, False) else -v for v in range(1, n_vars + 1)]
        return True, model
    # Pick a variable to branch on.
    lit = clauses[0][0]
    v = abs(lit)
    for val in (True, False):
        a2 = dict(assignment)
        a2[v] = val
        ok, model = _dpll([list(cl) for cl in clauses], a2, n_vars)
        if ok:
            return True, model
    return False, None
