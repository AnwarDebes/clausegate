"""Greedy minimum-edit counterfactual search.

This is the ClauseGate analogue of
``graphtm-cbr/graphtm/recourse/search.py``: a budgeted greedy descent
over the clause-attributed candidate set. We never enumerate beyond
``candidates_from_clauses``; at each step we pick the candidate that
moves the prediction furthest toward the target verdict, apply it,
re-encode, re-predict, and stop the moment the verdict flips (or we
exhaust the edit budget).

Why not BFS or 2^k enumeration: the same reason as graphtm-cbr --
this code runs inline on every blocked agent action and the sub-50 ms
P95 budget rules out any tree search. Greedy + clause-attribution is
the published-elsewhere pattern for clause-driven recourse.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..schema.actions import AgentAction, Policy, Verdict
from .candidates import ArgumentEdit, apply_edit, candidates_from_clauses


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class RecourseOutcome:
    """The output of ``greedy_argument_edit``.

    Attributes:
        found: True if the verdict flipped to the target within budget.
        edits: the sequence of edits applied, in order.
        final_verdict: verdict after applying ``edits``.
        final_action: AgentAction after applying ``edits``.
        final_policy: Policy after applying ``edits``.
        steps: number of greedy iterations.
        latency_ms: total wall-clock latency in milliseconds.
        candidates_tried: total candidates evaluated across all steps.
        score_trace: list of (verdict, scores) tuples per step.
        reason: human-readable explanation of why we stopped.
    """

    found: bool
    edits: tuple[ArgumentEdit, ...]
    final_verdict: int
    final_action: AgentAction
    final_policy: Policy
    steps: int = 0
    latency_ms: float = 0.0
    candidates_tried: int = 0
    score_trace: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# greedy_argument_edit
# ---------------------------------------------------------------------------


def greedy_argument_edit(
    *,
    action: AgentAction,
    policy: Policy,
    verifier: Any,  # ``TriGraphHGTM`` -- typed as Any to dodge circular import
    target_verdict: int,
    max_edits: int = 3,
    max_candidates: int = 64,
    max_latency_ms: int = 50,
    allow_lift_policy: bool = True,
) -> RecourseOutcome:
    """Greedy minimum-edit search to flip the verdict to ``target_verdict``.

    Args:
        action: the proposed tool call (already canonicalised).
        policy: the operator's policy at decision time.
        verifier: a fitted ``TriGraphHGTM``.
        target_verdict: ``int(Verdict.ALLOW)``, ``Verdict.DENY``, or
            ``Verdict.NEEDS_OVERSIGHT``.
        max_edits: hard upper bound on edits to apply.
        max_candidates: how many candidates the generator may emit.
        max_latency_ms: stop early if the search would exceed this.
        allow_lift_policy: if False, drop ``lift_policy`` candidates
            (policy authorship is often outside the verifier's mandate).

    Returns a ``RecourseOutcome``. ``found = True`` iff the verdict
    flipped to ``target_verdict`` within ``max_edits`` edits.
    """
    t0 = time.perf_counter()
    cur_action = action
    cur_policy = policy
    edits_applied: list[ArgumentEdit] = []
    score_trace: list[tuple[int, tuple[int, ...]]] = []
    candidates_total = 0
    reason = ""

    # Initial prediction.
    pred = verifier.predict_one(cur_action, cur_policy)
    score_trace.append((pred.verdict, pred.class_scores))

    if pred.verdict == target_verdict:
        return RecourseOutcome(
            found=True,
            edits=(),
            final_verdict=pred.verdict,
            final_action=cur_action,
            final_policy=cur_policy,
            steps=0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            candidates_tried=0,
            score_trace=score_trace,
            reason="already at target verdict",
        )

    for step in range(max_edits):
        if (time.perf_counter() - t0) * 1000.0 >= max_latency_ms:
            reason = "latency budget exceeded"
            break

        candidates = candidates_from_clauses(
            cur_action,
            cur_policy,
            pred.firing_clauses,
            pred.features.symbol_labels,
            max_candidates=max_candidates,
        )
        if not allow_lift_policy:
            candidates = [c for c in candidates if c.op != "lift_policy"]
        if not candidates:
            reason = "no candidates available"
            break

        best_edit: ArgumentEdit | None = None
        best_score = -10**9
        best_pred = None
        for cand in candidates:
            candidates_total += 1
            new_action, new_policy = apply_edit(cur_action, cur_policy, cand)
            cand_pred = verifier.predict_one(new_action, new_policy)
            # Score = how close we got to target_verdict.
            # Use signed margin into target class.
            target_score = int(cand_pred.class_scores[target_verdict])
            other = max(
                int(cand_pred.class_scores[c])
                for c in range(len(cand_pred.class_scores))
                if c != target_verdict
            )
            improvement = target_score - other
            if cand_pred.verdict == target_verdict:
                # Strong preference for verdict-flipping moves; bias
                # by negative #edits so we minimise total edits.
                improvement += 10_000
            if improvement > best_score:
                best_score = improvement
                best_edit = cand
                best_pred = cand_pred

        if best_edit is None or best_pred is None:
            reason = "no improving candidate"
            break

        edits_applied.append(best_edit)
        cur_action, cur_policy = apply_edit(cur_action, cur_policy, best_edit)
        pred = best_pred
        score_trace.append((pred.verdict, pred.class_scores))

        if pred.verdict == target_verdict:
            return RecourseOutcome(
                found=True,
                edits=tuple(edits_applied),
                final_verdict=pred.verdict,
                final_action=cur_action,
                final_policy=cur_policy,
                steps=step + 1,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                candidates_tried=candidates_total,
                score_trace=score_trace,
                reason="verdict flipped",
            )

    if not reason:
        reason = f"verdict did not flip after {len(edits_applied)} edits"
    return RecourseOutcome(
        found=False,
        edits=tuple(edits_applied),
        final_verdict=pred.verdict,
        final_action=cur_action,
        final_policy=cur_policy,
        steps=len(edits_applied),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        candidates_tried=candidates_total,
        score_trace=score_trace,
        reason=reason,
    )
