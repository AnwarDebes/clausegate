"""Canonical NumPy MultiClass Tsetlin Machine -- CPU reference.

This implementation follows Granmo (2018) and the Granmo & Saha
HierarchicalTM derivation. It is the numerical reference the CUDA
path is parity-tested against; correctness, not throughput, is the
goal.

API:
    MultiClassTM(n_classes, n_clauses, T, s, n_literals, n_states=200, seed=42)
    .fit(X, y, epochs=100)
    .predict(X)                              -> argmax of class sums
    .predict_with_clauses(X)                 -> argmax + firing clauses + scores
    .firing_clauses_for_class(x, class_id)   -> tuple of FiringClause
    .save(path)                              -> JSON state file
    .load(path)                              -> MultiClassTM

Conventions:
    * X: ``np.ndarray`` of shape ``(N, n_literals)``, dtype ``uint8``.
    * y: ``np.ndarray`` of shape ``(N,)``, dtype ``int64``,
      values in ``[0, n_classes)``.
    * Clauses are arranged in pairs: even ids vote ``+1`` for their
      class, odd ids vote ``-1``. ``n_clauses`` must be even.
    * Per class, per clause, per literal, two automata: one for the
      "include positive literal" decision, one for "include negation".
      State space: ``[1, n_states]``; include-action when state > n_states / 2.

This file is ~250 LoC and is self-contained -- no upstream
``GraphTsetlinMachine`` dependency, so the CPU reference path can
run on any laptop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..verify.sat_encoder import FiringClause


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class TMState:
    """Tsetlin Automata state tensor (one per class, clause, literal).

    Shape: ``(n_classes, n_clauses, 2, n_literals)`` where the size-2
    dim distinguishes positive vs negative literal automaton.
    """

    ta: np.ndarray  # int16 in [1, n_states]


# ---------------------------------------------------------------------------
# MultiClassTM
# ---------------------------------------------------------------------------


class MultiClassTM:
    """A canonical multi-class Tsetlin Machine in pure NumPy.

    All state is exposed for tests and reproducibility:
      * ``self.ta``: the automata states.
      * ``self.clause_outputs(X)``: clause-by-clause Boolean output.
      * ``self.class_scores(X)``: clipped class sums.
      * ``self.firing_clauses_for_class(...)``: list of FiringClause objects
        used by the SAT encoder.
    """

    def __init__(
        self,
        n_classes: int,
        n_clauses: int,
        T: int,
        s: float,
        n_literals: int,
        n_states: int = 200,
        seed: int = 42,
    ) -> None:
        if n_clauses % 2:
            raise ValueError("n_clauses must be even (alternating-sign clauses)")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if s <= 1.0:
            raise ValueError("s must be > 1.0")
        self.n_classes = int(n_classes)
        self.n_clauses = int(n_clauses)
        self.T = int(T)
        self.s = float(s)
        self.n_literals = int(n_literals)
        self.n_states = int(n_states)
        self.seed = int(seed)
        self._rng = np.random.default_rng(seed)

        mid = self.n_states // 2
        # Initialise just below the include-threshold so all clauses start
        # vacuous (always-true conjunctions).
        self.ta = np.full(
            (self.n_classes, self.n_clauses, 2, self.n_literals),
            mid,
            dtype=np.int16,
        )

    # ----- include masks ----
    def _include_mask(self) -> np.ndarray:
        """Bool tensor: True where the TA's action is 'include'."""
        return self.ta > (self.n_states // 2)

    # ----- forward ----
    def clause_outputs(self, X: np.ndarray) -> np.ndarray:
        """For each sample, each class, each clause, compute Boolean output.

        Vectorised: one big tensordot replaces the (C, K) Python loop.
        Returns ``(N, n_classes, n_clauses)`` uint8 tensor.

        Math:
          violation[n, c, k] = sum_l include[c, k, 0, l] * (1 - X[n, l])
                             + sum_l include[c, k, 1, l] * X[n, l]
          clause_output = 1 iff violation == 0
        """
        if X.ndim != 2 or X.shape[1] != self.n_literals:
            raise ValueError(
                f"X must have shape (N, {self.n_literals}); got {X.shape}"
            )
        if X.dtype != np.uint8:
            X = X.astype(np.uint8)

        include = self._include_mask().astype(np.int32)  # (C, K, 2, L)
        pos = include[:, :, 0, :]  # (C, K, L)
        neg = include[:, :, 1, :]  # (C, K, L)
        Xi = X.astype(np.int32)
        # violation due to positive-literal includes whose X bit is 0
        pos_v = pos @ (1 - Xi.T)         # (C, K, N)
        # violation due to negation-literal includes whose X bit is 1
        neg_v = neg @ Xi.T               # (C, K, N)
        violation = pos_v + neg_v        # (C, K, N)
        out = (violation == 0).astype(np.uint8)
        # Reshape to (N, C, K)
        return np.transpose(out, (2, 0, 1))

    def class_scores(self, X: np.ndarray) -> np.ndarray:
        """Per-class clipped vote sum: ``(N, n_classes)`` int32."""
        co = self.clause_outputs(X)
        # Alternating-sign clauses: even +1, odd -1.
        signs = np.ones(self.n_clauses, dtype=np.int8)
        signs[1::2] = -1
        votes = co.astype(np.int32) * signs.reshape(1, 1, -1)
        scores = votes.sum(axis=2)
        np.clip(scores, -self.T, self.T, out=scores)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax of class sums -> ``(N,)`` int64."""
        return self.class_scores(X).argmax(axis=1).astype(np.int64)

    def predict_with_clauses(self, X: np.ndarray, top_k_firing: int = 32) -> "list[dict]":
        """For each sample, return verdict + per-class score + firing clauses.

        Each entry is::

            {
              "verdict": int,                  # argmax class
              "class_scores": list[int],       # per-class clipped sum
              "firing_clauses": list[FiringClause]  # top-K firing clauses
            }
        """
        scores = self.class_scores(X)
        co = self.clause_outputs(X)
        out: list[dict] = []
        include = self._include_mask()
        for i in range(X.shape[0]):
            verdict = int(scores[i].argmax())
            firing: list[FiringClause] = []
            for c in range(self.n_classes):
                # Take the top-k firing clauses (positive-polarity first).
                fire_idx = np.where(co[i, c] == 1)[0]
                if fire_idx.size == 0:
                    continue
                # Heuristic for top-k: bias toward smaller clause id so the
                # output is deterministic.
                for k in fire_idx[:top_k_firing]:
                    pos_lits = tuple(int(l) for l in np.where(include[c, k, 0])[0])
                    neg_lits = tuple(int(l) for l in np.where(include[c, k, 1])[0])
                    polarity = 1 if (int(k) % 2 == 0) else -1
                    firing.append(
                        FiringClause(
                            clause_id=int(k),
                            class_id=int(c),
                            polarity=polarity,
                            weight=1,
                            positive_literal_ids=pos_lits,
                            negative_literal_ids=neg_lits,
                        )
                    )
            out.append(
                {
                    "verdict": verdict,
                    "class_scores": scores[i].tolist(),
                    "firing_clauses": firing,
                }
            )
        return out

    def firing_clauses_for_class(self, x: np.ndarray, class_id: int) -> tuple[FiringClause, ...]:
        """All firing clauses for ``x`` under ``class_id``. Used by recourse."""
        if x.ndim != 1 or x.shape[0] != self.n_literals:
            raise ValueError(f"x must be 1-D with length {self.n_literals}")
        co = self.clause_outputs(x.reshape(1, -1))[0, class_id]
        include = self._include_mask()
        out: list[FiringClause] = []
        for k, fired in enumerate(co):
            if not fired:
                continue
            pos_lits = tuple(int(l) for l in np.where(include[class_id, k, 0])[0])
            neg_lits = tuple(int(l) for l in np.where(include[class_id, k, 1])[0])
            polarity = 1 if (k % 2 == 0) else -1
            out.append(
                FiringClause(
                    clause_id=int(k),
                    class_id=int(class_id),
                    polarity=polarity,
                    weight=1,
                    positive_literal_ids=pos_lits,
                    negative_literal_ids=neg_lits,
                )
            )
        return tuple(out)

    # ----- fit ----
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, shuffle: bool = True) -> "MultiClassTM":
        """One-vs-rest Type-I / Type-II feedback (Granmo 2018).

        ``y`` is the integer class label. For each example we issue
        Type-I feedback to the target class and Type-II feedback to a
        single randomly-chosen non-target class. This is the
        multi-class scheme used by CAIR's reference implementation.
        """
        if X.ndim != 2 or X.shape[1] != self.n_literals:
            raise ValueError(f"X must have shape (N, {self.n_literals})")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(f"y must have shape ({X.shape[0]},)")
        if X.dtype != np.uint8:
            X = X.astype(np.uint8)
        y = y.astype(np.int64)
        N = X.shape[0]

        for _ in range(int(epochs)):
            order = np.arange(N)
            if shuffle:
                self._rng.shuffle(order)
            for i in order:
                xi = X[i]
                yi = int(y[i])

                # Target class: Type-I feedback (rewards include where literal=1).
                self._feedback(xi, yi, kind="I")

                # One negative class: Type-II feedback (penalises include where literal=0).
                neg_choices = [c for c in range(self.n_classes) if c != yi]
                if neg_choices:
                    cn = int(self._rng.choice(neg_choices))
                    self._feedback(xi, cn, kind="II")
        return self

    def _feedback(self, x: np.ndarray, class_id: int, kind: str) -> None:
        """Issue Type-I or Type-II feedback to a single class.

        Vectorised over clauses + literals. Per-clause + per-literal
        stochastic decisions are batched into a single uniform-random
        tensor of shape (K, L); the result is a delta tensor we add to
        ``self.ta[class_id]``.
        """
        s_inv = 1.0 / self.s
        s_inc = (self.s - 1.0) / self.s
        K = self.n_clauses
        L = self.n_literals
        n_states = self.n_states

        co_cls = self.clause_outputs(x.reshape(1, -1))[0, class_id]  # (K,)
        scores = self.class_scores(x.reshape(1, -1))[0]
        cs = int(scores[class_id])
        if kind == "I":
            prob = (self.T - max(min(cs, self.T), -self.T)) / (2.0 * self.T)
        else:
            prob = (self.T + max(min(cs, self.T), -self.T)) / (2.0 * self.T)
        if prob <= 0.0:
            return

        # Choose which clauses receive feedback this round.
        clause_mask = self._rng.random(K) <= prob  # (K,) bool
        if not clause_mask.any():
            return

        ta_view = self.ta[class_id]  # (K, 2, L) int16 -- views, mutated in place
        x_bool = x.astype(bool)
        fired = co_cls.astype(bool)  # (K,)

        if kind == "I":
            # Per-clause two cases: fired vs not fired.
            # Draw randoms for both cases up-front.
            r_main = self._rng.random((K, L))
            r_minor = self._rng.random((K, L))

            # Case A: clause fired -- reinforce supporting includes.
            fire_idx = np.where(clause_mask & fired)[0]
            if fire_idx.size:
                # For literals where x[l]==1:
                #   inc positive include with prob (s-1)/s
                #   dec negation include with prob 1/s
                # For literals where x[l]==0:
                #   dec positive include with prob 1/s
                #   inc negation include with prob (s-1)/s
                rm = r_main[fire_idx]
                rn = r_minor[fire_idx]
                x_row = np.broadcast_to(x_bool, rm.shape)
                inc_pos = ((rm < s_inc) & x_row).astype(np.int16)
                dec_neg = ((rn < s_inv) & x_row).astype(np.int16)
                dec_pos = ((rm < s_inv) & ~x_row).astype(np.int16)
                inc_neg = ((rn < s_inc) & ~x_row).astype(np.int16)
                self._apply_delta(ta_view, fire_idx, 0, inc_pos - dec_pos)
                self._apply_delta(ta_view, fire_idx, 1, inc_neg - dec_neg)

            # Case B: clause did NOT fire -- decay all includes.
            quiet_idx = np.where(clause_mask & ~fired)[0]
            if quiet_idx.size:
                rm = r_main[quiet_idx]
                rn = r_minor[quiet_idx]
                dec_pos = (rm < s_inv).astype(np.int16)
                dec_neg = (rn < s_inv).astype(np.int16)
                self._apply_delta(ta_view, quiet_idx, 0, -dec_pos)
                self._apply_delta(ta_view, quiet_idx, 1, -dec_neg)
        else:  # Type-II
            fire_idx = np.where(clause_mask & fired)[0]
            if fire_idx.size == 0:
                return
            include_arr = (ta_view > (n_states // 2))  # (K, 2, L) bool
            # Want to set:
            #   pos include when x[l]==0 AND positive not currently included
            #   neg include when x[l]==1 AND negation not currently included
            not_pos_inc = ~include_arr[fire_idx, 0, :]
            not_neg_inc = ~include_arr[fire_idx, 1, :]
            row_x = np.broadcast_to(x_bool, not_pos_inc.shape)
            d_pos = (not_pos_inc & ~row_x).astype(np.int16)
            d_neg = (not_neg_inc & row_x).astype(np.int16)
            self._apply_delta(ta_view, fire_idx, 0, d_pos)
            self._apply_delta(ta_view, fire_idx, 1, d_neg)

        # Clip to [1, n_states].
        np.clip(ta_view, 1, n_states, out=ta_view)

    def _apply_delta(self, ta_view: np.ndarray, clause_idx: np.ndarray, polarity: int, delta: np.ndarray) -> None:
        """In-place ``ta_view[clause_idx, polarity, :] += delta``."""
        if delta.shape[0] == 0:
            return
        ta_view[clause_idx, polarity, :] = ta_view[clause_idx, polarity, :].astype(np.int32) + delta.astype(np.int32)

    # ----- persistence ----
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "n_classes": self.n_classes,
            "n_clauses": self.n_clauses,
            "T": self.T,
            "s": self.s,
            "n_literals": self.n_literals,
            "n_states": self.n_states,
            "seed": self.seed,
            "ta": self.ta.tolist(),
        }
        with open(path, "w") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path: str | Path) -> "MultiClassTM":
        with open(path) as f:
            blob = json.load(f)
        tm = cls(
            n_classes=blob["n_classes"],
            n_clauses=blob["n_clauses"],
            T=blob["T"],
            s=blob["s"],
            n_literals=blob["n_literals"],
            n_states=blob.get("n_states", 200),
            seed=blob.get("seed", 42),
        )
        tm.ta = np.asarray(blob["ta"], dtype=np.int16)
        return tm

    # ----- introspection ----
    def model_version_id(self) -> str:
        """Stable model-version id for receipts.

        Hash of config + ta tensor. Two TMs with the same config and
        TA states produce the same id; this is what the receipt's
        ``model_version_id`` field pins.
        """
        from ..utils.io import sha256_of

        return sha256_of(
            {
                "n_classes": self.n_classes,
                "n_clauses": self.n_clauses,
                "T": self.T,
                "s": self.s,
                "n_literals": self.n_literals,
                "n_states": self.n_states,
                "ta_sum": int(self.ta.sum()),
                "ta_shape": list(self.ta.shape),
                "ta_xor": int(np.bitwise_xor.reduce(self.ta.flatten().astype(np.int64))),
            }
        )
