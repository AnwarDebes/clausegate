"""TriGraphHGTM: the production verifier wrapper.

The TriGraphHGTM ties together the graph builder, the binariser,
and the multi-class TM into a single ``predict`` / ``fit`` /
``predict_with_clauses`` API. It is the surface the recourse and
receipt modules call.

We deliberately do NOT depend on CUDA in this file. The reference
backend is ``clausegate.tm.vanilla.MultiClassTM`` (pure NumPy). A
CUDA-backed implementation is a drop-in replacement that exposes
the same ``predict_with_clauses`` shape; the parity test in
``tests/test_cuda_parity.py`` enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..graphs.binarize import SYMBOL_VOCAB, BinaryFeatures, Vocab, binarize
from ..graphs.builder import TriGraph, build_tri_graph
from ..schema.actions import AgentAction, Policy, Verdict
from ..utils.io import sha256_of
from ..verify.sat_encoder import FiringClause
from .vanilla import MultiClassTM


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HGTMSpec:
    """Architecture spec for the verifier.

    Mirrors graphtm-cbr's ``HGraphTMSpec`` but trimmed to the
    knobs ClauseGate exposes. The CUDA backend reads this and
    compiles kernels at constructor time; the NumPy backend uses
    only ``n_classes``, ``n_clauses``, ``T``, ``s``, ``n_states``.
    """

    n_classes: int = 3
    n_clauses: int = 200
    T: int = 50
    s: float = 3.9
    n_states: int = 200
    R: int = 2
    IA: int = 2
    IF: int = 5
    LA: int = 15
    LF: int = 3
    seed: int = 42
    tau_allow: int = 5
    tau_deny: int = 3
    tau_oversight: int = 2

    def __post_init__(self) -> None:
        if self.n_classes != 3:
            raise ValueError("ClauseGate expects exactly 3 classes (ALLOW, DENY, NEEDS_OVERSIGHT)")
        if self.n_clauses % 2:
            raise ValueError("n_clauses must be even")
        if self.s <= 1.0:
            raise ValueError("s must be > 1.0")
        if self.T < 1:
            raise ValueError("T must be >= 1")


# ---------------------------------------------------------------------------
# PredictionWithClauses
# ---------------------------------------------------------------------------


@dataclass
class PredictionWithClauses:
    """Result of ``TriGraphHGTM.predict_with_clauses`` for one sample.

    Attributes:
        verdict: ``Verdict`` enum value (0/1/2).
        verdict_name: human-readable string.
        class_scores: tuple of int, length 3.
        margin: integer; gap between top and second class.
        firing_clauses: tuple of FiringClause objects for the
            argmax class only.
        features: the BinaryFeatures used to derive this prediction
            (handy for the receipt + recourse modules).
        tri_graph_signature: signature of the source TriGraph.
        action_graph_hash / context_graph_hash / policy_graph_hash:
            per-sub-graph SHA-256 of the source TriGraph.
        model_version_id: pins the model to its ta-state snapshot.
    """

    verdict: int
    verdict_name: str
    class_scores: tuple[int, ...]
    margin: int
    firing_clauses: tuple[FiringClause, ...]
    features: BinaryFeatures
    tri_graph_signature: str
    model_version_id: str
    action_graph_hash: str = ""
    context_graph_hash: str = ""
    policy_graph_hash: str = ""

    def is_below_oversight_margin(self, threshold: int) -> bool:
        return self.margin < threshold


# ---------------------------------------------------------------------------
# TriGraphHGTM
# ---------------------------------------------------------------------------


class TriGraphHGTM:
    """The verifier surface used by recourse + receipt modules.

    The default backend is the NumPy MultiClassTM (CPU). A
    constructor argument ``backend="cuda"`` swaps in a GPU-backed
    implementation that has the same interface; that implementation
    lives in ``clausegate.tm.cuda_runner`` and is loaded lazily.
    """

    def __init__(
        self,
        spec: HGTMSpec | None = None,
        vocab: Vocab = SYMBOL_VOCAB,
        backend: str = "numpy",
    ) -> None:
        self.spec = spec if spec is not None else HGTMSpec()
        self.vocab = vocab
        self.backend = backend
        if backend == "numpy":
            self._impl = MultiClassTM(
                n_classes=self.spec.n_classes,
                n_clauses=self.spec.n_clauses,
                T=self.spec.T,
                s=self.spec.s,
                n_literals=self.vocab.total_size,
                n_states=self.spec.n_states,
                seed=self.spec.seed,
            )
        elif backend == "cuda":
            from .cuda_runner import CudaMultiClassTM

            self._impl = CudaMultiClassTM.from_spec(self.spec, n_literals=self.vocab.total_size)
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # ----- conversions -----
    def _features_for(self, actions: Sequence[AgentAction], policy: Policy) -> tuple[np.ndarray, list[BinaryFeatures], list[TriGraph]]:
        feats: list[BinaryFeatures] = []
        graphs: list[TriGraph] = []
        for a in actions:
            tg = build_tri_graph(a, policy)
            bf = binarize(tg, self.vocab)
            feats.append(bf)
            graphs.append(tg)
        X = np.stack([f.literal_vector for f in feats]).astype(np.uint8)
        return X, feats, graphs

    # ----- training -----
    def fit(
        self,
        actions: Sequence[AgentAction],
        policy: Policy,
        labels: Sequence[int] | np.ndarray,
        *,
        epochs: int = 100,
        shuffle: bool = True,
    ) -> "TriGraphHGTM":
        X, _, _ = self._features_for(actions, policy)
        y = np.asarray(labels, dtype=np.int64)
        if y.shape[0] != X.shape[0]:
            raise ValueError("labels length mismatch")
        self._impl.fit(X, y, epochs=epochs, shuffle=shuffle)
        return self

    # ----- inference -----
    def predict(self, actions: Sequence[AgentAction], policy: Policy) -> np.ndarray:
        X, _, _ = self._features_for(actions, policy)
        return self._impl.predict(X)

    def predict_one(self, action: AgentAction, policy: Policy) -> PredictionWithClauses:
        rows = self.predict_with_clauses([action], policy)
        return rows[0]

    def predict_with_clauses(
        self, actions: Sequence[AgentAction], policy: Policy
    ) -> list[PredictionWithClauses]:
        X, feats, graphs = self._features_for(actions, policy)
        raw = self._impl.predict_with_clauses(X)
        out: list[PredictionWithClauses] = []
        for i, row in enumerate(raw):
            scores = tuple(int(s) for s in row["class_scores"])
            verdict, verdict_name, margin = _resolve_verdict(scores, self.spec)
            # The TM's argmax may differ from the resolved verdict when
            # the margin falls into the NEEDS-OVERSIGHT band; we keep the
            # TM's firing clauses for the *true* argmax so the receipt
            # records the substantive vote, not the safety upgrade.
            argmax_class = int(np.argmax(scores))
            fc_all = tuple(row["firing_clauses"])
            fc_for_argmax = tuple(f for f in fc_all if f.class_id == argmax_class)
            out.append(
                PredictionWithClauses(
                    verdict=verdict,
                    verdict_name=verdict_name,
                    class_scores=scores,
                    margin=margin,
                    firing_clauses=fc_for_argmax,
                    features=feats[i],
                    tri_graph_signature=graphs[i].signature(),
                    model_version_id=self._impl.model_version_id(),
                    action_graph_hash=graphs[i].action_hash,
                    context_graph_hash=graphs[i].context_hash,
                    policy_graph_hash=graphs[i].policy_hash,
                )
            )
        return out

    # ----- accessors -----
    def model_version_id(self) -> str:
        return self._impl.model_version_id()

    def n_literals(self) -> int:
        return self.vocab.total_size

    # ----- raw access for recourse + verify -----
    def firing_clauses_for_class(self, x: np.ndarray, class_id: int) -> tuple[FiringClause, ...]:
        return self._impl.firing_clauses_for_class(x, class_id)

    def class_scores(self, x: np.ndarray) -> np.ndarray:
        return self._impl.class_scores(x.reshape(1, -1))[0]


# ---------------------------------------------------------------------------
# Verdict resolution
# ---------------------------------------------------------------------------


def _resolve_verdict(scores: tuple[int, ...], spec: HGTMSpec) -> tuple[int, str, int]:
    """Resolve TM class sums to a verdict with margin thresholds.

    Rules (mirror docs/ARCHITECTURE.md):
      * ALLOW iff scores[ALLOW] - max(scores[DENY], scores[OVERSIGHT]) >= tau_allow.
      * DENY iff argmax == DENY and margin >= tau_deny.
      * Otherwise NEEDS_OVERSIGHT.
    """
    s_allow = scores[Verdict.ALLOW]
    s_deny = scores[Verdict.DENY]
    s_over = scores[Verdict.NEEDS_OVERSIGHT]
    argmax_idx = int(np.argmax(scores))
    sorted_scores = sorted(scores, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]

    if s_allow - max(s_deny, s_over) >= spec.tau_allow:
        return int(Verdict.ALLOW), "ALLOW", margin
    if argmax_idx == int(Verdict.DENY) and (s_deny - max(s_allow, s_over)) >= spec.tau_deny:
        return int(Verdict.DENY), "DENY", margin
    return int(Verdict.NEEDS_OVERSIGHT), "NEEDS_OVERSIGHT", margin
