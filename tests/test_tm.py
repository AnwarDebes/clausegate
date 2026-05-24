"""Tests for the vanilla MultiClassTM + TriGraphHGTM verifier."""
from __future__ import annotations

import numpy as np
import pytest

from clausegate.data.synth import DEFAULT_POLICY, SyntheticGenerator
from clausegate.graphs.binarize import SYMBOL_VOCAB
from clausegate.tm.vanilla import MultiClassTM
from clausegate.tm.verifier import HGTMSpec, TriGraphHGTM


def test_vanilla_tm_construct_and_predict_shape():
    tm = MultiClassTM(n_classes=3, n_clauses=20, T=10, s=3.0, n_literals=64, seed=0)
    X = np.zeros((4, 64), dtype=np.uint8)
    X[:, :8] = 1
    pred = tm.predict(X)
    assert pred.shape == (4,)
    scores = tm.class_scores(X)
    assert scores.shape == (4, 3)


def test_vanilla_tm_rejects_invalid_args():
    with pytest.raises(ValueError):
        MultiClassTM(n_classes=3, n_clauses=21, T=10, s=3.0, n_literals=64)  # odd clauses
    with pytest.raises(ValueError):
        MultiClassTM(n_classes=1, n_clauses=20, T=10, s=3.0, n_literals=64)  # too few classes
    with pytest.raises(ValueError):
        MultiClassTM(n_classes=3, n_clauses=20, T=10, s=0.5, n_literals=64)  # s <= 1


def test_vanilla_tm_fit_improves_train_acc_on_synthetic_pattern():
    rng = np.random.default_rng(0)
    L = 16
    N = 256
    # Class is determined by first literal.
    y = rng.integers(0, 3, size=N).astype(np.int64)
    X = rng.integers(0, 2, size=(N, L)).astype(np.uint8)
    # Encode label deterministically into the first three literals.
    X[:, 0] = (y == 0).astype(np.uint8)
    X[:, 1] = (y == 1).astype(np.uint8)
    X[:, 2] = (y == 2).astype(np.uint8)

    tm = MultiClassTM(n_classes=3, n_clauses=30, T=10, s=3.9, n_literals=L, seed=0)
    init_pred = tm.predict(X)
    init_acc = float((init_pred == y).mean())
    tm.fit(X, y, epochs=15)
    final_acc = float((tm.predict(X) == y).mean())
    # We don't require perfect convergence on a single seed in 15 epochs,
    # but training must improve over the always-class-0 baseline.
    assert final_acc > init_acc + 0.05, f"acc did not improve: {init_acc} -> {final_acc}"


def test_vanilla_tm_save_load_round_trip(tmp_path):
    tm = MultiClassTM(n_classes=3, n_clauses=10, T=5, s=3.0, n_literals=12, seed=1)
    p = tmp_path / "tm.json"
    tm.save(p)
    tm2 = MultiClassTM.load(p)
    assert tm2.n_classes == tm.n_classes
    assert tm2.n_clauses == tm.n_clauses
    assert (tm2.ta == tm.ta).all()


def test_vanilla_tm_firing_clauses_for_class_shape():
    tm = MultiClassTM(n_classes=3, n_clauses=10, T=5, s=3.0, n_literals=8, seed=2)
    x = np.ones(8, dtype=np.uint8)
    fc = tm.firing_clauses_for_class(x, class_id=0)
    # All clauses fire by default (TA below include threshold = all-1 conjunction).
    assert len(fc) == 10
    assert all(f.class_id == 0 for f in fc)


def test_tri_graph_hgtm_construct():
    spec = HGTMSpec(n_clauses=20, T=10, s=3.0)
    verifier = TriGraphHGTM(spec=spec)
    assert verifier.n_literals() == SYMBOL_VOCAB.total_size
    assert verifier.spec.n_classes == 3


def test_tri_graph_hgtm_smoke_train():
    gen = SyntheticGenerator(seed=7)
    actions, labels = gen.batch(96)
    spec = HGTMSpec(n_clauses=40, T=10, s=3.5)
    verifier = TriGraphHGTM(spec=spec)
    verifier.fit(actions, DEFAULT_POLICY, labels, epochs=3)
    preds = verifier.predict_with_clauses(actions[:8], DEFAULT_POLICY)
    assert len(preds) == 8
    for p in preds:
        assert p.verdict in (0, 1, 2)
        assert p.verdict_name in ("ALLOW", "DENY", "NEEDS_OVERSIGHT")
        assert isinstance(p.firing_clauses, tuple)
