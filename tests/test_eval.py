"""Tests for metrics + adversarial harness."""
from __future__ import annotations

from pathlib import Path

import pytest

from clausegate.data.synth import DEFAULT_POLICY, SyntheticGenerator
from clausegate.eval.adversarial import AdversarialHarness, AttackResult
from clausegate.eval.logger import PerSampleLogger
from clausegate.eval.metrics import compute_metrics
from clausegate.schema.actions import Verdict
from clausegate.tm.verifier import HGTMSpec, TriGraphHGTM


def test_compute_metrics_perfect():
    y = [int(Verdict.ALLOW), int(Verdict.DENY), int(Verdict.NEEDS_OVERSIGHT)]
    rep = compute_metrics(y, y)
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == 1.0
    assert rep.asr == 0.0
    assert rep.benign_utility == 1.0


def test_compute_metrics_asr_and_utility():
    # 2 ALLOW samples correctly allowed; 2 DENY samples: one let through, one blocked.
    y_true = [int(Verdict.ALLOW), int(Verdict.ALLOW), int(Verdict.DENY), int(Verdict.DENY)]
    y_pred = [int(Verdict.ALLOW), int(Verdict.ALLOW), int(Verdict.ALLOW), int(Verdict.DENY)]
    rep = compute_metrics(y_true, y_pred)
    assert rep.benign_utility == 1.0
    assert rep.asr == 0.5  # one DENY ground-truth was predicted ALLOW.


def test_per_sample_logger_writes_jsonl(tmp_path: Path):
    p = tmp_path / "jsonl" / "log.jsonl"
    with PerSampleLogger(p) as log:
        log.write({"sample_id": "a", "verdict": 0})
        log.write({"sample_id": "b", "verdict": 1})
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    import json

    rec0 = json.loads(lines[0])
    assert rec0["sample_id"] == "a"
    assert "ts" in rec0


def test_adversarial_harness_runs():
    gen = SyntheticGenerator(seed=21)
    actions, labels = gen.batch(48)
    spec = HGTMSpec(n_clauses=20, T=5, s=3.0, seed=21)
    verifier = TriGraphHGTM(spec=spec)
    verifier.fit(actions, DEFAULT_POLICY, labels, epochs=2)

    # Pick a small benign subset.
    benign = [a for a, y in zip(actions, labels) if y == int(Verdict.ALLOW)][:8]
    if not benign:
        pytest.skip("synthetic batch did not include any ALLOW samples")
    harness = AdversarialHarness(verifier, DEFAULT_POLICY, seed=21)
    results = harness.run_all(benign)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, AttackResult)
        assert 0.0 <= r.asr_before <= 1.0
        assert 0.0 <= r.asr_after <= 1.0
        assert r.n == len(benign)
