#!/usr/bin/env python
"""Smoke training run on AgentDojo's ``workspace`` suite.

Pre-condition: ``pip install -e ".[full]"`` (pulls in ``agentdojo``).

Usage:
    python scripts/train_workspace_smoke.py --seed 1 --epochs 5

This script is intentionally short so it can be wall-clocked on
CPU. Production-scale 5-seed training uses
``scripts/train_full_agentdojo.py`` (not yet written; see
``docs/research_plan.md``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clausegate.data.agentdojo_loader import load_agentdojo
from clausegate.data.synth import DEFAULT_POLICY
from clausegate.eval.metrics import compute_metrics
from clausegate.tm.verifier import HGTMSpec, TriGraphHGTM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--suite", default="workspace")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    print(f"[smoke] loading agentdojo suite={args.suite} n={args.n}")
    actions, labels = load_agentdojo(suite=args.suite, n=args.n)
    print(f"[smoke] loaded {len(actions)} samples")

    spec = HGTMSpec(n_clauses=200, T=50, s=3.9, seed=args.seed)
    verifier = TriGraphHGTM(spec=spec)
    verifier.fit(actions, DEFAULT_POLICY, labels, epochs=args.epochs)

    preds = verifier.predict_with_clauses(actions, DEFAULT_POLICY)
    y_pred = [p.verdict for p in preds]
    margins = [p.margin for p in preds]
    rep = compute_metrics(labels, y_pred, margins=margins)
    print()
    print(f"[smoke] acc           = {rep.accuracy:.3f}")
    print(f"[smoke] macro-F1      = {rep.macro_f1:.3f}")
    print(f"[smoke] ASR           = {rep.asr:.3f}")
    print(f"[smoke] benign util.  = {rep.benign_utility:.3f}")
    print(f"[smoke] oversight rec = {rep.oversight_recall:.3f}")


if __name__ == "__main__":
    main()
