#!/usr/bin/env python
"""Zero-shot eval on R-Judge after training on AgentDojo.

R-Judge has binary labels (safe / unsafe); we map safe -> ALLOW
and unsafe -> DENY. ClauseGate's NEEDS_OVERSIGHT is also a
defence (operator reviews), so we report two metrics:
  * defended ASR = fraction of UNSAFE cases predicted ALLOW
  * macro-F1 over the 2-class projection {ALLOW, DENY}

Usage:
    python scripts/zero_shot_rjudge.py [--seed 1] [--epochs 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clausegate.data.agentdojo_loader import load_agentdojo
from clausegate.data.rjudge_loader import load_rjudge
from clausegate.data.synth import DEFAULT_POLICY
from clausegate.eval.metrics import compute_metrics
from clausegate.schema.actions import Verdict
from clausegate.tm.verifier import HGTMSpec, TriGraphHGTM
from clausegate.utils.seeding import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--clauses", type=int, default=200)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--s", type=float, default=3.9)
    ap.add_argument("--tau-allow", type=int, default=2)
    ap.add_argument("--tau-deny", type=int, default=2)
    ap.add_argument("--tau-oversight", type=int, default=1)
    ap.add_argument("--data-dir", default="data/external")
    ap.add_argument("--out", default="results/headline/zero_shot_rjudge.json")
    args = ap.parse_args()

    seed_all(args.seed)
    print(f"[ZS-RJ] seed={args.seed} epochs={args.epochs}")

    # Train on AgentDojo pooled.
    actions, labels = [], []
    for suite in ("workspace", "slack", "banking", "travel"):
        a, l = load_agentdojo(suite=suite, max_aug_per_user_task=3)
        actions.extend(a)
        labels.extend(l)
    print(f"[ZS-RJ] AgentDojo: {len(actions)} samples, labels={dict(Counter(labels))}")

    spec = HGTMSpec(
        n_clauses=args.clauses, T=args.T, s=args.s, seed=args.seed,
        tau_allow=args.tau_allow, tau_deny=args.tau_deny, tau_oversight=args.tau_oversight,
    )
    verifier = TriGraphHGTM(spec=spec)
    t0 = time.perf_counter()
    verifier.fit(actions, DEFAULT_POLICY, labels, epochs=args.epochs)
    print(f"[ZS-RJ] trained in {time.perf_counter() - t0:.1f}s")

    # Load R-Judge across all categories.
    print(f"[ZS-RJ] loading R-Judge from {args.data_dir} ...")
    import os

    rj_actions: list = []
    rj_labels: list = []
    for fname in sorted(os.listdir(args.data_dir)):
        if not fname.startswith("rjudge_"):
            continue
        try:
            a, l = load_rjudge(Path(args.data_dir) / fname)
        except Exception as e:
            print(f"  [skip] {fname}: {e!r}")
            continue
        rj_actions.extend(a)
        rj_labels.extend(l)
    print(f"[ZS-RJ] R-Judge: {len(rj_actions)} parsed traces, labels={dict(Counter(rj_labels))}")

    if not rj_actions:
        raise SystemExit("no R-Judge traces parsed")

    t0 = time.perf_counter()
    preds = verifier.predict_with_clauses(rj_actions, DEFAULT_POLICY)
    t_eval = time.perf_counter() - t0
    y_pred = [p.verdict for p in preds]
    n = len(rj_actions)
    latency_per_call_ms = (t_eval / max(1, n)) * 1000.0

    rep = compute_metrics(rj_labels, y_pred)
    # Compute 2-class defended view: NEEDS_OVERSIGHT collapsed into DENY
    # for the "defence triggered" view, and into ALLOW for the strict
    # 2-class evaluation.
    y_def = [int(Verdict.DENY) if v == int(Verdict.NEEDS_OVERSIGHT) else v for v in y_pred]
    rep_def = compute_metrics(rj_labels, y_def)
    print(
        f"[ZS-RJ] Strict 3-class:\n"
        f"  acc           = {rep.accuracy:.3f}\n"
        f"  macro-F1      = {rep.macro_f1:.3f}\n"
        f"  ASR           = {rep.asr:.3f}\n"
        f"  benign util.  = {rep.benign_utility:.3f}\n"
        f"\n"
        f"[ZS-RJ] Defended view (OVERSIGHT -> DENY):\n"
        f"  acc           = {rep_def.accuracy:.3f}\n"
        f"  macro-F1      = {rep_def.macro_f1:.3f}\n"
        f"  ASR           = {rep_def.asr:.3f}\n"
        f"\n"
        f"  Latency / call (ms) = {latency_per_call_ms:.2f}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "benchmark": "R-Judge",
                "trained_on": "AgentDojo pooled (workspace+slack+banking+travel)",
                "n_test": n,
                "seed": args.seed,
                "epochs": args.epochs,
                "clauses": args.clauses,
                "strict_3class": {
                    "accuracy": rep.accuracy,
                    "macro_f1": rep.macro_f1,
                    "asr": rep.asr,
                    "benign_utility": rep.benign_utility,
                    "confusion": rep.confusion,
                },
                "defended_2class": {
                    "accuracy": rep_def.accuracy,
                    "macro_f1": rep_def.macro_f1,
                    "asr": rep_def.asr,
                    "confusion": rep_def.confusion,
                },
                "latency_per_call_ms": latency_per_call_ms,
                "predicted_distribution": dict(Counter(y_pred)),
                "true_distribution": dict(Counter(rj_labels)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"[ZS-RJ] result -> {out}")


if __name__ == "__main__":
    main()
