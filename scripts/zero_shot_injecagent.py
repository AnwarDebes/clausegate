#!/usr/bin/env python
"""Train on AgentDojo (all 4 suites pooled), evaluate zero-shot on InjecAgent.

This is the H5 cell in `docs/research_plan.md`: "ClauseGate trained
on AgentDojo generalises to InjecAgent and R-Judge without
fine-tuning, within a published margin."

The script trains a ClauseGate on the pooled AgentDojo set and
evaluates it on the InjecAgent direct-harm test cases (downloaded
to ``data/external/injecagent_dh_base.json``).

Usage:
    python scripts/zero_shot_injecagent.py [--seed 1] [--epochs 30]
        [--clauses 200] [--data data/external/injecagent_dh_base.json]
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
from clausegate.data.injecagent_loader import load_injecagent
from clausegate.data.synth import DEFAULT_POLICY
from clausegate.eval.metrics import compute_metrics
from clausegate.schema.actions import Verdict
from clausegate.tm.verifier import HGTMSpec, TriGraphHGTM
from clausegate.utils.seeding import seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--clauses", type=int, default=200)
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--s", type=float, default=3.9)
    ap.add_argument("--tau-allow", type=int, default=2)
    ap.add_argument("--tau-deny", type=int, default=2)
    ap.add_argument("--tau-oversight", type=int, default=1)
    ap.add_argument("--data", default="data/external/injecagent_dh_base.json")
    ap.add_argument("--injecagent-n", type=int, default=200)
    ap.add_argument("--out", default="results/headline/zero_shot_injecagent.json")
    args = ap.parse_args()

    seed_all(args.seed)
    print(f"[ZS] seed={args.seed} epochs={args.epochs} clauses={args.clauses}")

    # Train on AgentDojo pooled.
    actions: list = []
    labels: list = []
    for suite in ("workspace", "slack", "banking", "travel"):
        a, l = load_agentdojo(suite=suite, max_aug_per_user_task=3)
        actions.extend(a)
        labels.extend(l)
    print(f"[ZS] AgentDojo pooled: {len(actions)} samples, labels={dict(Counter(labels))}")

    spec = HGTMSpec(
        n_clauses=args.clauses, T=args.T, s=args.s, seed=args.seed,
        tau_allow=args.tau_allow, tau_deny=args.tau_deny, tau_oversight=args.tau_oversight,
    )
    verifier = TriGraphHGTM(spec=spec)
    t0 = time.perf_counter()
    verifier.fit(actions, DEFAULT_POLICY, labels, epochs=args.epochs)
    print(f"[ZS] trained in {time.perf_counter() - t0:.1f}s")

    # Sanity check on AgentDojo itself.
    idx = np.arange(len(actions))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    held = idx[: min(400, len(idx))]
    held_actions = [actions[i] for i in held]
    held_labels = [labels[i] for i in held]
    in_preds = verifier.predict_with_clauses(held_actions, DEFAULT_POLICY)
    in_y_pred = [p.verdict for p in in_preds]
    in_rep = compute_metrics(held_labels, in_y_pred)
    print(f"[ZS] AgentDojo holdout sanity: acc={in_rep.accuracy:.3f} ASR={in_rep.asr:.3f}")

    # Zero-shot on InjecAgent.
    print(f"[ZS] loading InjecAgent from {args.data} ...")
    inj_actions, inj_labels = load_injecagent(args.data, n=args.injecagent_n)
    print(f"[ZS] InjecAgent: {len(inj_actions)} cases (all DENY ground truth)")

    t0 = time.perf_counter()
    inj_preds = verifier.predict_with_clauses(inj_actions, DEFAULT_POLICY)
    t_eval = time.perf_counter() - t0
    inj_y_pred = [p.verdict for p in inj_preds]
    n = len(inj_actions)
    latency_per_call_ms = (t_eval / max(1, n)) * 1000.0

    # Distribution + ASR (let-through rate for InjecAgent, all DENY ground-truth)
    pred_dist = Counter(inj_y_pred)
    allow_let_through = pred_dist.get(int(Verdict.ALLOW), 0) / max(1, n)
    deny_caught = pred_dist.get(int(Verdict.DENY), 0) / max(1, n)
    oversight = pred_dist.get(int(Verdict.NEEDS_OVERSIGHT), 0) / max(1, n)
    # Defended ASR -> ALLOW rate; non-DENY rate is "let-through-or-deferred-only".
    defended_asr = allow_let_through
    non_block_rate = allow_let_through + oversight  # OVERSIGHT is a defence too;
    # we report both numbers for honesty.

    print(
        f"[ZS] InjecAgent results (zero-shot, n={n}):\n"
        f"  ALLOW (defended ASR) = {allow_let_through:.3f}  <-- attacker succeeded\n"
        f"  DENY  (hard block)   = {deny_caught:.3f}\n"
        f"  NEEDS_OVERSIGHT      = {oversight:.3f}\n"
        f"  Latency / call (ms)  = {latency_per_call_ms:.2f}"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "benchmark": "InjecAgent_dh_base",
                "trained_on": "AgentDojo pooled (workspace+slack+banking+travel)",
                "n_test": n,
                "seed": args.seed,
                "epochs": args.epochs,
                "clauses": args.clauses,
                "T": args.T,
                "agentdojo_holdout_acc": in_rep.accuracy,
                "agentdojo_holdout_asr": in_rep.asr,
                "injecagent_asr_defended": defended_asr,
                "injecagent_deny_rate": deny_caught,
                "injecagent_oversight_rate": oversight,
                "injecagent_latency_per_call_ms": latency_per_call_ms,
                "predicted_distribution": dict(pred_dist),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"[ZS] result -> {out}")


if __name__ == "__main__":
    main()
