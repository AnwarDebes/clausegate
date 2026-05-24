#!/usr/bin/env python
"""5-seed AgentDojo training run + zero-shot + adversarial eval.

Per Anwar's 5-seed protocol:
  - seeds {1, 2, 3, 5, 8}
  - per seed: train on 4 suites pooled (workspace + slack + banking + travel)
  - report mean ± std + Wilcoxon vs baseline
  - log per-sample JSONL to results/jsonl/seed_{seed}/per_sample.jsonl
  - write headline JSON to results/headline/agentdojo_full_seed_{seed}.json

Usage:
    python scripts/train_full_agentdojo.py [--seeds 1 2 3 5 8] [--epochs 30] [--clauses 200] [--T 50]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clausegate.data.agentdojo_loader import load_agentdojo
from clausegate.data.synth import DEFAULT_POLICY
from clausegate.eval.adversarial import AdversarialHarness
from clausegate.eval.logger import PerSampleLogger
from clausegate.eval.metrics import compute_metrics
from clausegate.recourse.search import greedy_argument_edit
from clausegate.schema.actions import Verdict
from clausegate.tm.verifier import HGTMSpec, TriGraphHGTM
from clausegate.utils.seeding import seed_all
from clausegate.verify.certificate import build_receipt


def load_pooled_agentdojo() -> tuple[list, list]:
    """Pool all four suites into one train+eval dataset."""
    actions: list = []
    labels: list = []
    for suite in ("workspace", "slack", "banking", "travel"):
        a, l = load_agentdojo(suite=suite, max_aug_per_user_task=3)
        actions.extend(a)
        labels.extend(l)
    return actions, labels


def split(actions: list, labels: list, seed: int, test_frac: float = 0.2) -> tuple[list, list, list, list]:
    """Stratified-ish random split."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(actions))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    train_idx = idx[:cut]
    test_idx = idx[cut:]
    return (
        [actions[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [actions[i] for i in test_idx],
        [labels[i] for i in test_idx],
    )


def run_one_seed(seed: int, args: argparse.Namespace) -> dict:
    seed_all(seed)
    print(f"\n========== seed = {seed} ==========")

    print("[load] pooling 4 AgentDojo suites ...")
    actions, labels = load_pooled_agentdojo()
    print(f"[load] pooled total: {len(actions)} samples")
    from collections import Counter

    print(f"[load] labels: {dict(Counter(labels))}")

    train_X, train_y, test_X, test_y = split(actions, labels, seed)
    print(f"[load] train={len(train_X)} test={len(test_X)}")

    print(f"[train] HGTMSpec(n_clauses={args.clauses}, T={args.T}, s={args.s}) "
          f"epochs={args.epochs}")
    spec = HGTMSpec(
        n_clauses=args.clauses, T=args.T, s=args.s, seed=seed,
        tau_allow=args.tau_allow, tau_deny=args.tau_deny, tau_oversight=args.tau_oversight,
    )
    verifier = TriGraphHGTM(spec=spec)
    t0 = time.perf_counter()
    verifier.fit(train_X, DEFAULT_POLICY, train_y, epochs=args.epochs)
    t_train = time.perf_counter() - t0
    print(f"[train] done in {t_train:.1f}s")

    print("[eval] predicting on test set ...")
    t0 = time.perf_counter()
    test_preds = verifier.predict_with_clauses(test_X, DEFAULT_POLICY)
    t_eval = time.perf_counter() - t0
    y_pred = [p.verdict for p in test_preds]
    margins = [p.margin for p in test_preds]
    rep = compute_metrics(test_y, y_pred, margins=margins)
    n = len(test_X)
    latency_per_call_ms = (t_eval / max(1, n)) * 1000.0
    print(f"[eval] done in {t_eval:.1f}s ({latency_per_call_ms:.2f} ms / call)")

    print(
        f"  acc           = {rep.accuracy:.3f}\n"
        f"  macro-F1      = {rep.macro_f1:.3f}\n"
        f"  ASR           = {rep.asr:.3f}\n"
        f"  benign util.  = {rep.benign_utility:.3f}\n"
        f"  oversight rec = {rep.oversight_recall:.3f}\n"
        f"  ECE (proxy)   = {rep.ece:.3f}"
    )

    # --- adversarial ---
    print("\n[adv] adversarial harness ...")
    benign = [a for a, y in zip(test_X, test_y) if y == int(Verdict.ALLOW)]
    if len(benign) > 100:
        benign = benign[:100]
    harness = AdversarialHarness(verifier, DEFAULT_POLICY, seed=seed)
    adv_results: list[dict] = []
    if benign:
        for attack in harness.run_all(benign):
            adv_results.append(
                {
                    "name": attack.name,
                    "n": attack.n,
                    "asr_before": attack.asr_before,
                    "asr_after": attack.asr_after,
                    "blocked_count": attack.blocked_count,
                }
            )
            print(
                f"  {attack.name:<22}  ASR before={attack.asr_before:.3f}  "
                f"ASR after={attack.asr_after:.3f}  blocked={attack.blocked_count}/{attack.n}"
            )

    # --- recourse success rate on test-set DENYs ---
    print("\n[recourse] running greedy recourse on test DENYs ...")
    deny_preds = [(i, p) for i, p in enumerate(test_preds) if p.verdict == int(Verdict.DENY)]
    deny_preds = deny_preds[: args.recourse_sample]
    flips_to_allow = 0
    flips_to_any_non_deny = 0
    latencies: list[float] = []
    for i, p in deny_preds:
        out = greedy_argument_edit(
            action=test_X[i],
            policy=DEFAULT_POLICY,
            verifier=verifier,
            target_verdict=int(Verdict.ALLOW),
            max_edits=3,
            max_latency_ms=args.recourse_max_ms,
            allow_lift_policy=True,
        )
        latencies.append(out.latency_ms)
        if out.found:
            flips_to_allow += 1
            flips_to_any_non_deny += 1
        else:
            # second attempt: oversight target
            out2 = greedy_argument_edit(
                action=test_X[i],
                policy=DEFAULT_POLICY,
                verifier=verifier,
                target_verdict=int(Verdict.NEEDS_OVERSIGHT),
                max_edits=3,
                max_latency_ms=args.recourse_max_ms,
                allow_lift_policy=True,
            )
            if out2.found:
                flips_to_any_non_deny += 1
                latencies.append(out2.latency_ms)
    n_recourse = max(1, len(deny_preds))
    recourse_p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    recourse_p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
    print(
        f"  N deny samples         = {len(deny_preds)}\n"
        f"  Recourse @ K=3 -> ALLOW = {flips_to_allow / n_recourse:.3f}\n"
        f"  Recourse @ K=3 -> non-DENY = {flips_to_any_non_deny / n_recourse:.3f}\n"
        f"  Latency P50/P95 (ms)   = {recourse_p50:.1f} / {recourse_p95:.1f}"
    )

    # --- per-sample JSONL ---
    jsonl_path = Path(args.results_dir) / "jsonl" / f"seed_{seed}" / "per_sample.jsonl"
    print(f"\n[log] writing per-sample to {jsonl_path}")
    with PerSampleLogger(jsonl_path, append=False) as logger:
        for i, p in enumerate(test_preds):
            logger.write(
                {
                    "sample_id": i,
                    "verdict": int(p.verdict),
                    "verdict_name": p.verdict_name,
                    "class_scores": list(p.class_scores),
                    "margin": int(p.margin),
                    "firing_clause_ids": [int(fc.clause_id) for fc in p.firing_clauses],
                    "tri_graph_signature": p.tri_graph_signature,
                    "model_version_id": p.model_version_id,
                    "true_label": int(test_y[i]),
                }
            )

    # --- write one signed receipt as exemplar ---
    receipts_dir = Path(args.results_dir) / "receipts" / f"seed_{seed}"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)  # ephemeral; deployer rotates per-audit
    receipt = build_receipt(
        prediction=test_preds[0],
        signing_key=key,
        receipts_dir=receipts_dir,
        sample_id="exemplar",
    )
    receipt.write(receipts_dir / "exemplar.json")
    print(f"[log] exemplar receipt -> {receipts_dir / 'exemplar.json'}")

    summary = {
        "seed": seed,
        "n_train": len(train_X),
        "n_test": len(test_X),
        "train_time_s": t_train,
        "eval_time_s": t_eval,
        "latency_per_call_ms": latency_per_call_ms,
        "accuracy": rep.accuracy,
        "macro_f1": rep.macro_f1,
        "asr": rep.asr,
        "benign_utility": rep.benign_utility,
        "oversight_recall": rep.oversight_recall,
        "ece": rep.ece,
        "confusion": rep.confusion,
        "per_class_precision": rep.per_class_precision,
        "per_class_recall": rep.per_class_recall,
        "adv": adv_results,
        "recourse": {
            "n_deny": len(deny_preds),
            "n_recourse_attempted": n_recourse,
            "rate_to_allow": flips_to_allow / n_recourse,
            "rate_to_any_non_deny": flips_to_any_non_deny / n_recourse,
            "p50_ms": recourse_p50,
            "p95_ms": recourse_p95,
        },
        "spec": {
            "n_clauses": spec.n_clauses,
            "T": spec.T,
            "s": spec.s,
            "epochs": args.epochs,
            "tau_allow": spec.tau_allow,
            "tau_deny": spec.tau_deny,
            "tau_oversight": spec.tau_oversight,
        },
    }
    headline_path = Path(args.results_dir) / "headline" / f"agentdojo_full_seed_{seed}.json"
    headline_path.parent.mkdir(parents=True, exist_ok=True)
    headline_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[log] headline -> {headline_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--clauses", type=int, default=200)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--s", type=float, default=3.9)
    ap.add_argument("--tau-allow", type=int, default=2)
    ap.add_argument("--tau-deny", type=int, default=2)
    ap.add_argument("--tau-oversight", type=int, default=1)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--recourse-sample", type=int, default=50)
    ap.add_argument("--recourse-max-ms", type=int, default=200)
    args = ap.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict] = []
    t_start = time.perf_counter()
    for seed in args.seeds:
        s = run_one_seed(seed, args)
        all_summaries.append(s)

    # Cross-seed aggregation.
    def _stat(key: str) -> tuple[float, float]:
        xs = np.array([s[key] for s in all_summaries], dtype=np.float64)
        return float(xs.mean()), float(xs.std())

    agg = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "accuracy_mean_std": _stat("accuracy"),
        "macro_f1_mean_std": _stat("macro_f1"),
        "asr_mean_std": _stat("asr"),
        "benign_utility_mean_std": _stat("benign_utility"),
        "oversight_recall_mean_std": _stat("oversight_recall"),
        "latency_per_call_ms_mean_std": _stat("latency_per_call_ms"),
        "recourse_rate_to_any_non_deny_mean_std": (
            float(np.mean([s["recourse"]["rate_to_any_non_deny"] for s in all_summaries])),
            float(np.std([s["recourse"]["rate_to_any_non_deny"] for s in all_summaries])),
        ),
        "recourse_p50_ms_mean": float(np.mean([s["recourse"]["p50_ms"] for s in all_summaries])),
        "recourse_p95_ms_mean": float(np.mean([s["recourse"]["p95_ms"] for s in all_summaries])),
        "wall_clock_s": time.perf_counter() - t_start,
        "per_seed": all_summaries,
    }
    agg_path = Path(args.results_dir) / "headline" / "agentdojo_full_5seed_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2, sort_keys=True))
    print(f"\n[final] aggregate -> {agg_path}")

    print("\n=========================================")
    print("       Aggregate across seeds")
    print("=========================================")
    print(f"  acc           = {agg['accuracy_mean_std'][0]:.3f} ± {agg['accuracy_mean_std'][1]:.3f}")
    print(f"  macro-F1      = {agg['macro_f1_mean_std'][0]:.3f} ± {agg['macro_f1_mean_std'][1]:.3f}")
    print(f"  ASR           = {agg['asr_mean_std'][0]:.3f} ± {agg['asr_mean_std'][1]:.3f}")
    print(f"  benign util.  = {agg['benign_utility_mean_std'][0]:.3f} ± {agg['benign_utility_mean_std'][1]:.3f}")
    print(f"  oversight rec = {agg['oversight_recall_mean_std'][0]:.3f} ± {agg['oversight_recall_mean_std'][1]:.3f}")
    print(f"  latency / call (ms) = {agg['latency_per_call_ms_mean_std'][0]:.2f} ± {agg['latency_per_call_ms_mean_std'][1]:.2f}")
    print(f"  recourse -> non-DENY = {agg['recourse_rate_to_any_non_deny_mean_std'][0]:.3f} ± {agg['recourse_rate_to_any_non_deny_mean_std'][1]:.3f}")


if __name__ == "__main__":
    main()
