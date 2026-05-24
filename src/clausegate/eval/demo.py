"""Tiny CLI demo: ``python -m clausegate.eval.demo --seed 42``.

Trains a small TriGraphHGTM on the synthetic generator, runs the
adversarial harness, prints a calibrated summary table, and writes a
sample receipt + CNF to ``demo_out/``. Intended for sanity-check + screencast.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..data.synth import DEFAULT_POLICY, SyntheticGenerator
from ..eval.adversarial import AdversarialHarness
from ..eval.logger import PerSampleLogger
from ..eval.metrics import compute_metrics
from ..schema.actions import Verdict
from ..tm.verifier import HGTMSpec, TriGraphHGTM
from ..verify.certificate import build_receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default="demo_out")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[demo] seed={args.seed} out_dir={out_dir.resolve()}")

    gen_train = SyntheticGenerator(seed=args.seed)
    gen_test = SyntheticGenerator(seed=args.seed + 1)
    train_X, train_y = gen_train.batch(args.n_train)
    test_X, test_y = gen_test.batch(args.n_test)

    spec = HGTMSpec(n_clauses=80, T=20, s=3.5, seed=args.seed)
    verifier = TriGraphHGTM(spec=spec)
    print("[demo] training ...")
    verifier.fit(train_X, DEFAULT_POLICY, train_y, epochs=args.epochs)

    print("[demo] evaluating ...")
    preds = verifier.predict_with_clauses(test_X, DEFAULT_POLICY)
    y_pred = [p.verdict for p in preds]
    margins = [p.margin for p in preds]
    rep = compute_metrics(test_y, y_pred, margins=margins)

    print()
    print(f"  acc           = {rep.accuracy:.3f}")
    print(f"  macro-F1      = {rep.macro_f1:.3f}")
    print(f"  ASR           = {rep.asr:.3f}")
    print(f"  benign util.  = {rep.benign_utility:.3f}")
    print(f"  oversight rec = {rep.oversight_recall:.3f}")
    print(f"  ECE (proxy)   = {rep.ece:.3f}")
    print()

    # Adversarial harness
    benign = [a for a, y in zip(test_X, test_y) if y == int(Verdict.ALLOW)][:32]
    if benign:
        harness = AdversarialHarness(verifier, DEFAULT_POLICY, seed=args.seed)
        print("[demo] adversarial harness:")
        for r in harness.run_all(benign):
            print(
                f"  {r.name:<22}  ASR before={r.asr_before:.3f}  "
                f"ASR after={r.asr_after:.3f}  blocked={r.blocked_count}/{r.n}"
            )

    # Write a sample receipt.
    key = os.urandom(32)
    receipt = build_receipt(
        prediction=preds[0],
        signing_key=key,
        receipts_dir=out_dir,
        sample_id="demo",
    )
    receipt.write(out_dir / "demo_receipt.json")
    print(f"[demo] sample receipt -> {out_dir / 'demo_receipt.json'}")
    print(f"[demo] CNF             -> {out_dir / receipt.cnf_path}")

    # Log per-sample.
    with PerSampleLogger(out_dir / "jsonl" / "per_sample.jsonl", append=False) as log:
        for i, p in enumerate(preds):
            log.write(
                {
                    "sample_id": i,
                    "verdict": int(p.verdict),
                    "verdict_name": p.verdict_name,
                    "class_scores": list(p.class_scores),
                    "margin": int(p.margin),
                    "firing_clause_ids": [int(fc.clause_id) for fc in p.firing_clauses],
                    "tri_graph_signature": p.tri_graph_signature,
                    "model_version_id": p.model_version_id,
                }
            )
    print(f"[demo] per-sample log  -> {out_dir / 'jsonl' / 'per_sample.jsonl'}")


if __name__ == "__main__":
    main()
