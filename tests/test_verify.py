"""Tests for SAT encoding + receipt signing."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from clausegate.data.synth import DEFAULT_POLICY, SyntheticGenerator
from clausegate.tm.verifier import HGTMSpec, TriGraphHGTM
from clausegate.verify.certificate import (
    ClauseReceipt,
    build_receipt,
    sign,
    verify_receipt,
    verify_signature,
)
from clausegate.verify.sat_encoder import (
    FiringClause,
    encode_clauses_to_cnf,
    solve_cnf,
)


def test_encode_clauses_to_cnf_minimal(tmp_path: Path):
    fc = FiringClause(
        clause_id=0,
        class_id=0,
        polarity=1,
        weight=1,
        positive_literal_ids=(0, 1),
        negative_literal_ids=(2,),
    )
    cnf = encode_clauses_to_cnf(
        firing_clauses=[fc],
        literal_vector=[1, 1, 0, 1],
        label=0,
        n_classes=3,
        out_dir=tmp_path,
        sample_id="t1",
    )
    assert cnf.is_file()
    body = cnf.read_text()
    assert body.startswith("c ClauseGate")
    # Side-car JSON must exist with firing_clause_ids.
    side = json.loads((tmp_path / "t1.json").read_text())
    assert side["firing_clause_ids"] == [0]
    assert side["label"] == 0


def test_solve_cnf_is_sat_for_consistent_receipt(tmp_path: Path):
    cnf = encode_clauses_to_cnf(
        firing_clauses=[
            FiringClause(
                clause_id=0,
                class_id=0,
                polarity=1,
                weight=1,
                positive_literal_ids=(0,),
                negative_literal_ids=(2,),
            )
        ],
        literal_vector=[1, 1, 0, 1],
        label=0,
        n_classes=3,
        out_dir=tmp_path,
        sample_id="sat1",
    )
    sat, model = solve_cnf(cnf)
    assert sat is True
    assert model is not None


def test_sign_and_verify_signature_round_trip():
    payload = {"a": 1, "b": [1, 2, 3]}
    key = b"\x00" * 32
    sig = sign(payload, key)
    assert verify_signature(payload, sig, key) is True
    assert verify_signature({"a": 1, "b": [1, 2, 4]}, sig, key) is False


def test_sign_requires_nonempty_key():
    with pytest.raises(ValueError):
        sign({"a": 1}, b"")


def test_build_receipt_round_trip(tmp_path: Path):
    gen = SyntheticGenerator(seed=3)
    actions, labels = gen.batch(64)
    spec = HGTMSpec(n_clauses=20, T=5, s=3.0, seed=3)
    verifier = TriGraphHGTM(spec=spec)
    verifier.fit(actions, DEFAULT_POLICY, labels, epochs=2)
    preds = verifier.predict_with_clauses([actions[0]], DEFAULT_POLICY)
    p = preds[0]
    key = b"secretkey" + os.urandom(16)
    receipt = build_receipt(
        prediction=p,
        signing_key=key,
        receipts_dir=tmp_path,
        sample_id="r1",
    )
    assert isinstance(receipt, ClauseReceipt)
    # Signature verifies under the right key, fails under a wrong key.
    assert verify_signature(receipt.fields_for_signing(), receipt.signature, key) is True
    assert verify_signature(receipt.fields_for_signing(), receipt.signature, b"wrong") is False

    # CNF file lives under receipts_dir/cnf/r1.cnf.
    cnf_full = tmp_path / "cnf" / "r1.cnf"
    assert cnf_full.is_file()

    # verify_receipt with signing_key and receipts_dir is happy.
    ok, problems = verify_receipt(receipt, signing_key=key, receipts_dir=tmp_path, run_sat=True)
    assert ok is True, problems


def test_verify_receipt_detects_tampered_field(tmp_path: Path):
    gen = SyntheticGenerator(seed=4)
    actions, labels = gen.batch(16)
    spec = HGTMSpec(n_clauses=10, T=5, s=3.0, seed=4)
    verifier = TriGraphHGTM(spec=spec)
    preds = verifier.predict_with_clauses([actions[0]], DEFAULT_POLICY)
    key = b"k" * 32
    receipt = build_receipt(
        prediction=preds[0],
        signing_key=key,
        receipts_dir=tmp_path,
        sample_id="tamper",
    )
    # Forge a "different" label without re-signing.
    from dataclasses import replace

    forged = replace(receipt, label=99)
    ok, problems = verify_receipt(forged, signing_key=key, receipts_dir=tmp_path)
    assert ok is False
    assert any("HMAC" in m for m in problems)
