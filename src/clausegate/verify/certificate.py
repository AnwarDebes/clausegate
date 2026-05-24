"""Signed clause-receipt construction and verification.

Adapted from ``clauseguard/src/clauseguard/verify/certificate.py``
to the ClauseGate action-graph + context-graph + policy-graph
schema. Receipt format is the same: HMAC-SHA-256 over the
canonicalised JSON, with the signature stored as a hex string in
the same JSON.

The receipt is intentionally short -- a few hundred bytes -- so it
can be embedded in MCP tool-call logs without bloat.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io import canonical_json, safe_write_json, sha256_of
from .sat_encoder import FiringClause, encode_clauses_to_cnf, solve_cnf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ClauseReceipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClauseReceipt:
    """Signed audit artifact for one ClauseGate verification decision.

    The receipt contains everything an external auditor needs:
    * canonical hashes of the action / context / policy graph;
    * the recorded label + per-class scores;
    * the firing-clause ids;
    * the literal-vector hash;
    * a path to the DIMACS CNF (relative to the receipts dir);
    * an HMAC-SHA-256 signature over everything above.

    Threat model:
      * In scope: detect post-hoc tampering with any receipt field.
      * Out of scope: prove the deployer logged a different decision
        than they enforced (delegated to a separate trusted-log
        protocol).
    """

    action_graph_hash: str
    context_graph_hash: str
    policy_graph_hash: str
    tri_graph_signature: str
    model_version_id: str
    label: int
    label_name: str
    class_scores: tuple[int, ...]
    margin: int
    firing_clause_ids: tuple[int, ...]
    literal_vector_hash: str
    cnf_path: str
    vocab_hash: str
    sample_id: str = ""
    receipt_version: str = "clausegate/0.1"
    signature: str = ""

    def fields_for_signing(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("signature", None)
        return d

    def signed(self, key: bytes) -> "ClauseReceipt":
        sig = sign(self.fields_for_signing(), key)
        return replace(self, signature=sig)

    def write(self, path: str | Path) -> Path:
        safe_write_json(path, asdict(self))
        return Path(path)


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def sign(payload: Any, key: bytes) -> str:
    """HMAC-SHA-256 over ``canonical_json(payload)``, hex-encoded."""
    if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
        raise ValueError("signing key must be non-empty bytes")
    msg = canonical_json(payload).encode("ascii")
    return hmac.new(bytes(key), msg, hashlib.sha256).hexdigest()


def verify_signature(payload: Any, signature: str, key: bytes) -> bool:
    """Constant-time HMAC verify."""
    if not signature:
        return False
    want = sign(payload, key)
    return hmac.compare_digest(want, signature)


# ---------------------------------------------------------------------------
# Build receipt
# ---------------------------------------------------------------------------


def build_receipt(
    *,
    prediction: Any,  # ``PredictionWithClauses`` (typed as Any to dodge a circular import)
    signing_key: bytes,
    receipts_dir: str | Path,
    sample_id: str = "sample",
    write_cnf: bool = True,
) -> ClauseReceipt:
    """Build + sign a ClauseReceipt for a single prediction.

    The CNF is written under ``receipts_dir/cnf/``; the receipt
    references it by a relative path so the receipts directory is
    self-contained and portable to an auditor's machine.
    """
    receipts_dir = Path(receipts_dir)
    cnf_dir = receipts_dir / "cnf"
    cnf_dir.mkdir(parents=True, exist_ok=True)

    bf = prediction.features
    firing = list(prediction.firing_clauses)

    if write_cnf:
        cnf_path = encode_clauses_to_cnf(
            firing_clauses=firing,
            literal_vector=bf.literal_vector.tolist(),
            label=int(prediction.verdict),
            n_classes=len(prediction.class_scores),
            out_dir=cnf_dir,
            sample_id=sample_id,
        )
        cnf_rel = str(cnf_path.relative_to(receipts_dir))
    else:
        cnf_rel = ""

    # Pull the underlying graph hashes off the prediction object.
    receipt = ClauseReceipt(
        action_graph_hash=getattr(prediction, "action_graph_hash", "") or "",
        context_graph_hash=getattr(prediction, "context_graph_hash", "") or "",
        policy_graph_hash=getattr(prediction, "policy_graph_hash", "") or "",
        tri_graph_signature=prediction.tri_graph_signature,
        model_version_id=prediction.model_version_id,
        label=int(prediction.verdict),
        label_name=prediction.verdict_name,
        class_scores=tuple(int(s) for s in prediction.class_scores),
        margin=int(prediction.margin),
        firing_clause_ids=tuple(int(fc.clause_id) for fc in firing),
        literal_vector_hash=bf.literal_hash(),
        cnf_path=cnf_rel,
        vocab_hash=bf.vocab_hash,
        sample_id=sample_id,
    )
    return receipt.signed(signing_key)


def _graph_hash_from_sig(prediction: Any, which: str) -> str:
    """Recover sub-graph hash from PredictionWithClauses when available.

    The TriGraph signature is a hash of the four sub-hashes, so we
    cannot recover the originals from the signature alone. The
    upstream verifier attaches them as plain attributes when it
    produces the prediction; absent those, we leave the field empty
    and rely on the SAT auditor to detect the inconsistency.
    """
    return ""


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify_receipt(
    receipt: ClauseReceipt,
    *,
    signing_key: bytes | None = None,
    receipts_dir: str | Path | None = None,
    run_sat: bool = False,
) -> tuple[bool, list[str]]:
    """End-to-end auditor verification.

    Checks (in order):
      1. HMAC signature (if ``signing_key`` is provided).
      2. CNF file present at ``receipts_dir / receipt.cnf_path``.
      3. (optional) SAT solve returns SAT on the recorded CNF.

    Returns ``(ok, problems)`` where ``problems`` is a list of
    human-readable error messages.
    """
    problems: list[str] = []
    if signing_key is not None:
        ok = verify_signature(receipt.fields_for_signing(), receipt.signature, signing_key)
        if not ok:
            problems.append("HMAC signature does not verify")
    if receipts_dir is not None and receipt.cnf_path:
        cnf_full = Path(receipts_dir) / receipt.cnf_path
        if not cnf_full.is_file():
            problems.append(f"CNF file not found at {cnf_full}")
        elif run_sat:
            sat, _model = solve_cnf(cnf_full)
            if not sat:
                problems.append("Recorded CNF is UNSAT (receipt internally inconsistent)")
    return (len(problems) == 0), problems
