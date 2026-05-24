"""clausegate.verify: SAT encoding + signed clause receipts.

Public surface:
    FiringClause           , minimal description of a clause for SAT encoding
    encode_clauses_to_cnf  , FiringClause + literal vector -> DIMACS CNF
    solve_cnf              , Glucose-4 wrapper (best-effort; opt dep)
    ClauseReceipt          , signed audit artifact per decision
    build_receipt          , Prediction -> ClauseReceipt
    sign / verify_signature, HMAC-SHA-256 helpers
    verify_receipt         , end-to-end auditor function
"""
from .certificate import (
    ClauseReceipt,
    build_receipt,
    sign,
    verify_receipt,
    verify_signature,
)
from .sat_encoder import FiringClause, encode_clauses_to_cnf, solve_cnf

__all__ = [
    "ClauseReceipt",
    "FiringClause",
    "build_receipt",
    "encode_clauses_to_cnf",
    "sign",
    "solve_cnf",
    "verify_receipt",
    "verify_signature",
]
