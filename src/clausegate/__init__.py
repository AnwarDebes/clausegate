"""clausegate: interpretable verification of LLM agent tool calls.

Submodules:
  - schema:    AgentAction dataclass + canonicalisation + JSON I/O.
  - graphs:    tri-graph (action + context + policy) builder + binarisation.
  - tm:        tri-graph walking HGTM verifier (CPU reference + CUDA wrapper).
  - recourse:  counterfactual argument-edit search.
  - verify:    SAT encoding + HMAC-signed clause receipts.
  - data:      AgentDojo / InjecAgent / R-Judge / synthetic loaders.
  - eval:      metrics + adversarial harness + logger.
  - utils:     seeding + deterministic IO helpers.

Public top-level symbols are re-exported below for ergonomic imports.
"""
from .schema import (
    AgentAction,
    Chunk,
    Policy,
    PolicyRule,
    ToolOutput,
    Verdict,
    canonicalise,
    from_json,
    to_json,
)
from .verify.certificate import (
    ClauseReceipt,
    build_receipt,
    sign,
    verify_receipt,
    verify_signature,
)
from .verify.sat_encoder import (
    FiringClause,
    encode_clauses_to_cnf,
)
from .tm.verifier import (
    TriGraphHGTM,
    PredictionWithClauses,
)
from .recourse.search import (
    ArgumentEdit,
    RecourseOutcome,
    greedy_argument_edit,
)

__version__ = "0.1.0"

__all__ = [
    # schema
    "AgentAction",
    "Chunk",
    "Policy",
    "PolicyRule",
    "ToolOutput",
    "Verdict",
    "canonicalise",
    "from_json",
    "to_json",
    # verify
    "ClauseReceipt",
    "FiringClause",
    "build_receipt",
    "encode_clauses_to_cnf",
    "sign",
    "verify_receipt",
    "verify_signature",
    # tm
    "TriGraphHGTM",
    "PredictionWithClauses",
    # recourse
    "ArgumentEdit",
    "RecourseOutcome",
    "greedy_argument_edit",
]
