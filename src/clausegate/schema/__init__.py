"""clausegate.schema: agent-action and policy dataclasses.

Public surface:
    AgentAction   , the proposed tool call + context provenance
    ToolOutput    , an observed prior tool output
    Chunk         , a retrieved RAG chunk (sentence-level)
    Policy        , the operator's allowlist + scope policy
    PolicyRule    , a single (tool, permission, scope) rule
    Verdict       , the enum of {ALLOW, DENY, NEEDS_OVERSIGHT}
    canonicalise  , normalise tool/arg names + sort + fold types
    to_json       , canonical-JSON serialisation
    from_json     , inverse of to_json (asserts round-trip)
"""
from .actions import (
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

__all__ = [
    "AgentAction",
    "Chunk",
    "Policy",
    "PolicyRule",
    "ToolOutput",
    "Verdict",
    "canonicalise",
    "from_json",
    "to_json",
]
