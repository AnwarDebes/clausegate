"""Agent-action and policy dataclasses.

Design rules
------------
1. Every dataclass is ``frozen=True`` so an AgentAction can be
   hashed and used as a dict key. Mutating an action means making
   a new one.
2. ``canonicalise`` is idempotent (running it twice produces the
   same object) and it must be deterministic across runs.
3. ``to_json`` always emits ``canonical_json`` (sorted keys,
   fixed separators) so a (canonicalised, serialised) action has a
   stable hash that the receipt module can reuse.
4. The schema deliberately avoids encoding LLM-specific details
   (no model name, no temperature). ClauseGate is the verifier of
   the *action*, not the auditor of the LLM that emitted it.

The schema is purposefully a strict superset of what the AgentDojo
benchmark needs (tool name + kwargs + observed tool outputs +
retrieved RAG chunks); other benchmarks (InjecAgent, R-Judge) and
production MCP servers can subset it cleanly.
"""
from __future__ import annotations

import enum
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..utils.io import canonical_json, sha256_of

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Verdict(enum.IntEnum):
    """Verifier verdict on a proposed tool call.

    Integer ordering is meaningful; the SAT encoder uses it to
    derive the "label is the unique argmax" constraint.
    """

    ALLOW = 0
    DENY = 1
    NEEDS_OVERSIGHT = 2

    @classmethod
    def from_str(cls, s: str) -> "Verdict":
        s = s.strip().lower().replace("-", "_")
        if s in ("allow", "permit", "ok"):
            return cls.ALLOW
        if s in ("deny", "refuse", "block"):
            return cls.DENY
        if s in ("needs_oversight", "oversight", "review", "human"):
            return cls.NEEDS_OVERSIGHT
        raise ValueError(f"unknown verdict string {s!r}")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A retrieved RAG / search chunk the agent observed.

    Attributes:
        chunk_id: stable id from the retriever; used to deduplicate.
        text: the chunk's sentence-level text. We do not store the
            full document, only the chunk the agent attended to.
        origin: free-form provenance string (e.g., "email:user@x"
            or "url:https://example.com/page#para-3"). Critical for
            the ``injected-from`` cross-graph edge.
        trust_level: integer in {0, 1, 2, 3}; 0 = untrusted
            (user-controlled webpage / email), 3 = first-party
            (operator-owned KB).
    """

    chunk_id: str
    text: str
    origin: str = ""
    trust_level: int = 0


@dataclass(frozen=True)
class ToolOutput:
    """A prior tool output observed by the agent.

    Used by both (a) the context graph (the agent saw this output
    and may be acting on it) and (b) the action graph
    (``chained-from-prev-tool`` edge from the current call to the
    prior call).
    """

    call_id: str
    tool: str
    output_summary: str = ""
    success: bool = True


@dataclass(frozen=True)
class PolicyRule:
    """A single rule from the operator's policy.

    permission ∈ {"allow", "deny", "requires_oversight"}
    scope is a fold of {"public", "internal", "secret", "any"}
      plus extra free-form tags for the deployer to extend.
    """

    tool: str
    permission: str = "allow"
    scope: str = "any"
    sensitivity_label: str = "public"

    def __post_init__(self) -> None:
        if self.permission not in ("allow", "deny", "requires_oversight"):
            raise ValueError(f"unknown permission {self.permission!r}")


@dataclass(frozen=True)
class Policy:
    """The operator's policy as a frozen rule set.

    Rules are matched by tool name; the first match wins. A tool
    not in the policy defaults to ``deny`` (closed-world policy).
    """

    rules: tuple[PolicyRule, ...] = ()
    policy_id: str = "default"

    def get(self, tool: str) -> PolicyRule | None:
        for r in self.rules:
            if r.tool == tool:
                return r
        return None

    def hash(self) -> str:
        return sha256_of({"policy_id": self.policy_id, "rules": [r.__dict__ for r in self.rules]})


# ---------------------------------------------------------------------------
# AgentAction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentAction:
    """The agent's proposed tool call + everything we need to verify it.

    Attributes:
        tool: tool name; lowercase, alnum + ``_-`` after canonicalisation.
        args: kwargs to the tool; values may be str / int / float /
            bool / None / nested list/dict of the same. Canonicalisation
            asserts JSON-serialisability.
        prior_tool_outputs: tuple of outputs the agent observed
            before proposing this call. Order is execution order.
        rag_chunks: tuple of retrieved RAG chunks the agent had
            in context. Order is retrieval order.
        policy_id: id of the policy under which this call is
            being verified; must match a ``Policy.policy_id``.
        meta: free-form metadata that does NOT enter the
            canonicalisation hash. Useful for benchmark-source ids.
    """

    tool: str
    args: Mapping[str, Any] = field(default_factory=dict)
    prior_tool_outputs: tuple[ToolOutput, ...] = ()
    rag_chunks: tuple[Chunk, ...] = ()
    policy_id: str = "default"
    meta: Mapping[str, Any] = field(default_factory=dict)

    # ---- hashing surface (omits meta by design) ----
    def hash(self) -> str:
        return sha256_of(self._hashable())

    def _hashable(self) -> dict:
        return {
            "tool": self.tool,
            "args": _stable(self.args),
            "prior_tool_outputs": [t.__dict__ for t in self.prior_tool_outputs],
            "rag_chunks": [c.__dict__ for c in self.rag_chunks],
            "policy_id": self.policy_id,
        }


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

_TOOL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-\.]*$")


def _canonical_tool_name(tool: str) -> str:
    """Lowercase + strip + collapse-whitespace tool name."""
    t = tool.strip().lower()
    # Collapse internal whitespace + replace spaces with underscores.
    t = re.sub(r"\s+", "_", t)
    if not t:
        raise ValueError("tool name is empty after canonicalisation")
    if not _TOOL_NAME_RE.match(t):
        # Tolerant fallback for legacy benchmarks: replace odd chars with '_'.
        t = re.sub(r"[^a-z0-9_\-\.]", "_", t)
        if not _TOOL_NAME_RE.match(t):
            raise ValueError(f"tool name {tool!r} cannot be canonicalised")
    return t


def _canonical_value(v: Any) -> Any:
    """Recursively canonicalise an arg value.

    - dicts are recursed and key-sorted on serialisation;
    - lists/tuples are recursed and stored as lists;
    - sets/frozensets are sorted-listed;
    - strings are stripped of trailing whitespace;
    - everything else passes through.
    """
    if isinstance(v, dict):
        return {k: _canonical_value(v[k]) for k in sorted(v.keys())}
    if isinstance(v, (list, tuple)):
        return [_canonical_value(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted([_canonical_value(x) for x in v], key=lambda x: canonical_json(x))
    if isinstance(v, str):
        return v.rstrip()
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    # Last-ditch: try JSON round-trip.
    try:
        return json.loads(json.dumps(v))
    except TypeError as e:
        raise TypeError(f"cannot canonicalise value of type {type(v).__name__}") from e


def _stable(args: Mapping[str, Any]) -> dict[str, Any]:
    """Return a sorted-key dict where every value is canonicalised."""
    return {k: _canonical_value(args[k]) for k in sorted(args.keys())}


def canonicalise(action: AgentAction) -> AgentAction:
    """Normalise an AgentAction for stable hashing.

    Operations:
      - lowercase / regex-sanitise tool name;
      - canonicalise args (sorted keys + value normalisation);
      - sort prior tool outputs by call_id;
      - sort RAG chunks by chunk_id;
      - drop meta from the hash but preserve it on the returned object;
      - keep policy_id verbatim.

    Idempotent: ``canonicalise(canonicalise(x)) == canonicalise(x)``.
    """
    tool = _canonical_tool_name(action.tool)
    args = _stable(action.args)
    prior = tuple(sorted(action.prior_tool_outputs, key=lambda t: t.call_id))
    chunks = tuple(sorted(action.rag_chunks, key=lambda c: c.chunk_id))
    return AgentAction(
        tool=tool,
        args=args,
        prior_tool_outputs=prior,
        rag_chunks=chunks,
        policy_id=action.policy_id,
        meta=dict(action.meta),
    )


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


def to_json(action: AgentAction) -> str:
    """Serialise an AgentAction to canonical JSON.

    Output is bytewise-stable: same canonical input → same string.
    Use ``from_json`` to parse back; round-tripping after
    ``canonicalise`` is a fixed point.
    """
    canonical = canonicalise(action)
    blob = {
        "tool": canonical.tool,
        "args": canonical.args,
        "prior_tool_outputs": [t.__dict__ for t in canonical.prior_tool_outputs],
        "rag_chunks": [c.__dict__ for c in canonical.rag_chunks],
        "policy_id": canonical.policy_id,
        "meta": _stable(canonical.meta),
    }
    return canonical_json(blob)


def from_json(blob: str) -> AgentAction:
    """Inverse of ``to_json``.

    Parses the JSON, reconstructs nested dataclasses, and returns
    a canonicalised AgentAction. The caller can compare hashes
    against the original for round-trip checks.
    """
    obj = json.loads(blob)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object at top level")
    prior = tuple(ToolOutput(**t) for t in obj.get("prior_tool_outputs", []))
    chunks = tuple(Chunk(**c) for c in obj.get("rag_chunks", []))
    return canonicalise(
        AgentAction(
            tool=obj["tool"],
            args=obj.get("args", {}),
            prior_tool_outputs=prior,
            rag_chunks=chunks,
            policy_id=obj.get("policy_id", "default"),
            meta=obj.get("meta", {}),
        )
    )
