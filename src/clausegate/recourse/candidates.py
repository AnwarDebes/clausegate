"""Clause-driven candidate generation for ClauseGate recourse.

For a proposed tool call predicted with verdict ``c``, this module
emits a bounded set of candidate ``ArgumentEdit`` operations that
touch the *symbols currently deciding* the verdict. The design
mirrors graphtm-cbr's clause-attributed candidate generator, but
operates in argument-edit space rather than graph-edit space.

Strategy
--------
1. For the verdict's firing clauses, walk the (positive, negative)
   literal sets and map each literal id back to a symbol label
   (from ``BinaryFeatures.symbol_labels``).
2. For each symbol label that came from an arg-value slot, emit
   candidate edits that **clear** (remove substring), **clip**
   (clamp numeric below a threshold), **replace** (string swap),
   or **drop** (remove kwarg) the arg.
3. For each cross-graph ``injected-from`` edge implicated, emit a
   candidate to **remove the offending RAG chunk** from the
   context.
4. For each policy-graph denial edge implicated, emit a candidate
   to **lift policy** (relax the rule).

No BFS, no 2^k enumeration. The candidate set is upper-bounded by
``max_candidates`` and seeded by the structure of the clauses that
actually fired on the proposed call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace as dc_replace
from typing import Any, Iterable, Mapping, Sequence

from ..schema.actions import AgentAction, Chunk, Policy, PolicyRule, ToolOutput
from ..verify.sat_encoder import FiringClause


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


VALID_OPS = (
    "clear_substring",
    "clip_numeric",
    "replace_string",
    "drop_arg",
    "remove_chunk",
    "lift_policy",
)


@dataclass(frozen=True)
class ArgumentEdit:
    """A single edit on an AgentAction or its accompanying Policy.

    op:
      - ``clear_substring``: remove a substring from a string arg value.
        target = arg_name, payload = substring to delete.
      - ``clip_numeric``: clamp a numeric arg below a threshold.
        target = arg_name, payload = float threshold.
      - ``replace_string``: swap a substring inside a string arg value.
        target = arg_name, payload = (old, new).
      - ``drop_arg``: remove the kwarg entirely.
        target = arg_name, payload = None.
      - ``remove_chunk``: drop a RAG chunk from the action's context.
        target = chunk_id, payload = None.
      - ``lift_policy``: relax a policy rule from deny/oversight to allow.
        target = tool name, payload = new permission ("allow" / "requires_oversight").
    """

    op: str
    target: str
    payload: Any = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"unknown op {self.op!r}")


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


_INJECT_SUBSTRINGS = (
    "BCC:attacker@",
    "BCC:secret@",
    "ignore previous",
    "system:",
    "<script",
    "javascript:",
    "data:text",
    "file://",
    "../",
    "%2e%2e",
    "drop table",
    "rm -rf",
    "DELETE",
)


def candidates_from_clauses(
    action: AgentAction,
    policy: Policy,
    firing_clauses: Sequence[FiringClause],
    symbol_labels: Sequence[str],
    *,
    max_candidates: int = 64,
) -> list[ArgumentEdit]:
    """Emit candidate edits implicated by the firing clauses.

    Pre-condition: ``action`` is canonicalised; ``symbol_labels``
    is the per-literal label produced by ``graphs.binarize``.
    """
    candidates: list[ArgumentEdit] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(edit: ArgumentEdit) -> None:
        key = (edit.op, edit.target, str(edit.payload))
        if key in seen:
            return
        seen.add(key)
        candidates.append(edit)

    # Aggregate literal ids implicated by the firing clauses.
    implicated_pos: set[int] = set()
    implicated_neg: set[int] = set()
    for fc in firing_clauses:
        implicated_pos.update(fc.positive_literal_ids)
        implicated_neg.update(fc.negative_literal_ids)

    # Walk implicated symbols (positive includes -- the literal that
    # was 1 and the clause used it as a precondition for firing).
    for lit_id in implicated_pos:
        if lit_id >= len(symbol_labels):
            continue
        label = symbol_labels[lit_id]
        # ``string-contains:<sub>`` literal -> clear that substring.
        if label.startswith("string-contains:"):
            sub = label[len("string-contains:"):]
            arg_name = _find_arg_containing(action.args, sub)
            if arg_name is not None:
                _add(
                    ArgumentEdit(
                        op="clear_substring",
                        target=arg_name,
                        payload=sub,
                        rationale=f"firing clause cited {label}",
                    )
                )
                _add(
                    ArgumentEdit(
                        op="drop_arg",
                        target=arg_name,
                        payload=None,
                        rationale=f"dropping arg {arg_name} containing {sub!r}",
                    )
                )
        # ``thermometer:>=<t>`` literal at value 1 -> clip numeric below t.
        elif label.startswith("thermometer:>="):
            try:
                t = float(label[len("thermometer:>="):])
            except ValueError:
                continue
            # Find a numeric arg whose value triggers this threshold.
            for k, v in action.args.items():
                if isinstance(v, (int, float)) and float(v) >= t:
                    _add(
                        ArgumentEdit(
                            op="clip_numeric",
                            target=k,
                            payload=max(0.0, t - 1.0),
                            rationale=f"clip below threshold {t}",
                        )
                    )
        # ``cross-edge:injected-from`` literal -> remove the implicated chunk.
        elif label == "cross-edge:injected-from":
            for ch in action.rag_chunks:
                if ch.trust_level == 0:
                    _add(
                        ArgumentEdit(
                            op="remove_chunk",
                            target=ch.chunk_id,
                            payload=None,
                            rationale=f"chunk {ch.chunk_id} is untrusted-origin",
                        )
                    )
        # ``policy-edge:denies`` literal -> lift the policy.
        elif label == "policy-edge:denies":
            r = policy.get(action.tool)
            if r is not None:
                _add(
                    ArgumentEdit(
                        op="lift_policy",
                        target=action.tool,
                        payload="requires_oversight",
                        rationale="lift deny to oversight",
                    )
                )

    # Always also offer "scan-for-known-injection-strings" candidates.
    for sub in _INJECT_SUBSTRINGS:
        for k, v in action.args.items():
            if isinstance(v, str) and sub.lower() in v.lower():
                _add(
                    ArgumentEdit(
                        op="clear_substring",
                        target=k,
                        payload=sub,
                        rationale=f"known-injection-string {sub}",
                    )
                )

    # And "remove every untrusted chunk" as a fallback bulk option.
    for ch in action.rag_chunks:
        if ch.trust_level == 0:
            _add(
                ArgumentEdit(
                    op="remove_chunk",
                    target=ch.chunk_id,
                    payload=None,
                    rationale="fallback: drop untrusted chunk",
                )
            )

    # Unconditional fallback: if the policy denies / requires oversight
    # for the action's tool, always offer to lift it. The clause-walk
    # above only emits this when the "policy-edge:denies" literal happens
    # to be in a firing clause, which after a few greedy steps may not
    # be true any more even though the policy is still the blocker.
    pol_rule = policy.get(action.tool)
    if pol_rule is not None and pol_rule.permission != "allow":
        _add(
            ArgumentEdit(
                op="lift_policy",
                target=action.tool,
                payload="allow",
                rationale="fallback: lift restrictive policy",
            )
        )
        _add(
            ArgumentEdit(
                op="lift_policy",
                target=action.tool,
                payload="requires_oversight",
                rationale="fallback: relax deny to oversight",
            )
        )

    # Unconditional fallback: every numeric arg gets a clip candidate.
    for k, v in action.args.items():
        if isinstance(v, (int, float)) and float(v) > 0:
            _add(
                ArgumentEdit(
                    op="clip_numeric",
                    target=k,
                    payload=0.0,
                    rationale=f"fallback: clip {k} to zero",
                )
            )

    # Unconditional fallback: drop each kwarg.
    for k in list(action.args.keys()):
        _add(
            ArgumentEdit(
                op="drop_arg",
                target=k,
                payload=None,
                rationale=f"fallback: drop {k}",
            )
        )

    return candidates[:max_candidates]


def _find_arg_containing(args: Mapping[str, Any], substring: str) -> str | None:
    needle = substring.lower()
    for k, v in args.items():
        if isinstance(v, str) and needle in v.lower():
            return k
    return None


# ---------------------------------------------------------------------------
# Edit application
# ---------------------------------------------------------------------------


def apply_edit(
    action: AgentAction,
    policy: Policy,
    edit: ArgumentEdit,
) -> tuple[AgentAction, Policy]:
    """Return ``(new_action, new_policy)`` with ``edit`` applied.

    Pure: never mutates inputs.
    """
    if edit.op == "clear_substring":
        if edit.target not in action.args:
            return action, policy
        v = action.args[edit.target]
        if not isinstance(v, str):
            return action, policy
        # Case-insensitive substring removal but keep original casing context.
        sub = str(edit.payload)
        new_v = re.sub(re.escape(sub), "", v, flags=re.IGNORECASE)
        new_args = dict(action.args)
        new_args[edit.target] = new_v
        return _with_args(action, new_args), policy

    if edit.op == "clip_numeric":
        if edit.target not in action.args:
            return action, policy
        v = action.args[edit.target]
        if not isinstance(v, (int, float)):
            return action, policy
        thresh = float(edit.payload)
        new_args = dict(action.args)
        new_args[edit.target] = thresh
        return _with_args(action, new_args), policy

    if edit.op == "replace_string":
        if edit.target not in action.args:
            return action, policy
        v = action.args[edit.target]
        if not isinstance(v, str):
            return action, policy
        old, new = edit.payload
        new_args = dict(action.args)
        new_args[edit.target] = v.replace(old, new)
        return _with_args(action, new_args), policy

    if edit.op == "drop_arg":
        new_args = {k: action.args[k] for k in action.args if k != edit.target}
        return _with_args(action, new_args), policy

    if edit.op == "remove_chunk":
        new_chunks = tuple(c for c in action.rag_chunks if c.chunk_id != edit.target)
        return _with_chunks(action, new_chunks), policy

    if edit.op == "lift_policy":
        new_rules = []
        for r in policy.rules:
            if r.tool == edit.target:
                new_rules.append(
                    PolicyRule(
                        tool=r.tool,
                        permission=str(edit.payload),
                        scope=r.scope,
                        sensitivity_label=r.sensitivity_label,
                    )
                )
            else:
                new_rules.append(r)
        return action, Policy(rules=tuple(new_rules), policy_id=policy.policy_id)

    raise ValueError(f"unknown op {edit.op!r}")


def _with_args(action: AgentAction, new_args: Mapping[str, Any]) -> AgentAction:
    return AgentAction(
        tool=action.tool,
        args=dict(new_args),
        prior_tool_outputs=action.prior_tool_outputs,
        rag_chunks=action.rag_chunks,
        policy_id=action.policy_id,
        meta=dict(action.meta),
    )


def _with_chunks(action: AgentAction, new_chunks: tuple[Chunk, ...]) -> AgentAction:
    return AgentAction(
        tool=action.tool,
        args=dict(action.args),
        prior_tool_outputs=action.prior_tool_outputs,
        rag_chunks=tuple(new_chunks),
        policy_id=action.policy_id,
        meta=dict(action.meta),
    )
