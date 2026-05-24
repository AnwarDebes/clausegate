"""Schema unit tests: canonicalisation, JSON round-trip, hashing."""
from __future__ import annotations

import pytest

from clausegate.schema.actions import (
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


def _example() -> AgentAction:
    return AgentAction(
        tool="Send_Email",  # uppercase to test canonicalisation
        args={
            "body": "Hello",
            "to": "alice@company.com",
            "subject": "Status",
            "amount": 42,
        },
        prior_tool_outputs=(
            ToolOutput(call_id="x2", tool="read_email"),
            ToolOutput(call_id="x1", tool="search_web"),
        ),
        rag_chunks=(
            Chunk(chunk_id="c-2", text="Quarterly report.", origin="kb"),
            Chunk(chunk_id="c-1", text="Project status.", origin="kb"),
        ),
        policy_id="default",
        meta={"trace": "abc"},
    )


def test_canonicalise_lowercases_tool_and_sorts():
    a = _example()
    c = canonicalise(a)
    assert c.tool == "send_email"
    assert list(c.args.keys()) == sorted(c.args.keys())
    assert [p.call_id for p in c.prior_tool_outputs] == ["x1", "x2"]
    assert [ch.chunk_id for ch in c.rag_chunks] == ["c-1", "c-2"]


def test_canonicalise_is_idempotent():
    a = _example()
    c1 = canonicalise(a)
    c2 = canonicalise(c1)
    # Hashes equal -> canonicalisation is a fixed point.
    assert c1.hash() == c2.hash()


def test_json_round_trip_is_a_fixed_point():
    a = _example()
    blob = to_json(a)
    a2 = from_json(blob)
    blob2 = to_json(a2)
    assert blob == blob2


def test_hash_is_stable_across_kwarg_permutation():
    a1 = AgentAction(tool="send_email", args={"to": "x@y", "subject": "s"})
    a2 = AgentAction(tool="send_email", args={"subject": "s", "to": "x@y"})
    assert canonicalise(a1).hash() == canonicalise(a2).hash()


def test_verdict_from_str():
    assert Verdict.from_str("allow") == Verdict.ALLOW
    assert Verdict.from_str("DENY") == Verdict.DENY
    assert Verdict.from_str("needs-oversight") == Verdict.NEEDS_OVERSIGHT
    with pytest.raises(ValueError):
        Verdict.from_str("frobnicate")


def test_policy_rule_validates_permission():
    with pytest.raises(ValueError):
        PolicyRule(tool="x", permission="frobnicate")


def test_canonicalise_rejects_unserialisable_value():
    class NotJSON:
        pass

    a = AgentAction(tool="x", args={"k": NotJSON()})
    with pytest.raises(TypeError):
        canonicalise(a)


def test_canonicalise_with_weird_tool_chars_recovers():
    a = AgentAction(tool="  Send/Email!  ", args={})
    c = canonicalise(a)
    # `/` and `!` are stripped to `_` by the fallback path.
    assert "/" not in c.tool and "!" not in c.tool
    assert c.tool.startswith("send_email")
