"""Synthetic data + loader-shape tests."""
from __future__ import annotations

import pytest

from clausegate.data.synth import DEFAULT_POLICY, SyntheticGenerator
from clausegate.schema.actions import Verdict


def test_default_policy_covers_canonical_tools():
    tools = {r.tool for r in DEFAULT_POLICY.rules}
    assert "send_email" in tools
    assert "transfer_funds" in tools
    assert "delete_file" in tools


def test_synthetic_generator_deterministic():
    g1 = SyntheticGenerator(seed=11)
    g2 = SyntheticGenerator(seed=11)
    a1, l1 = g1.batch(20)
    a2, l2 = g2.batch(20)
    assert l1 == l2
    for x, y in zip(a1, a2):
        assert x.tool == y.tool
        assert dict(x.args) == dict(y.args)


def test_synthetic_generator_emits_all_three_classes():
    g = SyntheticGenerator(seed=7)
    _, labels = g.batch(200)
    seen = set(labels)
    # We expect all three classes to appear given the default rates.
    assert int(Verdict.ALLOW) in seen
    assert int(Verdict.DENY) in seen
    assert int(Verdict.NEEDS_OVERSIGHT) in seen


def test_synthetic_generator_rates_validate():
    with pytest.raises(ValueError):
        SyntheticGenerator(seed=0, deny_rate=0.6, oversight_rate=0.6)


def test_agentdojo_loader_raises_when_dep_missing():
    """AgentDojo dep is optional; the loader must say so if missing."""
    from clausegate.data.agentdojo_loader import load_agentdojo

    try:
        import agentdojo  # noqa: F401
    except ImportError:
        with pytest.raises(ModuleNotFoundError):
            load_agentdojo()


def test_injecagent_loader_handles_missing_file():
    from clausegate.data.injecagent_loader import load_injecagent

    with pytest.raises(FileNotFoundError):
        load_injecagent("/nonexistent/path.json")


def test_rjudge_loader_handles_missing_file():
    from clausegate.data.rjudge_loader import load_rjudge

    with pytest.raises(FileNotFoundError):
        load_rjudge("/nonexistent/path.json")
