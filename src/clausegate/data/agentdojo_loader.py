"""AgentDojo adapter (https://agentdojo.spylab.ai/, arXiv:2406.13352).

AgentDojo (NeurIPS 2024) is the primary training + eval surface
for ClauseGate. It ships as a ``pip install agentdojo`` package
that defines four task suites -- workspace, slack, banking, travel
-- each with user tasks and injection cases.

This module provides a thin adapter that maps an AgentDojo task to
our ``AgentAction`` + label triple. We import ``agentdojo`` lazily
so the package's heavy transformer dep does not block tests that
only need synthetic data.

Adapter contract
----------------
* If ``agentdojo`` is not installed, raise ``ModuleNotFoundError``
  with a helpful message.
* If installed, ``load_agentdojo(suite=..., split=..., n=...)``
  returns ``(actions, labels)`` where labels are
  ``int(Verdict.ALLOW)`` / ``int(Verdict.DENY)`` /
  ``int(Verdict.NEEDS_OVERSIGHT)`` based on AgentDojo's
  injection-presence + tool-call-safety fields.
* The adapter does NOT instantiate an LLM. It walks AgentDojo's
  ground-truth tool-call trajectories.
"""
from __future__ import annotations

from typing import Sequence

from ..schema.actions import AgentAction, Chunk, Policy, ToolOutput, Verdict
from .synth import DEFAULT_POLICY


_DEFAULT_AGENTDOJO_VERSION = "v1.2.1"
_KNOWN_SUITES = ("workspace", "slack", "banking", "travel")


def load_agentdojo(
    suite: str = "workspace",
    split: str = "test",
    n: int | None = None,
    policy: Policy = DEFAULT_POLICY,
    version: str = _DEFAULT_AGENTDOJO_VERSION,
    augment_inject_variants: bool = True,
    max_aug_per_user_task: int = 3,
    augment_hijack_variants: bool = True,
) -> tuple[list[AgentAction], list[int]]:
    """Load AgentDojo tasks as (AgentAction, label) pairs.

    Mapping:
      * ``user_task.ground_truth(env)`` -> ALLOW or NEEDS_OVERSIGHT
        depending on whether the policy denies / requires-oversight
        the tool the benign agent would have called.
      * ``injection_task.ground_truth(env)`` -> DENY; the
        ground-truth call here is what a successfully-injected
        agent would do (i.e., what we want to block).
      * If ``augment_inject_variants`` is True, also emit
        injected variants of each user task with the injection
        task's GOAL added as a low-trust RAG chunk. This gives
        the verifier richer training signal.
    """
    try:
        from agentdojo.benchmark import get_suite  # type: ignore
    except ImportError as e:
        raise ModuleNotFoundError(
            "agentdojo is not installed. Install with: pip install 'clausegate[agentdojo]'"
        ) from e

    if suite not in _KNOWN_SUITES:
        raise ValueError(f"unknown agentdojo suite {suite!r}; pick from {sorted(_KNOWN_SUITES)}")

    ts = get_suite(version, suite)
    env = ts.load_and_inject_default_environment({})

    actions: list[AgentAction] = []
    labels: list[int] = []

    cnt = 0
    # User tasks -> ALLOW / OVERSIGHT
    for ut_id, ut in ts.user_tasks.items():
        try:
            gt = ut.ground_truth(env)
        except Exception:
            continue
        prompt_text = getattr(ut, "PROMPT", "") or ""
        prior_chunks = (
            Chunk(
                chunk_id=f"agentdojo-prompt-{ut_id}",
                text=str(prompt_text)[:800],
                origin="agentdojo:user_prompt",
                trust_level=2,
            ),
        ) if prompt_text else ()
        for call in gt:
            tool = getattr(call, "function", None) or getattr(call, "name", None)
            args = dict(getattr(call, "args", {}) or {})
            if not tool:
                continue
            a = AgentAction(
                tool=str(tool),
                args=args,
                rag_chunks=prior_chunks,
                policy_id=policy.policy_id,
                meta={"source": "agentdojo", "suite": suite, "user_task_id": str(ut_id)},
            )
            actions.append(a)
            labels.append(_label_for_user_task(ut, policy))
            cnt += 1
            if n is not None and cnt >= n:
                return actions, labels

    # Injection tasks -> DENY
    for it_id, it in ts.injection_tasks.items():
        try:
            gt = it.ground_truth(env)
        except Exception:
            continue
        goal = getattr(it, "GOAL", "") or ""
        for call in gt:
            tool = getattr(call, "function", None) or getattr(call, "name", None)
            args = dict(getattr(call, "args", {}) or {})
            if not tool:
                continue
            chunks = (
                Chunk(
                    chunk_id=f"agentdojo-inj-{it_id}",
                    text=str(goal)[:1000],
                    origin="agentdojo:injection",
                    trust_level=0,
                ),
            )
            actions.append(
                AgentAction(
                    tool=str(tool),
                    args=args,
                    rag_chunks=chunks,
                    policy_id=policy.policy_id,
                    meta={"source": "agentdojo", "suite": suite, "injection_task_id": str(it_id)},
                )
            )
            labels.append(int(Verdict.DENY))
            cnt += 1
            if n is not None and cnt >= n:
                return actions, labels

    # Optionally augment with two kinds of poisoned variants:
    # A. POISONED-BENIGN  : user-task call + injection chunk in context
    #                       -> NEEDS_OVERSIGHT (clean call, dirty context)
    # B. HIJACK           : injection-task call + injection chunk + user
    #                       prompt in context -> DENY (hijacked agent)
    # We cap per user_task at max_aug_per_user_task pairings to control
    # the OVERSIGHT inflation.
    if augment_inject_variants or augment_hijack_variants:
        inj_pool = list(ts.injection_tasks.items())
        for ut_id, ut in ts.user_tasks.items():
            try:
                ut_gt = ut.ground_truth(env)
            except Exception:
                continue
            ut_prompt = getattr(ut, "PROMPT", "") or ""

            for it_id, it in inj_pool[:max_aug_per_user_task]:
                goal = getattr(it, "GOAL", "") or ""
                try:
                    it_gt = it.ground_truth(env)
                except Exception:
                    it_gt = []

                # A. POISONED-BENIGN: clean call + injection in context.
                if augment_inject_variants:
                    for call in ut_gt:
                        tool = getattr(call, "function", None)
                        args = dict(getattr(call, "args", {}) or {})
                        if not tool:
                            continue
                        chunks = (
                            Chunk(
                                chunk_id=f"adj-poison-{ut_id}-{it_id}",
                                text=str(goal)[:1000],
                                origin="agentdojo:rag_poison",
                                trust_level=0,
                            ),
                        )
                        if ut_prompt:
                            chunks = chunks + (
                                Chunk(
                                    chunk_id=f"adj-prompt-{ut_id}",
                                    text=str(ut_prompt)[:600],
                                    origin="agentdojo:user_prompt",
                                    trust_level=2,
                                ),
                            )
                        actions.append(
                            AgentAction(
                                tool=str(tool),
                                args=args,
                                rag_chunks=chunks,
                                policy_id=policy.policy_id,
                                meta={
                                    "source": "agentdojo_poisoned_benign",
                                    "suite": suite,
                                    "user_task_id": str(ut_id),
                                    "injection_task_id": str(it_id),
                                },
                            )
                        )
                        labels.append(int(Verdict.NEEDS_OVERSIGHT))
                        cnt += 1
                        if n is not None and cnt >= n:
                            return actions, labels

                # B. HIJACK: malicious call + injection in context.
                if augment_hijack_variants:
                    for call in it_gt:
                        tool = getattr(call, "function", None)
                        args = dict(getattr(call, "args", {}) or {})
                        if not tool:
                            continue
                        chunks = (
                            Chunk(
                                chunk_id=f"adj-hijack-{ut_id}-{it_id}",
                                text=str(goal)[:1000],
                                origin="agentdojo:rag_poison",
                                trust_level=0,
                            ),
                        )
                        if ut_prompt:
                            chunks = chunks + (
                                Chunk(
                                    chunk_id=f"adj-hijack-prompt-{ut_id}",
                                    text=str(ut_prompt)[:600],
                                    origin="agentdojo:user_prompt",
                                    trust_level=2,
                                ),
                            )
                        actions.append(
                            AgentAction(
                                tool=str(tool),
                                args=args,
                                rag_chunks=chunks,
                                policy_id=policy.policy_id,
                                meta={
                                    "source": "agentdojo_hijack",
                                    "suite": suite,
                                    "user_task_id": str(ut_id),
                                    "injection_task_id": str(it_id),
                                },
                            )
                        )
                        labels.append(int(Verdict.DENY))
                        cnt += 1
                        if n is not None and cnt >= n:
                            return actions, labels

    return actions, labels


def _label_for_user_task(ut, policy: Policy) -> int:
    """Derive the verdict label from an AgentDojo user task.

    Heuristic:
      * If the task's first ground-truth tool call has a tool that
        the policy denies -> DENY.
      * If the policy requires oversight -> NEEDS_OVERSIGHT.
      * Otherwise -> ALLOW.

    AgentDojo's user tasks are benign by construction; the labels
    above are sufficient to teach ClauseGate the "policy-tool"
    pattern in the absence of explicit annotations.
    """
    try:
        gt = ut.ground_truth({})  # type: ignore[attr-defined]
    except Exception:
        return int(Verdict.ALLOW)
    if not gt:
        return int(Verdict.ALLOW)
    first = gt[0]
    tool = getattr(first, "function", None) or getattr(first, "name", None)
    if not tool:
        return int(Verdict.ALLOW)
    rule = policy.get(str(tool))
    if rule is None:
        return int(Verdict.NEEDS_OVERSIGHT)
    if rule.permission == "deny":
        return int(Verdict.DENY)
    if rule.permission == "requires_oversight":
        return int(Verdict.NEEDS_OVERSIGHT)
    return int(Verdict.ALLOW)
