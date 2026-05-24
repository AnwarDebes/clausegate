"""InjecAgent adapter (ACL 2024, arXiv:2403.02691).

InjecAgent ships as a small JSON dataset of 1054 single-turn
injection cases. We deliberately do not vendor the dataset; the
loader expects the user to clone or download
``github.com/uiuc-kang-lab/InjecAgent`` and pass the path to the
JSON file.

The adapter maps each case to an ``AgentAction`` with a single
untrusted RAG chunk containing the injection text and ``DENY``
label.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..schema.actions import AgentAction, Chunk, Policy, Verdict
from .synth import DEFAULT_POLICY


def load_injecagent(
    path: str | Path,
    *,
    n: int | None = None,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[list[AgentAction], list[int]]:
    """Load InjecAgent cases from a local JSON path.

    Args:
        path: path to the ``test_cases.json`` (or analogous) file
            from the InjecAgent repository.
        n: optional cap on the number of samples returned.
        policy: policy to attach to every action.

    Returns ``(actions, labels)``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"InjecAgent JSON not found at {p}")
    blob = json.loads(p.read_text())
    if isinstance(blob, dict) and "cases" in blob:
        cases = blob["cases"]
    elif isinstance(blob, list):
        cases = blob
    else:
        raise ValueError(f"unrecognised InjecAgent JSON shape at {p}")

    actions: list[AgentAction] = []
    labels: list[int] = []
    for i, c in enumerate(cases):
        if n is not None and i >= n:
            break
        # InjecAgent uses Capitalized keys in the official release:
        #   "Attacker Tools": [name], "User Tool": name, "Tool Parameters": str-dict,
        #   "Attacker Instruction": text, "Tool Response": injected RAG output.
        attacker_tools = c.get("Attacker Tools") or []
        attacker_tool = attacker_tools[0] if attacker_tools else None
        user_tool = c.get("User Tool")
        tool = (
            c.get("attacker_tool")
            or c.get("user_tool")
            or c.get("tool")
            or attacker_tool
            or user_tool
        )
        # Parameters are stored as a python-repr string in InjecAgent; tolerate
        # both eval-friendly and JSON-friendly shapes.
        params_raw = c.get("attacker_tool_args") or c.get("user_tool_args") or c.get("Tool Parameters") or {}
        if isinstance(params_raw, str):
            try:
                args = json.loads(params_raw.replace("'", '"'))
            except Exception:
                args = {"_raw": params_raw}
        else:
            args = dict(params_raw)
        inj = (
            c.get("attacker_instruction")
            or c.get("injection")
            or c.get("Attacker Instruction")
            or ""
        )
        tool_response = c.get("Tool Response") or c.get("Tool Response Template") or ""
        user_instruction = c.get("User Instruction") or ""

        if not tool:
            continue
        chunks: list[Chunk] = []
        chunks.append(
            Chunk(
                chunk_id=f"injecagent-{i}-inj",
                text=str(inj)[:1000],
                origin="injecagent:attacker_instruction",
                trust_level=0,
            )
        )
        if tool_response:
            chunks.append(
                Chunk(
                    chunk_id=f"injecagent-{i}-resp",
                    text=str(tool_response)[:1000],
                    origin="injecagent:tool_response",
                    trust_level=0,
                )
            )
        if user_instruction:
            chunks.append(
                Chunk(
                    chunk_id=f"injecagent-{i}-user",
                    text=str(user_instruction)[:600],
                    origin="injecagent:user_instruction",
                    trust_level=2,
                )
            )
        actions.append(
            AgentAction(
                tool=str(tool),
                args=dict(args),
                rag_chunks=tuple(chunks),
                policy_id=policy.policy_id,
                meta={
                    "source": "injecagent",
                    "case_id": str(c.get("case_id", i)),
                    "attack_type": c.get("Attack Type", ""),
                },
            )
        )
        labels.append(int(Verdict.DENY))
    return actions, labels
