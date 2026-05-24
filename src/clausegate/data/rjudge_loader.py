"""R-Judge adapter (EMNLP-Findings 2024, arXiv:2401.10019).

R-Judge ships 569 multi-turn agent traces labelled with one of 10
risk types over 27 scenarios. The loader expects the user to have
cloned ``github.com/Lordog/R-Judge`` and to pass the path to the
data file.

We map each trace's *final* proposed tool call to an ``AgentAction``
with the prior turns as ``prior_tool_outputs`` and the user input
+ retrieved documents as ``rag_chunks``. Labels:
  * R-Judge "unsafe" / "risk" -> ``DENY``
  * R-Judge "ambiguous" / "needs review" -> ``NEEDS_OVERSIGHT``
  * R-Judge "safe" -> ``ALLOW``
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema.actions import AgentAction, Chunk, Policy, ToolOutput, Verdict
from .synth import DEFAULT_POLICY


def load_rjudge(
    path: str | Path,
    *,
    n: int | None = None,
    policy: Policy = DEFAULT_POLICY,
) -> tuple[list[AgentAction], list[int]]:
    """Load R-Judge traces from a local JSON path."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"R-Judge JSON not found at {p}")
    items = json.loads(p.read_text())
    if isinstance(items, dict) and "data" in items:
        items = items["data"]
    if not isinstance(items, list):
        raise ValueError(f"unrecognised R-Judge JSON shape at {p}")

    actions: list[AgentAction] = []
    labels: list[int] = []
    for i, item in enumerate(items):
        if n is not None and i >= n:
            break

        # R-Judge canonical shape: ``contents`` is a list of "turns",
        # where each turn is itself a list of role-tagged messages.
        contents = item.get("contents") or item.get("messages") or []
        # ``label`` is 0 (safe) or 1 (unsafe) in the published JSON.
        raw_label = item.get("label", item.get("risk", 0))

        final_tool = None
        final_args: dict = {}
        prior: list[ToolOutput] = []
        chunks: list[Chunk] = []
        # Flatten: each turn -> list of messages OR a single message dict
        flat_messages: list[dict] = []
        for j, turn in enumerate(contents):
            if isinstance(turn, list):
                flat_messages.extend(turn)
            elif isinstance(turn, dict):
                flat_messages.append(turn)

        for j, msg in enumerate(flat_messages):
            role = (msg.get("role") or "").lower()
            text = msg.get("content") or msg.get("text") or msg.get("action") or ""
            if role == "user":
                chunks.append(
                    Chunk(
                        chunk_id=f"rj-{i}-user-{j}",
                        text=str(text)[:800],
                        origin="rjudge:user",
                        trust_level=1,
                    )
                )
            elif role in ("assistant", "agent"):
                # R-Judge encodes the tool call inside the agent's "action"
                # field as a string like "ToolName: {arg-dict-as-str}".
                action_str = msg.get("action") or text
                tool, args = _parse_agent_action(action_str)
                if tool:
                    final_tool = tool
                    final_args = args
            elif role in ("tool", "observation", "function", "environment"):
                out_text = str(text)[:400]
                prior.append(
                    ToolOutput(
                        call_id=f"rj-{i}-tool-{j}",
                        tool=str(msg.get("name") or msg.get("function") or "unknown_tool"),
                        output_summary=out_text,
                        success=True,
                    )
                )
                # Environment outputs are also retrievable context.
                if out_text:
                    chunks.append(
                        Chunk(
                            chunk_id=f"rj-{i}-env-{j}",
                            text=out_text,
                            origin="rjudge:environment",
                            trust_level=1,
                        )
                    )

        if final_tool is None:
            continue
        actions.append(
            AgentAction(
                tool=final_tool,
                args=final_args,
                prior_tool_outputs=tuple(prior),
                rag_chunks=tuple(chunks),
                policy_id=policy.policy_id,
                meta={
                    "source": "rjudge",
                    "case_id": str(item.get("id", i)),
                    "scenario": item.get("scenario", ""),
                    "attack_type": item.get("attack_type", ""),
                },
            )
        )
        # R-Judge's binary label: 1 -> unsafe -> DENY, 0 -> safe -> ALLOW.
        labels.append(_map_label_int(raw_label))
    return actions, labels


def _parse_agent_action(action_str: str) -> tuple[str | None, dict]:
    """Parse R-Judge's "ToolName: {args}" agent action string."""
    if not action_str or not isinstance(action_str, str):
        return None, {}
    # Look for "ToolName:" followed by a JSON-ish dict.
    if ":" not in action_str:
        return None, {}
    tool_name, rest = action_str.split(":", 1)
    tool_name = tool_name.strip()
    rest = rest.strip()
    if not tool_name or not rest:
        return None, {}
    # Trim to the first balanced { ... } so we don't accidentally pull
    # in surrounding prose.
    if rest.startswith("{"):
        depth = 0
        end = -1
        for ix, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = ix
                    break
        if end > 0:
            blob = rest[: end + 1]
            try:
                args = json.loads(blob.replace("'", '"'))
                if isinstance(args, dict):
                    return tool_name, args
            except Exception:
                return tool_name, {"_raw": blob[:500]}
    return tool_name, {"_raw": rest[:500]}


def _map_label_int(label: int | str) -> int:
    if isinstance(label, int):
        return int(Verdict.DENY) if label == 1 else int(Verdict.ALLOW)
    return _map_label(str(label).lower())


def _map_label(s: str) -> int:
    if any(x in s for x in ("unsafe", "risk", "harm")):
        return int(Verdict.DENY)
    if any(x in s for x in ("ambig", "review", "uncertain")):
        return int(Verdict.NEEDS_OVERSIGHT)
    return int(Verdict.ALLOW)
