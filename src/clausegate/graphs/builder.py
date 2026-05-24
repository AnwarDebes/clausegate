"""Tri-graph builder: (AgentAction, Policy) -> TriGraph.

The tri-graph is the canonical structured representation a clause
walks over. It contains three coupled typed-edge MultiDiGraphs and
the cross-graph edges that make prompt-injection patterns
syntactically visible to a clause.

Graphs
------
* **Action graph** -- the proposed tool call decomposed.
  Node types: ``tool``, ``arg_name``, ``arg_value_literal``,
              ``arg_value_symbol``.
  Edge types: ``has-arg``, ``has-type``, ``binds-to``,
              ``chained-from-prev-tool``,
              ``cites-context-chunk-id``.

* **Context graph** -- retrieved RAG chunks + prior tool outputs.
  Node types: ``chunk``, ``sentence``, ``entity``,
              ``prior_tool_output``, ``origin``.
  Edge types: ``contains``, ``cites``, ``quotes``,
              ``derived-from``, ``provenance``.

* **Policy graph** -- operator's allowlist + scope policy.
  Node types: ``tool``, ``permission``, ``scope``,
              ``sensitivity_label``.
  Edge types: ``allows``, ``denies``, ``requires-oversight``,
              ``scope-of``.

* **Cross-graph edges** -- action <-> context <-> policy:
              ``grounds-in``       (action arg value -> context chunk),
              ``permitted-by``     (action tool -> policy tool),
              ``injected-from``    (context chunk -> action arg
                                    -- emitted only by adversarial
                                    fixtures that flag a chunk as
                                    ``trust_level == 0`` and a chunk
                                    text shares a literal substring
                                    with an arg value).

Implementation notes
--------------------
We use ``networkx.MultiDiGraph`` because (a) clauses must distinguish
multiple edges between the same node pair (e.g., an arg both
``grounds-in`` and ``injected-from`` the same chunk), and (b)
direction is meaningful (a ``cites-context-chunk-id`` edge points
from the arg to the chunk, not the reverse).

Node ids are deterministic strings: ``"<graph-prefix>:<scope>:<token>"``.
The prefix is one of {``a``, ``c``, ``p``} for action / context /
policy. The scope distinguishes sub-categories (e.g., ``a:tool`` vs
``a:arg``). The token is the canonical literal or symbol. This
encoding makes graph identity hashable and human-readable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import networkx as nx

from ..schema.actions import AgentAction, Chunk, Policy, PolicyRule, ToolOutput
from ..utils.io import canonical_json, sha256_of


# ---------------------------------------------------------------------------
# TriGraph dataclass
# ---------------------------------------------------------------------------


@dataclass
class TriGraph:
    """The three coupled graphs + the cross-graph edges + their hashes.

    ``cross`` is an undirected list of ``(src_node, dst_node, edge_type)``
    tuples; we keep it as a list rather than a fourth MultiDiGraph so
    the receipt can hash it independently of NetworkX internals.
    """

    action: nx.MultiDiGraph
    context: nx.MultiDiGraph
    policy: nx.MultiDiGraph
    cross: list[tuple[str, str, str]] = field(default_factory=list)

    # Hashes are populated by ``_finalise``.
    action_hash: str = ""
    context_hash: str = ""
    policy_hash: str = ""
    cross_hash: str = ""

    def signature(self) -> str:
        """Stable signature combining all four hashes; used in receipts."""
        return sha256_of(
            {
                "action": self.action_hash,
                "context": self.context_hash,
                "policy": self.policy_hash,
                "cross": self.cross_hash,
            }
        )


# ---------------------------------------------------------------------------
# Node id helpers
# ---------------------------------------------------------------------------


def _node(prefix: str, scope: str, token: Any) -> str:
    token_s = str(token).replace(":", "_")
    return f"{prefix}:{scope}:{token_s}"


def _value_token(v: Any) -> str:
    """Render an arg value to a stable string token."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return str(v)
    return canonical_json(v)


# ---------------------------------------------------------------------------
# Builders for each sub-graph
# ---------------------------------------------------------------------------


def _build_action_graph(action: AgentAction) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    tool_node = _node("a", "tool", action.tool)
    g.add_node(tool_node, node_type="tool", token=action.tool)

    for arg_name, arg_value in action.args.items():
        arg_node = _node("a", "arg", arg_name)
        g.add_node(arg_node, node_type="arg_name", token=arg_name)
        g.add_edge(tool_node, arg_node, edge_type="has-arg")

        token = _value_token(arg_value)
        if isinstance(arg_value, (int, float, bool)) or arg_value is None:
            value_node = _node("a", "val_lit", f"{arg_name}={token}")
            g.add_node(
                value_node,
                node_type="arg_value_literal",
                token=token,
                arg_name=arg_name,
                python_type=type(arg_value).__name__,
            )
            g.add_edge(arg_node, value_node, edge_type="binds-to")
            g.add_edge(value_node, arg_node, edge_type="has-type")
        else:
            # String / list / dict -> symbol node, token = the value.
            value_node = _node("a", "val_sym", f"{arg_name}={sha256_of(token)[:12]}")
            g.add_node(
                value_node,
                node_type="arg_value_symbol",
                token=token,
                arg_name=arg_name,
                python_type=type(arg_value).__name__,
            )
            g.add_edge(arg_node, value_node, edge_type="binds-to")
            g.add_edge(value_node, arg_node, edge_type="has-type")

    for prior in action.prior_tool_outputs:
        prior_node = _node("a", "prev_tool", prior.tool)
        g.add_node(prior_node, node_type="prev_tool", token=prior.tool, call_id=prior.call_id)
        g.add_edge(tool_node, prior_node, edge_type="chained-from-prev-tool")

    for ch in action.rag_chunks:
        ref_node = _node("a", "ctx_ref", ch.chunk_id)
        g.add_node(ref_node, node_type="context_ref", chunk_id=ch.chunk_id)
        g.add_edge(tool_node, ref_node, edge_type="cites-context-chunk-id")

    return g


def _build_context_graph(action: AgentAction) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for ch in action.rag_chunks:
        chunk_node = _node("c", "chunk", ch.chunk_id)
        g.add_node(
            chunk_node,
            node_type="chunk",
            token=ch.chunk_id,
            trust_level=ch.trust_level,
            origin=ch.origin,
        )
        for sent in _split_sentences(ch.text):
            sent_node = _node("c", "sent", f"{ch.chunk_id}#{sha256_of(sent)[:8]}")
            g.add_node(sent_node, node_type="sentence", token=sent)
            g.add_edge(chunk_node, sent_node, edge_type="contains")
        if ch.origin:
            origin_node = _node("c", "origin", ch.origin)
            g.add_node(origin_node, node_type="origin", token=ch.origin)
            g.add_edge(chunk_node, origin_node, edge_type="provenance")

    for prior in action.prior_tool_outputs:
        prior_node = _node("c", "prev_out", prior.call_id)
        g.add_node(
            prior_node,
            node_type="prior_tool_output",
            token=prior.tool,
            call_id=prior.call_id,
            success=prior.success,
        )
        if prior.output_summary:
            sum_node = _node("c", "prev_sum", f"{prior.call_id}#sum")
            g.add_node(sum_node, node_type="sentence", token=prior.output_summary)
            g.add_edge(prior_node, sum_node, edge_type="contains")

    return g


def _build_policy_graph(policy: Policy) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node(_node("p", "policy", policy.policy_id), node_type="policy", token=policy.policy_id)

    for rule in policy.rules:
        tool_node = _node("p", "tool", rule.tool)
        g.add_node(tool_node, node_type="tool", token=rule.tool)

        perm_node = _node("p", "perm", f"{rule.tool}->{rule.permission}")
        g.add_node(perm_node, node_type="permission", token=rule.permission)

        scope_node = _node("p", "scope", f"{rule.tool}->{rule.scope}")
        g.add_node(scope_node, node_type="scope", token=rule.scope)

        sens_node = _node("p", "sens", f"{rule.tool}->{rule.sensitivity_label}")
        g.add_node(sens_node, node_type="sensitivity_label", token=rule.sensitivity_label)

        # Edge type encodes the rule's permission directly so a clause
        # can pattern-match {allows, denies, requires-oversight} cheaply.
        edge_type = {
            "allow": "allows",
            "deny": "denies",
            "requires_oversight": "requires-oversight",
        }[rule.permission]
        g.add_edge(tool_node, perm_node, edge_type=edge_type)
        g.add_edge(tool_node, scope_node, edge_type="scope-of")
        g.add_edge(tool_node, sens_node, edge_type="scope-of")
    return g


# ---------------------------------------------------------------------------
# Cross-graph edges
# ---------------------------------------------------------------------------


_CROSS_INJECTION_NEEDLES = (
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


def _build_cross_edges(
    action: AgentAction,
    policy: Policy,
    a_g: nx.MultiDiGraph,
    c_g: nx.MultiDiGraph,
    p_g: nx.MultiDiGraph,
) -> list[tuple[str, str, str]]:
    cross: list[tuple[str, str, str]] = []

    # 1. permitted-by: action tool -> policy tool (if rule exists)
    if policy.get(action.tool) is not None:
        cross.append(
            (
                _node("a", "tool", action.tool),
                _node("p", "tool", action.tool),
                "permitted-by",
            )
        )

    # 2. grounds-in + injected-from
    # For every arg whose canonical string value shares a substantive
    # substring with a chunk's text (either the full arg value or a
    # known-injection needle), emit a "grounds-in" edge from the
    # arg-value node to the chunk node. If the chunk's trust_level is
    # 0 (untrusted), the SAME pair gets an "injected-from" edge instead.
    for arg_name, arg_value in action.args.items():
        if not isinstance(arg_value, str):
            continue
        token = _value_token(arg_value)
        if len(token) < 3:
            continue
        for ch in action.rag_chunks:
            shared = _shared_signal(token, ch.text)
            if not shared:
                continue
            src = _node("a", "val_sym", f"{arg_name}={sha256_of(token)[:12]}")
            if src not in a_g.nodes:
                src = _node("a", "val_lit", f"{arg_name}={token}")
            dst = _node("c", "chunk", ch.chunk_id)
            edge_type = "injected-from" if ch.trust_level == 0 else "grounds-in"
            cross.append((src, dst, edge_type))

    # 3. chained-from-prev-tool also creates an action -> context edge
    # to the prior tool output node (so clauses can pattern-match
    # "I'm calling X after observing prior tool Y produced output Z").
    for prior in action.prior_tool_outputs:
        src = _node("a", "prev_tool", prior.tool)
        dst = _node("c", "prev_out", prior.call_id)
        if src in a_g.nodes and dst in c_g.nodes:
            cross.append((src, dst, "derived-from"))

    # Deduplicate while preserving order.
    seen: set[tuple[str, str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for e in cross:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _hash_graph(g: nx.MultiDiGraph) -> str:
    """Stable hash of a NetworkX MultiDiGraph.

    Sort nodes by id, then sort edges by (src, dst, type, key). We
    only hash semantically meaningful attributes (``node_type``,
    ``edge_type``, ``token``); ``call_id`` and ``origin`` enter the
    hash so the same proposed call with a different chunk origin
    yields a different graph hash.
    """
    nodes = sorted(
        (
            (n, _filter_attrs(dict(g.nodes[n])))
            for n in g.nodes
        ),
        key=lambda kv: kv[0],
    )
    edges = sorted(
        (
            (u, v, _filter_attrs(d))
            for u, v, d in g.edges(data=True)
        ),
        key=lambda e: (e[0], e[1], e[2].get("edge_type", "")),
    )
    return sha256_of({"nodes": nodes, "edges": edges})


_NODE_HASH_KEYS = (
    "node_type",
    "token",
    "arg_name",
    "python_type",
    "trust_level",
    "origin",
    "call_id",
    "success",
)
_EDGE_HASH_KEYS = ("edge_type",)


def _filter_attrs(d: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(_NODE_HASH_KEYS) | set(_EDGE_HASH_KEYS)
    return {k: d[k] for k in d if k in keys}


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def build_tri_graph(action: AgentAction, policy: Policy) -> TriGraph:
    """Build the (action, context, policy) tri-graph + cross-graph edges.

    Pre-condition: ``action`` is canonicalised. Callers should pass
    through ``clausegate.schema.canonicalise`` first; this function
    does NOT re-canonicalise so the graph identity is reproducible
    across runs that already share canonical inputs.
    """
    a_g = _build_action_graph(action)
    c_g = _build_context_graph(action)
    p_g = _build_policy_graph(policy)
    cross = _build_cross_edges(action, policy, a_g, c_g, p_g)

    tg = TriGraph(action=a_g, context=c_g, policy=p_g, cross=cross)
    tg.action_hash = _hash_graph(a_g)
    tg.context_hash = _hash_graph(c_g)
    tg.policy_hash = _hash_graph(p_g)
    tg.cross_hash = sha256_of(sorted(cross))
    return tg


# ---------------------------------------------------------------------------
# Cross-edge signal detection
# ---------------------------------------------------------------------------


def _shared_signal(arg_value: str, chunk_text: str) -> str | None:
    """Return a non-trivial common signal between arg value and chunk text.

    Strategy:
      * Direct: if the full arg value appears in the chunk (or vice versa).
      * Known-injection needles: if any known-attack substring appears
        in BOTH.
      * Longest-common contiguous substring of >= 10 chars.

    Returns the matching substring, or ``None`` if no signal found.
    """
    if not arg_value or not chunk_text:
        return None
    lo_arg = arg_value.lower()
    lo_chunk = chunk_text.lower()

    if lo_arg in lo_chunk or lo_chunk in lo_arg:
        return arg_value if lo_arg in lo_chunk else chunk_text

    for needle in _CROSS_INJECTION_NEEDLES:
        n = needle.lower()
        if n in lo_arg and n in lo_chunk:
            return needle

    # Coarse LCS: scan trigrams from the arg, look for the longest run
    # that also appears in the chunk.
    best = ""
    L = len(lo_arg)
    for start in range(L):
        for end in range(start + 10, min(L, start + 64) + 1):
            sub = lo_arg[start:end]
            if sub in lo_chunk and len(sub) > len(best):
                best = sub
            else:
                if not (sub in lo_chunk):
                    break
    return best if len(best) >= 10 else None


# ---------------------------------------------------------------------------
# Mini sentence splitter
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Lightweight regex-based sentence split.

    spaCy / NLTK are NOT pulled in here on purpose: this module
    runs on every verifier call, sub-50 ms budget, and the sentence
    boundaries only feed into the context graph topology. Quality
    is sufficient for clauses to fire on per-sentence patterns;
    research/06 demonstrates this for FEVER and the same applies here.
    """
    import re

    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text)
    return [p.strip() for p in parts if p.strip()]
