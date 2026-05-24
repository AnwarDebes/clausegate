"""TriGraph -> flat binary literal vector (the TM input).

The HGTM (graph-walking) takes node/edge sequences directly via the
upstream ``GraphTsetlinMachine.Graphs`` data structure. For the
CPU reference verifier (``clausegate.tm.verifier.TriGraphHGTM``)
and for the SAT receipt encoder, we also need a **flat literal
vector** that records which symbols from a frozen vocabulary are
present in the input. That flat vector is what a vanilla TM
consumes for testing, and it is what the receipt's CNF encoder
quotes verbatim.

Design rules
------------
1. **Frozen vocabulary**: the vocabulary is stored on disk in a
   model artefact (``configs/symbol_vocab.json``) and never grows
   at inference time. New symbols at inference are mapped to an
   out-of-vocabulary literal whose presence is itself a useful
   pattern for clauses.
2. **Thermometer-encode numeric args**: a numeric arg value is
   encoded against a fixed list of thresholds so a clause can
   express ``amount >= 1000`` as a Boolean literal.
3. **Edge-type literals**: the presence of each edge type in each
   sub-graph is its own literal. This is what makes
   ``injected-from`` syntactically visible to clauses.
4. **Determinism**: ``binarize(tg)`` is a pure function of the
   tri-graph + vocabulary; same input bits every time.

Layout of the literal vector
----------------------------
``literal_vector`` has a fixed schema; the index ranges are stable
once the vocabulary is frozen:

  [0 .. N_TOOLS)             one-hot for the action tool
  [N_TOOLS .. N_TOOLS + N_ARGS)   one-hot for each arg name
  [.. N_ARG_VALUE_BUCKETS)   thermometer + one-hot for arg values
  [.. N_EDGE_TYPES_ACTION)   presence of action-graph edge types
  [.. N_EDGE_TYPES_CTX)      presence of context-graph edge types
  [.. N_EDGE_TYPES_POLICY)   presence of policy-graph edge types
  [.. N_CROSS_EDGE_TYPES)    presence of cross-graph edge types
  [.. N_TRUST_LEVELS)        bucketed trust levels of chunks
  [.. N_POLICY_PERMS)        permission set for the action tool
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..utils.io import sha256_of
from .builder import TriGraph


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vocab:
    """The frozen symbol vocabulary used at inference + receipt time.

    Slots are deterministic-ordered tuples; each tuple's index is
    the literal id used by the SAT encoder.
    """

    tools: tuple[str, ...] = ()
    arg_names: tuple[str, ...] = ()
    arg_string_symbols: tuple[str, ...] = ()
    arg_numeric_thresholds: tuple[float, ...] = ()
    action_edge_types: tuple[str, ...] = (
        "has-arg",
        "has-type",
        "binds-to",
        "chained-from-prev-tool",
        "cites-context-chunk-id",
    )
    context_edge_types: tuple[str, ...] = (
        "contains",
        "cites",
        "quotes",
        "derived-from",
        "provenance",
    )
    policy_edge_types: tuple[str, ...] = (
        "allows",
        "denies",
        "requires-oversight",
        "scope-of",
    )
    cross_edge_types: tuple[str, ...] = (
        "grounds-in",
        "permitted-by",
        "injected-from",
        "derived-from",
    )
    trust_levels: tuple[int, ...] = (0, 1, 2, 3)
    policy_perms: tuple[str, ...] = ("allow", "deny", "requires_oversight", "unknown")
    oov_token: str = "<OOV>"

    def vocab_hash(self) -> str:
        return sha256_of(self.__dict__)

    # ----- index helpers ----
    @property
    def slot_sizes(self) -> dict[str, int]:
        return {
            "tools": len(self.tools) + 1,  # + OOV
            "arg_names": len(self.arg_names) + 1,
            "arg_string_symbols": len(self.arg_string_symbols) + 1,
            "arg_numeric_thresholds": len(self.arg_numeric_thresholds),
            "action_edge_types": len(self.action_edge_types),
            "context_edge_types": len(self.context_edge_types),
            "policy_edge_types": len(self.policy_edge_types),
            "cross_edge_types": len(self.cross_edge_types),
            "trust_levels": len(self.trust_levels),
            "policy_perms": len(self.policy_perms),
        }

    @property
    def total_size(self) -> int:
        return sum(self.slot_sizes.values())

    def slot_offset(self, slot: str) -> int:
        off = 0
        for k, v in self.slot_sizes.items():
            if k == slot:
                return off
            off += v
        raise KeyError(slot)


SYMBOL_VOCAB = Vocab(
    # The default vocabulary is purposefully small + AgentDojo-flavoured;
    # production deployments pin a larger vocab built from their tool
    # registry. The CI test suite uses the defaults below.
    tools=(
        "send_email",
        "read_email",
        "search_web",
        "post_message",
        "read_calendar",
        "create_event",
        "transfer_funds",
        "read_balance",
        "delete_file",
        "share_file",
        "create_file",
        "list_directory",
    ),
    arg_names=(
        "to",
        "cc",
        "bcc",
        "subject",
        "body",
        "amount",
        "recipient",
        "currency",
        "query",
        "url",
        "path",
        "filename",
        "channel",
        "message",
        "title",
        "start",
        "end",
        "attendees",
    ),
    arg_string_symbols=(
        "BCC:attacker@",
        "BCC:secret@",
        "transfer-100k",
        "DELETE",
        "rm -rf",
        "drop table",
        "ignore previous",
        "system:",
        "<script",
        "javascript:",
        "data:text",
        "file://",
        "../",
        "%2e%2e",
        "USD",
        "EUR",
        "@company.com",
        "@external.com",
        "confidential",
        "secret",
        "publish",
    ),
    arg_numeric_thresholds=(0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0),
)


# ---------------------------------------------------------------------------
# BinaryFeatures
# ---------------------------------------------------------------------------


@dataclass
class BinaryFeatures:
    """The TM-input flat literal vector + provenance.

    Attributes:
        literal_vector: ``np.ndarray`` of ``uint8`` with shape
            ``(vocab.total_size,)``; values are 0 / 1.
        symbol_labels: tuple of strings, the same length, naming
            each literal slot for the receipt and the recourse
            module's edit labels.
        vocab_hash: hash of the vocabulary used; pins the receipt.
        graph_signature: signature of the source tri-graph.
    """

    literal_vector: np.ndarray
    symbol_labels: tuple[str, ...]
    vocab_hash: str
    graph_signature: str

    def __post_init__(self) -> None:
        if self.literal_vector.dtype != np.uint8:
            raise TypeError("literal_vector must be uint8")
        if self.literal_vector.ndim != 1:
            raise ValueError("literal_vector must be 1-D")
        if len(self.symbol_labels) != self.literal_vector.shape[0]:
            raise ValueError(
                f"symbol_labels length {len(self.symbol_labels)} "
                f"!= literal_vector length {self.literal_vector.shape[0]}"
            )

    def n_literals(self) -> int:
        return int(self.literal_vector.shape[0])

    def active_indices(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.where(self.literal_vector == 1)[0])

    def literal_hash(self) -> str:
        return sha256_of({"v": self.literal_vector.tolist(), "vocab": self.vocab_hash})


# ---------------------------------------------------------------------------
# Slot fillers
# ---------------------------------------------------------------------------


def _set_one_hot(out: np.ndarray, labels: list[str], offset: int, items: tuple[str, ...], value: str, oov: bool = True) -> str:
    """Set a one-hot slot. Returns the label that was set."""
    try:
        idx = items.index(value)
        out[offset + idx] = 1
        labels[offset + idx] = f"one-hot:{value}"
        return f"one-hot:{value}"
    except ValueError:
        if oov:
            out[offset + len(items)] = 1
            labels[offset + len(items)] = f"one-hot:<OOV>:{value}"
            return f"one-hot:<OOV>:{value}"
    return ""


def _set_thermometer(out: np.ndarray, labels: list[str], offset: int, thresholds: tuple[float, ...], value: float) -> None:
    for i, t in enumerate(thresholds):
        bit = 1 if value >= t else 0
        out[offset + i] = bit
        labels[offset + i] = f"thermometer:>={t}"


# ---------------------------------------------------------------------------
# binarize
# ---------------------------------------------------------------------------


def binarize(tri_graph: TriGraph, vocab: Vocab = SYMBOL_VOCAB) -> BinaryFeatures:
    """TriGraph -> BinaryFeatures.

    Pure function. The output literal vector + vocab hash + graph
    signature uniquely determine the SAT receipt body.
    """
    n = vocab.total_size
    out = np.zeros(n, dtype=np.uint8)
    labels: list[str] = [""] * n

    # Walk the action graph to find tool + args.
    a_g = tri_graph.action
    tool_token = None
    for nid, attrs in a_g.nodes(data=True):
        if attrs.get("node_type") == "tool":
            tool_token = attrs["token"]
            break
    if tool_token is not None:
        _set_one_hot(out, labels, vocab.slot_offset("tools"), vocab.tools, tool_token)

    # arg names + values
    arg_string_off = vocab.slot_offset("arg_string_symbols")
    arg_num_off = vocab.slot_offset("arg_numeric_thresholds")
    arg_name_off = vocab.slot_offset("arg_names")
    for nid, attrs in a_g.nodes(data=True):
        nt = attrs.get("node_type")
        if nt == "arg_name":
            _set_one_hot(out, labels, arg_name_off, vocab.arg_names, attrs["token"])
        elif nt == "arg_value_symbol":
            token = str(attrs["token"])
            # Match against arg_string_symbols by substring (so an
            # arg containing "BCC:attacker@malice.com" lights up the
            # "BCC:attacker@" literal).
            for i, sym in enumerate(vocab.arg_string_symbols):
                if sym.lower() in token.lower():
                    out[arg_string_off + i] = 1
                    labels[arg_string_off + i] = f"string-contains:{sym}"
            # The remaining OOV bit is always lit so the clause can
            # condition on "this arg had some string value".
            out[arg_string_off + len(vocab.arg_string_symbols)] = 1
            labels[arg_string_off + len(vocab.arg_string_symbols)] = "string-present"
        elif nt == "arg_value_literal":
            token = attrs["token"]
            try:
                v = float(token)
            except (TypeError, ValueError):
                continue
            _set_thermometer(out, labels, arg_num_off, vocab.arg_numeric_thresholds, v)

    # Action-graph edge type presence
    a_edge_off = vocab.slot_offset("action_edge_types")
    a_edge_types = {d["edge_type"] for _, _, d in a_g.edges(data=True) if "edge_type" in d}
    for i, t in enumerate(vocab.action_edge_types):
        out[a_edge_off + i] = 1 if t in a_edge_types else 0
        labels[a_edge_off + i] = f"action-edge:{t}"

    # Context-graph edge type presence
    c_edge_off = vocab.slot_offset("context_edge_types")
    c_g = tri_graph.context
    c_edge_types = {d["edge_type"] for _, _, d in c_g.edges(data=True) if "edge_type" in d}
    for i, t in enumerate(vocab.context_edge_types):
        out[c_edge_off + i] = 1 if t in c_edge_types else 0
        labels[c_edge_off + i] = f"context-edge:{t}"

    # Policy-graph edge type presence
    p_edge_off = vocab.slot_offset("policy_edge_types")
    p_g = tri_graph.policy
    p_edge_types = {d["edge_type"] for _, _, d in p_g.edges(data=True) if "edge_type" in d}
    for i, t in enumerate(vocab.policy_edge_types):
        out[p_edge_off + i] = 1 if t in p_edge_types else 0
        labels[p_edge_off + i] = f"policy-edge:{t}"

    # Cross-graph edge type presence
    x_edge_off = vocab.slot_offset("cross_edge_types")
    x_edge_types = {et for _, _, et in tri_graph.cross}
    for i, t in enumerate(vocab.cross_edge_types):
        out[x_edge_off + i] = 1 if t in x_edge_types else 0
        labels[x_edge_off + i] = f"cross-edge:{t}"

    # Trust-level bucket presence
    trust_off = vocab.slot_offset("trust_levels")
    trust_values = {
        int(attrs.get("trust_level", -1))
        for _, attrs in c_g.nodes(data=True)
        if attrs.get("node_type") == "chunk"
    }
    for i, t in enumerate(vocab.trust_levels):
        out[trust_off + i] = 1 if t in trust_values else 0
        labels[trust_off + i] = f"trust-level={t}"

    # Policy permission set (for the action's tool only)
    perm_off = vocab.slot_offset("policy_perms")
    perm_token = "unknown"
    for u, v, d in p_g.edges(data=True):
        # The edge type from a tool node encodes its permission.
        if d.get("edge_type") in ("allows", "denies", "requires-oversight"):
            n_u = p_g.nodes[u]
            if n_u.get("token") == tool_token:
                perm_token = {
                    "allows": "allow",
                    "denies": "deny",
                    "requires-oversight": "requires_oversight",
                }[d["edge_type"]]
                break
    _set_one_hot(out, labels, perm_off, vocab.policy_perms, perm_token, oov=False)

    return BinaryFeatures(
        literal_vector=out,
        symbol_labels=tuple(labels),
        vocab_hash=vocab.vocab_hash(),
        graph_signature=tri_graph.signature(),
    )
