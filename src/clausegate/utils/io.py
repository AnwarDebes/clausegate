"""Deterministic JSON + hashing helpers.

The receipt format and graph-hash invariants depend on canonical
JSON encoding: sorted keys, fixed separators, ASCII-only output.
Anything that crosses a hash boundary must use ``canonical_json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable JSON dump.

    Sorted keys, no whitespace, ASCII-only. The same object always
    produces the same string; that is what makes downstream hashes
    deterministic.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )


def _json_default(o: Any) -> Any:
    # Allow tuples, sets, frozensets, dataclasses, and bytes to be
    # serialised so we never silently fail on an unhashable wrapper.
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, bytes):
        return o.hex()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def sha256_of(obj: Any) -> str:
    """SHA-256 of ``canonical_json(obj)``, hex-encoded.

    Use this anywhere a stable cross-process content id is needed:
    receipt fields, graph identity, model-version pinning.
    """
    blob = canonical_json(obj).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def safe_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    """Atomic JSON write: temp file in same dir, then rename.

    Avoids leaving a half-written file if the process dies mid-write,
    which would otherwise look like a tampered receipt on the next run.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(canonical_json(obj))
        os.replace(tmp, p)
    except Exception:
        # Best-effort cleanup; re-raise the original.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
