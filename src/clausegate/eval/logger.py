"""Per-sample JSONL logger.

Mirrors graphtm-cbr's pattern: every verifier decision is appended
as one canonical-JSON line to ``results/jsonl/seed_<seed>/per_sample.jsonl``.
Downstream stats / plots / case studies are computed from this
JSONL alone.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable

from ..utils.io import canonical_json


class PerSampleLogger:
    """Append-only JSONL logger.

    Usage:
        with PerSampleLogger("results/jsonl/seed_1/per_sample.jsonl") as log:
            log.write({"sample_id": "x", "verdict": 1, ...})

    Lines are canonical JSON; same key order across runs, so a
    `diff` between two seeds is a useful regression-finder.
    """

    def __init__(self, path: str | Path, *, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = "a" if append else "w"
        self._fh = None

    def __enter__(self) -> "PerSampleLogger":
        self._fh = open(self.path, self._mode, encoding="ascii")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("PerSampleLogger must be used as a context manager")
        record_with_ts = dict(record)
        record_with_ts.setdefault("ts", time.time())
        self._fh.write(canonical_json(record_with_ts) + "\n")

    def write_many(self, records: Iterable[dict[str, Any]]) -> None:
        for r in records:
            self.write(r)
