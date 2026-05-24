"""clausegate.eval: metrics + adversarial harness + per-sample logger.

Public surface:
    metrics        , compute ASR, utility, F1, ECE on (actions, labels)
    PerSampleLogger, JSONL logger that mirrors graphtm-cbr's pattern
    AdversarialHarness , run ChatInject / RAG-poison / GCG-MCP stubs
"""
from .metrics import EvalReport, compute_metrics
from .logger import PerSampleLogger
from .adversarial import AdversarialHarness, AttackResult

__all__ = [
    "AdversarialHarness",
    "AttackResult",
    "EvalReport",
    "PerSampleLogger",
    "compute_metrics",
]
