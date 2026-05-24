"""clausegate.tm: tri-graph walking HGTM verifier.

Public surface:
    TriGraphHGTM           , top-level verifier (CPU reference; CUDA wrapper in tm.cuda_runner)
    PredictionWithClauses  , dataclass returned by predict_with_clauses
    MultiClassTM           , vanilla NumPy multi-class TM used as the CPU reference
    HGTMSpec               , architecture spec (mirrors graphtm-cbr HGraphTMSpec)
"""
from .vanilla import MultiClassTM
from .verifier import HGTMSpec, PredictionWithClauses, TriGraphHGTM

__all__ = [
    "HGTMSpec",
    "MultiClassTM",
    "PredictionWithClauses",
    "TriGraphHGTM",
]
