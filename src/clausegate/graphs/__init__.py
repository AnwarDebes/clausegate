"""clausegate.graphs: tri-graph builder + binarisation.

Public surface:
    TriGraph         , dataclass holding three NetworkX MultiDiGraphs + cross-edges
    build_tri_graph  , (AgentAction, Policy) -> TriGraph
    binarize         , TriGraph -> BinaryFeatures (flat literal vector + provenance)
    BinaryFeatures   , dataclass exposing the literal vector + symbol vocab
    SYMBOL_VOCAB     , the default canonical symbol vocabulary (frozen for receipts)
    Vocab            , the symbol-vocabulary class (overridable for benchmark-specific extensions)
"""
from .builder import TriGraph, build_tri_graph
from .binarize import BinaryFeatures, SYMBOL_VOCAB, Vocab, binarize

__all__ = [
    "BinaryFeatures",
    "SYMBOL_VOCAB",
    "TriGraph",
    "Vocab",
    "binarize",
    "build_tri_graph",
]
