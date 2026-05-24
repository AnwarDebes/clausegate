"""clausegate.data: dataset loaders + synthetic-data generators.

Public surface:
    SyntheticGenerator     , reproducible AgentAction generator
    load_agentdojo         , adapter for the AgentDojo benchmark
    load_injecagent        , adapter for InjecAgent
    load_rjudge            , adapter for R-Judge
    DEFAULT_POLICY         , sensible default Policy used by tests
"""
from .synth import DEFAULT_POLICY, SyntheticGenerator
from .agentdojo_loader import load_agentdojo
from .injecagent_loader import load_injecagent
from .rjudge_loader import load_rjudge

__all__ = [
    "DEFAULT_POLICY",
    "SyntheticGenerator",
    "load_agentdojo",
    "load_injecagent",
    "load_rjudge",
]
