"""clausegate.recourse: counterfactual minimum-argument-edit search.

Public surface:
    ArgumentEdit            , a single edit (op + target + new_value)
    RecourseOutcome         , dataclass returned by greedy_argument_edit
    candidates_from_clauses , clause-attributed candidate generator
    apply_edit              , apply an ArgumentEdit to an AgentAction
    greedy_argument_edit    , top-level greedy minimum-edit search
"""
from .candidates import ArgumentEdit, apply_edit, candidates_from_clauses
from .search import RecourseOutcome, greedy_argument_edit

__all__ = [
    "ArgumentEdit",
    "RecourseOutcome",
    "apply_edit",
    "candidates_from_clauses",
    "greedy_argument_edit",
]
