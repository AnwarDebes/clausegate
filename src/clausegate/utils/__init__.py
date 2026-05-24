"""clausegate.utils: deterministic helpers.

Public surface:
    seed_all       , seed numpy + python random
    canonical_json , JSON dump with sorted keys, fixed separators, ASCII fold
    sha256_of      , SHA-256 of an arbitrary JSON-serialisable object
    safe_write_json, atomic JSON write (write-to-temp, rename)
"""
from .seeding import seed_all
from .io import canonical_json, safe_write_json, sha256_of

__all__ = ["seed_all", "canonical_json", "safe_write_json", "sha256_of"]
