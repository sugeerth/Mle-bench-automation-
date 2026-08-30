"""Comparison and power tooling for MLE-bench sweeps."""

from .compare import Comparison, compare
from .power import (
    DESIGNS,
    Design,
    PowerResult,
    minimum_detectable_effect,
    power_for_effect,
    seeds_needed,
)
from .records import (
    Fingerprint,
    IncomparableError,
    RunRecord,
    RunSet,
    assert_comparable,
)

__version__ = "0.1.0"

__all__ = [
    "Comparison",
    "DESIGNS",
    "Design",
    "Fingerprint",
    "IncomparableError",
    "PowerResult",
    "RunRecord",
    "RunSet",
    "__version__",
    "assert_comparable",
    "compare",
    "minimum_detectable_effect",
    "power_for_effect",
    "seeds_needed",
]
