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
from .harness import (
    CommandAgent,
    HarnessError,
    RunConfig,
    RunResult,
    Task,
    run_one,
    run_sweep,
)
from .triage import (
    Outcome,
    RetryNotAllowed,
    RunArtifacts,
    TriageReport,
    TriageResult,
    assert_retry_allowed,
    classify,
    triage_run_group,
)

__version__ = "0.1.0"

__all__ = [
    "CommandAgent",
    "Comparison",
    "DESIGNS",
    "Design",
    "Fingerprint",
    "HarnessError",
    "IncomparableError",
    "Outcome",
    "PowerResult",
    "RetryNotAllowed",
    "RunArtifacts",
    "RunConfig",
    "RunRecord",
    "RunResult",
    "RunSet",
    "Task",
    "TriageReport",
    "TriageResult",
    "__version__",
    "assert_comparable",
    "assert_retry_allowed",
    "classify",
    "compare",
    "triage_run_group",
    "minimum_detectable_effect",
    "power_for_effect",
    "seeds_needed",
]
