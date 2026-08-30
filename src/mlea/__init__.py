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
from .bench import SUITE, CompetitionSpec, make_competition, make_suite
from .grade import GradingReport, grade_submission, medal_ranks
from .harness import (
    CommandAgent,
    HarnessError,
    RunConfig,
    RunResult,
    Task,
    run_one,
    run_sweep,
)
from .metrics import METRICS, InvalidSubmission, get_metric
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
    "CompetitionSpec",
    "GradingReport",
    "InvalidSubmission",
    "METRICS",
    "SUITE",
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
