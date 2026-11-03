"""SWE-bench support: generate instances, run patch-producing agents, grade them.

The same treatment this package gives MLE-bench, applied to the other benchmark
in this space. The shape of the problem is different in one important way, and
that difference is the reason the SWE-bench side can be verified more deeply.

MLE-bench has **no correctness oracle**: nothing can decide automatically
whether a competition was prepared correctly, which is why a mis-split shows up
as a silently valid-looking task. SWE-bench does: an instance is well-formed iff
its gold patch flips every FAIL_TO_PASS test from failing to passing while
leaving every PASS_TO_PASS test passing. That is mechanical, so every generated
instance here is *verified against it at generation time* -- a generator that
cannot produce a broken instance without noticing.

Grading needs no Docker. Upstream's resolution logic is log parsing plus set
comparison; containers exist only to produce the log safely. So instances
generated here are graded by upstream's own ``get_eval_report``, and this
package's independent implementation is checked against it.
"""

from .agents import EXPECTED, STRATEGIES, patch_for
from .bench import SweInstance, SweSpec, make_instance, make_suite
from .grade import ResolutionReport, grade_log, run_tests
from .harness import apply_patch, run_agent
from . import upstream

__all__ = [
    "EXPECTED",
    "STRATEGIES",
    "ResolutionReport",
    "SweInstance",
    "SweSpec",
    "apply_patch",
    "grade_log",
    "make_instance",
    "make_suite",
    "patch_for",
    "run_agent",
    "run_tests",
    "upstream",
]
