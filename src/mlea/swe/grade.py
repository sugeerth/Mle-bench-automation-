"""Run a test suite, and decide whether an instance was resolved.

An independent implementation of SWE-bench's resolution rule, kept deliberately
separate from upstream's so the two can be compared. ``mlea swe conform`` grades
the same logs both ways and asserts they agree; a disagreement is a bug here.

The rule itself is small and the subtlety is entirely in the second half:

* every FAIL_TO_PASS test must now pass, **and**
* every PASS_TO_PASS test must *still* pass.

The second clause is what makes SWE-bench hard and what a naive harness omits.
A patch that fixes the reported bug and quietly breaks something else is not a
resolution, and a harness that only checks the first clause will score it as one.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: Markers upstream's log reader requires. Reproduced verbatim so a log written
#: here is readable by upstream's parser without translation.
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
TEST_EXIT_CODE = ">>>>> Test Exit Code"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"

_STATUS_RE = re.compile(
    r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.MULTILINE
)


def run_tests(repo: str | Path, test_file: str, timeout: float = 120.0) -> str:
    """Run pytest and return a log wrapped in upstream's markers.

    ``-rA`` is not optional: it is what makes pytest emit the per-test
    ``PASSED``/``FAILED`` summary lines every log parser reads. Without it the
    log is unparseable and every instance grades as unresolved.
    """
    repo = Path(repo)
    try:
        proc = subprocess.run(
            ["python", "-m", "pytest", "-rA", "-p", "no:cacheprovider", test_file],
            cwd=str(repo), capture_output=True, text=True, timeout=timeout,
        )
        body, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        body, code = "tests timed out", 124
    return (
        f"{START_TEST_OUTPUT}\n{body}\n{TEST_EXIT_CODE}: {code}\n{END_TEST_OUTPUT}\n"
    )


def parse_pytest_statuses(log: str) -> dict[str, str]:
    """Map test id to status from a pytest ``-rA`` summary."""
    body = log
    if START_TEST_OUTPUT in log and END_TEST_OUTPUT in log:
        body = log.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]
    out: dict[str, str] = {}
    for status, test in _STATUS_RE.findall(body):
        out[test] = "XFAIL" if status == "XPASS" else status
    return out


@dataclass
class ResolutionReport:
    instance_id: str
    resolved: bool = False
    #: False when the patch never applied, so no test ever ran.
    patch_applied: bool = True
    f2p_success: list[str] = field(default_factory=list)
    f2p_failure: list[str] = field(default_factory=list)
    p2p_success: list[str] = field(default_factory=list)
    p2p_failure: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        """The interesting failure: the bug got fixed and something else broke."""
        return bool(self.p2p_failure) and not self.f2p_failure

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "resolved": self.resolved,
            "patch_applied": self.patch_applied,
            "tests_status": {
                "FAIL_TO_PASS": {"success": self.f2p_success,
                                 "failure": self.f2p_failure},
                "PASS_TO_PASS": {"success": self.p2p_success,
                                 "failure": self.p2p_failure},
            },
        }

    def summary(self) -> str:
        if not self.patch_applied:
            return "patch did not apply"
        if self.resolved:
            return "resolved"
        if self.regressed:
            return f"regressed ({len(self.p2p_failure)} PASS_TO_PASS broken)"
        return f"unresolved ({len(self.f2p_failure)} FAIL_TO_PASS still failing)"


def grade_log(
    log: str, fail_to_pass: list[str], pass_to_pass: list[str],
    instance_id: str = "instance",
) -> ResolutionReport:
    """Decide resolution from a test log."""
    report = ResolutionReport(instance_id=instance_id)
    if APPLY_PATCH_FAIL in log or not (
        START_TEST_OUTPUT in log and END_TEST_OUTPUT in log
    ):
        # No tests ran, so the sets stay empty. Marking every test as failed --
        # which this did until a conformance run against upstream caught it --
        # asserts something the log does not support, and would attribute
        # failures to an agent whose patch never executed. `patch_applied` keeps
        # the distinction upstream folds into a bare `resolved=False`.
        report.patch_applied = False
        return report

    statuses = parse_pytest_statuses(log)
    # A test that did not appear in the log did not pass. Treating a missing
    # test as a pass is the classic way to score a suite that never ran as a
    # resolution.
    for test in fail_to_pass:
        (report.f2p_success if statuses.get(test) == "PASSED"
         else report.f2p_failure).append(test)
    for test in pass_to_pass:
        (report.p2p_success if statuses.get(test) == "PASSED"
         else report.p2p_failure).append(test)
    report.resolved = not report.f2p_failure and not report.p2p_failure
    return report


__all__ = [
    "APPLY_PATCH_FAIL",
    "END_TEST_OUTPUT",
    "ResolutionReport",
    "START_TEST_OUTPUT",
    "TEST_EXIT_CODE",
    "grade_log",
    "parse_pytest_statuses",
    "run_tests",
]
