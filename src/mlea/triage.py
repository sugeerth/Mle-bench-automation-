"""Classify why a run ended, from its artifacts.

A raw medal rate blends three different things: genuine ML underperformance, an
agent that produced a malformed file, and our own harness breaking. Only the
first is capability signal. Reporting them as one percentage makes the number
uninterpretable, and -- worse -- silently retrying the third inflates it.

This module is deliberately rules-based and deterministic: exit codes, wall
clock, and log signatures. The residual ``CRASH`` bucket is where an LLM
classifier would earn its keep later; the common cases should not pay for one.

Attribution that logs cannot settle is marked ``ambiguous`` with a note naming
the telemetry that would resolve it, rather than guessed at.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


class Outcome(str, Enum):
    """Why a run ended, in precedence order of diagnosis."""

    #: Our fault. Excluded from capability metrics entirely and the only class
    #: that may be retried.
    INFRA = "infra"
    #: Ran out of wall clock without leaving a gradeable submission.
    TIMEOUT = "timeout_no_submission"
    #: Killed for memory. Host OOM and CUDA OOM are distinguished in evidence.
    OOM = "oom"
    #: Non-zero exit that is not OOM, timeout, or infra.
    CRASH = "crash"
    #: Exited cleanly but never wrote submission.csv.
    NO_SUBMISSION = "no_submission"
    #: Wrote a submission the grader refuses.
    INVALID_SUBMISSION = "invalid_submission"
    #: Produced a gradeable submission. The only class that is capability signal.
    VALID = "valid"

    @property
    def is_capability_signal(self) -> bool:
        """Whether this outcome says anything about how good the agent is at ML."""
        return self is Outcome.VALID

    @property
    def attributable_to_agent(self) -> bool:
        """Whether the agent's author is the person who can fix this."""
        return self in {
            Outcome.TIMEOUT,
            Outcome.OOM,
            Outcome.CRASH,
            Outcome.NO_SUBMISSION,
            Outcome.INVALID_SUBMISSION,
        }


@dataclass(frozen=True)
class Evidence:
    """One signal that fired, kept so a classification can be argued with."""

    signal: str
    detail: str

    def __str__(self) -> str:
        return f"{self.signal}: {self.detail}"


@dataclass(frozen=True)
class Signature:
    pattern: re.Pattern[str]
    signal: str
    outcome: Outcome
    ambiguous: bool = False
    note: str = ""


def _sig(
    regex: str, signal: str, outcome: Outcome, ambiguous: bool = False, note: str = ""
) -> Signature:
    return Signature(re.compile(regex, re.IGNORECASE), signal, outcome, ambiguous, note)


#: Ordered. The first match wins, so put the least ambiguous signatures first.
INFRA_SIGNATURES: tuple[Signature, ...] = (
    _sig(r"spot instance.{0,40}(interrupt|terminat)", "spot-interruption", Outcome.INFRA),
    _sig(r"\bpreempt(ed|ion)\b", "preemption", Outcome.INFRA),
    _sig(r"errimagepull|imagepullbackoff|failed to pull image|manifest unknown",
         "image-pull", Outcome.INFRA),
    _sig(r"failed to mount|mount error|cannot mount|invalid mount", "mount", Outcome.INFRA),
    _sig(r"docker daemon.{0,40}(not running|connection refused)", "docker-daemon",
         Outcome.INFRA),
    _sig(r"node .{0,30}(terminated|notready)", "node-lost", Outcome.INFRA),
    _sig(
        r"no space left on device",
        "disk-full",
        Outcome.INFRA,
        ambiguous=True,
        note=(
            "disk sizing is a harness responsibility, but an agent writing "
            "unbounded checkpoints causes the same message. Resolve with the "
            "container's disk-usage telemetry before retrying."
        ),
    ),
)

MEMORY_SIGNATURES: tuple[Signature, ...] = (
    _sig(r"cuda out of memory|cudaerrormemoryallocation", "cuda-oom", Outcome.OOM,
         note="GPU OOM: the agent chose the batch size or model. Agent-attributable."),
    _sig(r"out of memory: killed process|oom-kill|oom_reaper|memorykilled", "host-oom",
         Outcome.OOM,
         note="host OOM at the configured memory limit. Agent-attributable unless "
              "the node was oversubscribed; check node memory telemetry."),
)

#: Exit codes with a fixed meaning.
EXIT_TIMEOUT = 124  # GNU coreutils timeout
EXIT_SIGKILL = 137  # 128 + SIGKILL, used by both OOM killers and timeout enforcers


@dataclass
class RunArtifacts:
    """Everything triage needs about one run.

    Constructed from an upstream ``run_agent.py`` competition directory, or
    directly in tests. ``time_cap_seconds`` of ``None`` means no cap was set.
    """

    competition_id: str
    seed: int = 0
    exit_code: int = 0
    wall_clock_seconds: float = 0.0
    time_cap_seconds: float | None = None
    has_submission: bool = False
    submission_rows: int | None = None
    #: Set when the grader or the upstream validation server rejected the file.
    validation_error: str | None = None
    log_tail: str = ""

    @property
    def hit_time_cap(self) -> bool:
        if self.time_cap_seconds is None:
            return False
        # A 1% margin: enforcement is not instantaneous and runs are hours long.
        return self.wall_clock_seconds >= self.time_cap_seconds * 0.99


@dataclass(frozen=True)
class TriageResult:
    competition_id: str
    seed: int
    outcome: Outcome
    evidence: tuple[Evidence, ...] = ()
    ambiguous: bool = False
    notes: tuple[str, ...] = ()
    #: True when the run produced a gradeable submission despite ending early.
    truncated: bool = False

    @property
    def should_retry(self) -> bool:
        """Only infra failures may be retried.

        Retrying anything else re-rolls the dice on an agent failure and
        inflates the medal rate. This property is the whole reason the module
        exists; see :func:`assert_retry_allowed`.
        """
        return self.outcome is Outcome.INFRA

    @property
    def is_capability_signal(self) -> bool:
        return self.outcome.is_capability_signal

    def __str__(self) -> str:
        bits = [f"{self.competition_id}[seed {self.seed}]", self.outcome.value]
        if self.truncated:
            bits.append("(truncated)")
        if self.ambiguous:
            bits.append("(ambiguous)")
        if self.evidence:
            bits.append(f"<- {self.evidence[0]}")
        return " ".join(bits)


class RetryNotAllowed(RuntimeError):
    """Raised when something tries to retry a non-infra failure."""


def assert_retry_allowed(result: TriageResult) -> None:
    """Guard for the scheduler. Enforces PLAN.md section 3.1 in code.

    "Retry infra failures, never agent failures" is the kind of rule that erodes
    the moment someone is staring at a red dashboard, so it is a raised
    exception rather than a comment.
    """
    if not result.should_retry:
        raise RetryNotAllowed(
            f"{result.competition_id} seed {result.seed}: outcome "
            f"{result.outcome.value!r} is a RESULT, not an infrastructure fault. "
            "Retrying it re-rolls the dice on an agent failure and inflates the "
            "medal rate. Record it and move on."
        )


def _scan(log: str, signatures: Iterable[Signature]) -> tuple[Signature, str] | None:
    """Return the first matching signature and the log line it fired on.

    The evidence detail is the whole line, not the matched span: a person
    auditing a classification wants to read the log line that caused it, and a
    bare span clipped at a pattern boundary ("Spot Instance interrupt") reads as
    a bug rather than as evidence.
    """
    for sig in signatures:
        m = sig.pattern.search(log)
        if m:
            start = log.rfind("\n", 0, m.start()) + 1
            end = log.find("\n", m.end())
            line = (log[start:] if end == -1 else log[start:end]).strip()
            if len(line) > 160:
                line = line[:157] + "..."
            return sig, line or m.group(0).strip()
    return None


def classify(artifacts: RunArtifacts) -> TriageResult:
    """Diagnose one run.

    Order of reasoning:

    1. An infra signature invalidates everything else -- if the node vanished,
       nothing downstream tells you anything about the agent.
    2. Otherwise, if a gradeable submission exists, the run is a **result**, even
       if it was killed at the time cap. MLE-bench grades whatever
       ``submission.csv`` is on disk at the end, so a timeout that still left a
       valid file is a successful run that happened to be truncated -- not a
       failure. Getting this backwards would discard real results.
    3. Only when there is no gradeable submission do we diagnose why.
    """
    log = artifacts.log_tail
    notes: list[str] = []

    hit = _scan(log, INFRA_SIGNATURES)
    if hit is not None:
        infra, matched = hit
        if infra.note:
            notes.append(infra.note)
        return TriageResult(
            competition_id=artifacts.competition_id,
            seed=artifacts.seed,
            outcome=Outcome.INFRA,
            evidence=(Evidence(infra.signal, matched),),
            ambiguous=infra.ambiguous,
            notes=tuple(notes),
        )

    if artifacts.has_submission and artifacts.validation_error is None:
        truncated = artifacts.hit_time_cap or artifacts.exit_code != 0
        if truncated:
            notes.append(
                "gradeable submission present despite an early end; graded as a "
                "result, not a failure"
            )
        if artifacts.submission_rows == 0:
            # An empty file is on disk but nothing can be graded from it.
            return TriageResult(
                artifacts.competition_id,
                artifacts.seed,
                Outcome.INVALID_SUBMISSION,
                (Evidence("empty-submission", "submission.csv has 0 rows"),),
                notes=tuple(notes),
            )
        return TriageResult(
            artifacts.competition_id,
            artifacts.seed,
            Outcome.VALID,
            notes=tuple(notes),
            truncated=truncated,
        )

    if artifacts.validation_error is not None:
        return TriageResult(
            artifacts.competition_id,
            artifacts.seed,
            Outcome.INVALID_SUBMISSION,
            (Evidence("grader-rejected", artifacts.validation_error),),
        )

    # No gradeable submission. Diagnose the cause.
    mem = _scan(log, MEMORY_SIGNATURES)


    if artifacts.hit_time_cap:
        # SIGKILL at the cap is the time enforcer, not the OOM killer.
        return TriageResult(
            artifacts.competition_id,
            artifacts.seed,
            Outcome.TIMEOUT,
            (
                Evidence(
                    "wall-clock",
                    f"{artifacts.wall_clock_seconds:.0f}s of "
                    f"{artifacts.time_cap_seconds:.0f}s cap",
                ),
            ),
            notes=("more budget might have helped; check the run's score curve",),
        )

    if mem is not None:
        mem_sig, mem_matched = mem
        return TriageResult(
            artifacts.competition_id,
            artifacts.seed,
            Outcome.OOM,
            (Evidence(mem_sig.signal, mem_matched),),
            notes=(mem_sig.note,) if mem_sig.note else (),
        )

    if artifacts.exit_code == EXIT_TIMEOUT:
        return TriageResult(
            artifacts.competition_id,
            artifacts.seed,
            Outcome.TIMEOUT,
            (Evidence("exit-code", f"exit {EXIT_TIMEOUT} (timeout)"),),
        )

    if artifacts.exit_code == EXIT_SIGKILL:
        return TriageResult(
            artifacts.competition_id,
            artifacts.seed,
            Outcome.OOM,
            (Evidence("exit-code", "exit 137 (SIGKILL), no time cap reached"),),
            ambiguous=True,
            notes=(
                "exit 137 is SIGKILL, which both OOM killers and time enforcers "
                "send. No cap was hit and no OOM signature was logged, so this is "
                "a guess. Resolve with container memory telemetry.",
            ),
        )

    if artifacts.exit_code != 0:
        return TriageResult(
            artifacts.competition_id,
            artifacts.seed,
            Outcome.CRASH,
            (Evidence("exit-code", f"exit {artifacts.exit_code}"),),
        )

    return TriageResult(
        artifacts.competition_id,
        artifacts.seed,
        Outcome.NO_SUBMISSION,
        (Evidence("clean-exit", "exit 0 but no submission.csv"),),
    )


@dataclass
class TriageReport:
    """Aggregate view over a run group."""

    results: list[TriageResult] = field(default_factory=list)

    def counts(self) -> dict[Outcome, int]:
        out = {o: 0 for o in Outcome}
        for r in self.results:
            out[r.outcome] += 1
        return out

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def gradeable(self) -> list[TriageResult]:
        return [r for r in self.results if r.is_capability_signal]

    @property
    def infra(self) -> list[TriageResult]:
        return [r for r in self.results if r.outcome is Outcome.INFRA]

    @property
    def mechanical(self) -> list[TriageResult]:
        """Agent-attributable failures: real bugs, but not ML capability."""
        return [
            r
            for r in self.results
            if r.outcome.attributable_to_agent and not r.is_capability_signal
        ]

    @property
    def ambiguous(self) -> list[TriageResult]:
        return [r for r in self.results if r.ambiguous]

    def effective_denominator(self) -> int:
        """Runs that should count toward a capability metric.

        Infra failures are excluded because they are our fault; counting them as
        agent failures understates the agent. Mechanical failures stay in --
        they are the agent's bugs.
        """
        return self.total - len(self.infra)

    def summary(self) -> str:
        c = self.counts()
        lines = [f"{self.total} run(s)"]
        for outcome in Outcome:
            if c[outcome]:
                lines.append(f"  {outcome.value:<22} {c[outcome]:>4}")
        lines.append("")
        lines.append(
            f"  capability signal      {len(self.gradeable):>4}  "
            f"(gradeable submissions)"
        )
        lines.append(
            f"  agent bugs             {len(self.mechanical):>4}  "
            f"(counted against the agent, but not ML capability)"
        )
        lines.append(
            f"  our fault              {len(self.infra):>4}  "
            f"(excluded; retryable)"
        )
        lines.append(
            f"  effective denominator  {self.effective_denominator():>4}  "
            f"of {self.total}"
        )
        if self.ambiguous:
            lines.append("")
            lines.append(
                f"  !! {len(self.ambiguous)} run(s) could not be attributed from "
                f"logs alone:"
            )
            for r in self.ambiguous[:5]:
                lines.append(f"     {r}")
        if self.mechanical and self.total:
            share = len(self.mechanical) / self.total
            if share > 0.15:
                lines.append("")
                lines.append(
                    f"  !! {share:.0%} of runs failed mechanically rather than on ML. "
                    "The medal rate is measuring plumbing, not capability -- fix "
                    "these before reading anything into the score."
                )
        return "\n".join(lines)

    def to_runset_records(self, medals: dict[tuple[str, int], bool] | None = None):
        """Emit records for :class:`mlea.records.RunSet`.

        ``medals`` maps ``(competition_id, seed)`` to the graded result. Runs
        without an entry are recorded as no-medal, which is correct: a run that
        never produced a gradeable submission did not earn one.
        """
        medals = medals or {}
        return [
            {
                "competition_id": r.competition_id,
                "seed": r.seed,
                "any_medal": bool(medals.get((r.competition_id, r.seed), False)),
                "infra_failure": r.outcome is Outcome.INFRA,
            }
            for r in self.results
        ]


def _read_tail(path: Path, max_bytes: int = 64_000) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-max_bytes:].decode("utf-8", errors="replace")


def from_run_dir(path: str | Path, competition_id: str | None = None) -> RunArtifacts:
    """Build artifacts from an upstream ``run_agent.py`` competition directory.

    Expected layout (missing pieces degrade gracefully -- a run that died early
    may have almost nothing on disk, and that absence is itself a signal)::

        <dir>/submission/submission.csv
        <dir>/logs/run.log
        <dir>/metadata.json      {"exit_code":..,"wall_clock_seconds":..,
                                  "time_cap_seconds":..,"seed":..}
    """
    d = Path(path)
    meta = {}
    meta_path = d / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}

    submission = d / "submission" / "submission.csv"
    rows: int | None = None
    if submission.exists():
        try:
            with submission.open() as fh:
                # Subtract the header; a header-only file has 0 data rows.
                rows = max(sum(1 for _ in fh) - 1, 0)
        except OSError:
            rows = None

    log = ""
    logs_dir = d / "logs"
    if logs_dir.is_dir():
        for f in sorted(logs_dir.glob("*.log")):
            log += _read_tail(f)

    return RunArtifacts(
        competition_id=competition_id or meta.get("competition_id", d.name),
        seed=int(meta.get("seed", 0)),
        exit_code=int(meta.get("exit_code", 0)),
        wall_clock_seconds=float(meta.get("wall_clock_seconds", 0.0)),
        time_cap_seconds=(
            float(meta["time_cap_seconds"])
            if meta.get("time_cap_seconds") is not None
            else None
        ),
        has_submission=submission.exists(),
        submission_rows=rows,
        validation_error=meta.get("validation_error"),
        log_tail=log,
    )


def triage_run_group(path: str | Path) -> TriageReport:
    """Triage every competition directory in a run group."""
    root = Path(path)
    results = [
        classify(from_run_dir(child))
        for child in sorted(root.iterdir())
        if child.is_dir()
    ]
    return TriageReport(results)


__all__ = [
    "Evidence",
    "Outcome",
    "RetryNotAllowed",
    "RunArtifacts",
    "TriageReport",
    "TriageResult",
    "assert_retry_allowed",
    "classify",
    "from_run_dir",
    "triage_run_group",
]
