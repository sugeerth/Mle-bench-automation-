"""The eval harness: run an agent against a competition and collect artifacts.

This is the missing middle of the pipeline. :mod:`mlea.triage` reads run
directories and :mod:`mlea.compare` reads run sets; this is what produces a run
directory in the first place, in exactly the layout triage expects.

Design commitments, each of which is a thing naive harnesses get wrong:

* **The submission on disk at the end is the result.** A run killed at its time
  cap that still left a valid ``submission.csv`` is a graded result, not a
  failure. The harness never deletes a submission because the run ended badly.
* **Kill the process group, not the process.** Agents spawn training children.
  Killing only the parent leaves them running, holding the GPU, and quietly
  poisoning the next run on that node.
* **SIGTERM, then a grace period, then SIGKILL.** An agent that handles SIGTERM
  gets a chance to flush its best submission. Going straight to SIGKILL throws
  that away.
* **Terminal run directories are immutable.** Re-running into a completed run
  directory requires an explicit override, because overwriting a result is how
  a medal rate quietly becomes fiction.
* **Isolation level is recorded, not assumed.** ``isolation="none"`` lands in
  the metadata and flows into the comparison fingerprint, so unsandboxed runs
  can never be silently compared against sandboxed ones.

.. warning::
   With ``isolation="none"`` the agent's command runs with this process's
   privileges and full network access. That is the documented free-tier tradeoff
   (see ``docs/SOTA-AND-FREE-TIER.md``), and it means an agent can reach the
   public solution for the competition it is being graded on. Fine for pipeline
   work, disqualifying for a reported number.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

#: Wall-clock marks (seconds) at which the running submission is snapshotted.
#: Log-spaced: agent progress is front-loaded, so early marks are what carry the
#: information. See docs/PROPOSAL-anytime-eval.md.
DEFAULT_CHECKPOINT_MARKS: tuple[float, ...] = (
    900, 1800, 3600, 7200, 14400, 28800, 57600, 86400,
)

SIGTERM_GRACE_SECONDS = 30.0


class HarnessError(RuntimeError):
    """Raised for harness faults, distinct from anything the agent did."""


def resolve_competition_data_dir(data_root: str | Path, competition_id: str) -> Path:
    """Find a competition's agent-visible data directory under ``data_root``.

    Upstream ``mlebench prepare`` writes
    ``<data-dir>/<competition-id>/{raw,prepared/public,prepared/private}``, and
    the agent is only ever shown ``prepared/public`` -- ``prepared/private``
    holds the answers and is mounted root-only. Pointing an agent at the
    competition directory itself would hand it the labels.

    Falls back to a flat ``<data_root>/<competition-id>`` layout, which is what
    a hand-assembled or test fixture directory looks like.

    .. warning::
       In the prepared layout ``prepared/private`` -- which holds the answers --
       is a *sibling* of the directory returned here. Upstream keeps the agent
       away from it by running in a container that mounts private at an
       unrelated path with mode 700. With ``isolation="none"`` there is no such
       boundary: the answers are one ``..`` away. :func:`run_one` warns loudly
       when it sees this, and the results of such a run must never be reported.
    """
    comp = Path(data_root) / competition_id
    public = comp / "prepared" / "public"
    if public.is_dir():
        return public
    return comp


def answers_reachable(data_dir: Path) -> bool:
    """True when the private answers sit beside the agent's data directory.

    Only meaningful for ``isolation="none"``; a container makes it moot.
    """
    return (Path(data_dir).parent / "private").is_dir()


@dataclass(frozen=True)
class Task:
    """One competition to run, with prepared data already on disk."""

    competition_id: str
    data_dir: Path
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.competition_id:
            raise ValueError("competition_id is required")
        # The agent runs with cwd set to its code directory, so every path it
        # is handed must be absolute or it will resolve against the wrong root.
        object.__setattr__(self, "data_dir", Path(self.data_dir).resolve())

    @property
    def slug(self) -> str:
        """Directory name. Includes the seed so seeds never collide."""
        return f"{self.competition_id}__seed{self.seed}"


class Agent(Protocol):
    """Anything that can be launched to produce a submission.

    The contract is deliberately thin: build a command line. Everything the
    agent needs arrives through the environment, matching upstream MLE-bench's
    ``/home/{data,submission,logs,code}`` convention.
    """

    name: str

    def build_command(self, task: Task, workspace: "Workspace") -> Sequence[str]:
        ...


@dataclass(frozen=True)
class CommandAgent:
    """An agent that is a shell command.

    The command is formatted with ``{data_dir}``, ``{submission_dir}``,
    ``{logs_dir}``, ``{code_dir}``, ``{competition_id}``, ``{seed}`` and
    ``{time_cap_seconds}``, and also receives all of them as environment
    variables. Both, because some agents are easier to wire one way than the
    other and neither costs anything.
    """

    name: str
    template: str

    def build_command(self, task: Task, workspace: "Workspace") -> Sequence[str]:
        return ["sh", "-c", self.template.format(**workspace.substitutions(task))]


@dataclass(frozen=True)
class Workspace:
    """The directory layout handed to one run.

    Matches what :func:`mlea.triage.from_run_dir` reads, so the loop closes.
    """

    root: Path

    @property
    def submission_dir(self) -> Path:
        return self.root / "submission"

    @property
    def submission_path(self) -> Path:
        return self.submission_dir / "submission.csv"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def harness_log_path(self) -> Path:
        """Harness-written log. Trusted: triage reads infra signatures only here."""
        return self.logs_dir / "harness.log"

    @property
    def agent_log_path(self) -> Path:
        """The agent's stdout and stderr. Untrusted -- it is attacker-controlled
        with respect to classification, so an agent printing "preempted" must
        not be able to get its own failure written off as our fault."""
        return self.logs_dir / "agent.log"

    @property
    def code_dir(self) -> Path:
        return self.root / "code"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.json"

    def create(self) -> None:
        for d in (self.submission_dir, self.logs_dir, self.code_dir):
            d.mkdir(parents=True, exist_ok=True)

    def substitutions(self, task: Task) -> dict[str, str]:
        return {
            "data_dir": str(task.data_dir),
            "submission_dir": str(self.submission_dir),
            "submission_path": str(self.submission_path),
            "logs_dir": str(self.logs_dir),
            "code_dir": str(self.code_dir),
            "competition_id": task.competition_id,
            "seed": str(task.seed),
            "time_cap_seconds": "",  # filled by the runner
        }


@dataclass
class RunConfig:
    output_root: Path
    time_cap_seconds: float = 86_400.0
    isolation: str = "none"
    checkpoint_marks: Sequence[float] = DEFAULT_CHECKPOINT_MARKS
    grace_seconds: float = SIGTERM_GRACE_SECONDS
    #: Overwrite a run directory that already holds a completed run.
    force: bool = False
    #: Glob for the agent's own submission file, mirrored into the run's
    #: ``submission/submission.csv``. Real agents write where they like --
    #: AIDE lands at ``workspaces/0-run/working/submission.csv``, MLEvolve at
    #: ``runs/<timestamp>_<id>/workspace/best_submission/submission.csv`` --
    #: so without this every agent needs a polling loop bolted onto its command.
    #: Relative patterns resolve against the run's code directory. The
    #: most-recently-modified match wins.
    submission_glob: str | None = None
    #: Extra environment for the agent process.
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_cap_seconds <= 0:
            raise ValueError("time_cap_seconds must be positive")
        if self.isolation not in ("none", "docker"):
            raise ValueError("isolation must be 'none' or 'docker'")
        # Resolved for the same reason as Task.data_dir: the agent's cwd is its
        # code directory, so a relative output root would send its submission
        # somewhere neither it nor the harness expects.
        self.output_root = Path(self.output_root).resolve()


@dataclass(frozen=True)
class Checkpoint:
    elapsed_seconds: float
    path: Path
    sha256: str


@dataclass(frozen=True)
class RunResult:
    task: Task
    run_dir: Path
    exit_code: int
    wall_clock_seconds: float
    timed_out: bool
    has_submission: bool
    checkpoints: tuple[Checkpoint, ...] = ()
    harness_error: str | None = None

    @property
    def submission_path(self) -> Path:
        return self.run_dir / "submission" / "submission.csv"


def mirror_submission(workspace: "Workspace", pattern: str) -> Path | None:
    """Copy the agent's own submission into the run's canonical location.

    Returns the source path copied from, or ``None`` if nothing matched yet.
    Never raises: a run mid-write, or an agent that has not produced anything,
    is an ordinary state rather than a harness fault.
    """
    root = workspace.code_dir
    try:
        base = Path(pattern)
        matches = (
            list(Path(base.anchor).glob(str(base.relative_to(base.anchor))))
            if base.is_absolute()
            else list(root.glob(pattern))
        )
        candidates = [m for m in matches if m.is_file() and m.stat().st_size > 0]
        if not candidates:
            return None
        newest = max(candidates, key=lambda m: m.stat().st_mtime)
        workspace.submission_dir.mkdir(parents=True, exist_ok=True)
        if newest.resolve() == workspace.submission_path.resolve():
            return newest
        shutil.copy2(newest, workspace.submission_path)
        return newest
    except OSError:
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Checkpointer(threading.Thread):
    """Snapshots the running submission at wall-clock marks.

    Passive by design: it copies whatever is on disk and never signals the
    agent. A run with snapshotting produces the identical final result to one
    without, so headline numbers stay comparable
    (docs/PROPOSAL-anytime-eval.md). Agents that only write at the end produce
    sparse curves; that is a finding about the agent, not a defect here.
    """

    def __init__(
        self,
        workspace: Workspace,
        marks: Sequence[float],
        cap: float,
        submission_glob: str | None = None,
    ):
        super().__init__(daemon=True, name="mlea-checkpointer")
        self._ws = workspace
        self._glob = submission_glob
        self._marks = sorted({m for m in marks if 0 < m < cap})
        # Not `_stop`: that name shadows threading.Thread._stop, which the
        # interpreter calls during join() teardown.
        self._stop_event = threading.Event()
        self._last_hash: str | None = None
        self.checkpoints: list[Checkpoint] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        start = time.monotonic()
        for mark in self._marks:
            remaining = mark - (time.monotonic() - start)
            if remaining > 0 and self._stop_event.wait(remaining):
                return
            if self._stop_event.is_set():
                return
            self._snapshot(mark)

    def _snapshot(self, mark: float) -> None:
        if self._glob:
            mirror_submission(self._ws, self._glob)
        src = self._ws.submission_path
        try:
            if not src.exists() or src.stat().st_size == 0:
                return
            digest = _sha256(src)
        except OSError:
            return  # mid-write or vanished; the next mark will catch it
        if digest == self._last_hash:
            return  # unchanged since the last mark: nothing new to grade
        dest_dir = self._ws.checkpoints_dir / f"t={int(mark)}"
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / "submission.csv")
        except OSError:
            return
        self._last_hash = digest
        self.checkpoints.append(
            Checkpoint(float(mark), dest_dir / "submission.csv", digest)
        )


def _terminate_group(proc: subprocess.Popen, grace: float, log) -> None:
    """SIGTERM the process group, wait, then SIGKILL what is left.

    Agents spawn training children. Signalling only ``proc`` leaves them alive
    holding the GPU, which corrupts the next run scheduled on that node.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return

    def _signal(sig: int) -> None:
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    log(f"[mlea] time cap reached; SIGTERM to process group {pgid}")
    _signal(signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
        log("[mlea] agent exited during the SIGTERM grace period")
        return
    except subprocess.TimeoutExpired:
        pass
    log(f"[mlea] grace period expired; SIGKILL to process group {pgid}")
    _signal(signal.SIGKILL)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log("[mlea] process group did not die after SIGKILL")


def run_one(agent: Agent, task: Task, config: RunConfig) -> RunResult:
    """Run one agent against one competition, collecting artifacts.

    Never raises for anything the agent does -- a crashed, hung, or silent agent
    is a *result* and is recorded as one. It raises only for harness faults the
    caller must fix, such as writing over a completed run.
    """
    run_dir = config.output_root / task.slug
    ws = Workspace(run_dir)

    if ws.metadata_path.exists() and not config.force:
        raise HarnessError(
            f"{run_dir} already holds a completed run. Overwriting it would "
            f"replace a recorded result, which is how a medal rate becomes "
            f"fiction. Pass force=True only if you mean to discard it."
        )
    if run_dir.exists() and config.force:
        shutil.rmtree(run_dir)
    ws.create()

    if not task.data_dir.exists():
        raise HarnessError(
            f"{task.competition_id}: data directory {task.data_dir} does not "
            f"exist. Run `mlebench prepare -c {task.competition_id}` first."
        )

    subs = ws.substitutions(task)
    subs["time_cap_seconds"] = str(int(config.time_cap_seconds))
    env = os.environ.copy()
    env.update({k.upper(): v for k, v in subs.items()})
    # Upstream's own agent images additionally export AGENT_DIR (see
    # agents/.shared_env). Upstream does NOT export DATA_DIR -- it hardcodes the
    # read-only mount at /home/data and tells the agent about it in
    # instructions.txt. We export DATA_DIR because without a container there is
    # no fixed mount point to hardcode; agents ported from upstream will expect
    # /home/data, so a port needs one line of glue.
    env["AGENT_DIR"] = str(ws.root)
    env.update(config.env)

    command = list(agent.build_command(task, ws))
    checkpointer = _Checkpointer(
        ws, config.checkpoint_marks, config.time_cap_seconds, config.submission_glob
    )

    start = time.monotonic()
    timed_out = False
    harness_error: str | None = None

    # Two log files, deliberately. The agent's output goes to agent.log and is
    # never trusted for classification; the harness's own observations go to
    # harness.log. Interleaving them would let an agent print "Spot instance
    # interruption" and have its failure reclassified as our fault and retried.
    with (
        ws.harness_log_path.open("w", encoding="utf-8", errors="replace") as log_file,
        ws.agent_log_path.open("w", encoding="utf-8", errors="replace") as agent_out,
    ):

        def log(msg: str) -> None:
            log_file.write(msg + "\n")
            log_file.flush()

        log(f"[mlea] competition={task.competition_id} seed={task.seed}")
        log(f"[mlea] isolation={config.isolation} cap={config.time_cap_seconds:.0f}s")
        if config.isolation == "none":
            log("[mlea] WARNING: no sandbox; the agent has network access and "
                "could retrieve public solutions for this competition")
            if answers_reachable(task.data_dir):
                log(
                    "[mlea] WARNING: prepared/private is a sibling of the data "
                    "directory and nothing prevents the agent reading it. "
                    "Upstream mounts the answers at an unrelated path, mode 700; "
                    "without a container there is no such boundary. Treat any "
                    "score from this run as void."
                )
        log(f"[mlea] agent={getattr(agent, 'name', '?')} argv={len(command)} token(s)")

        try:
            proc = subprocess.Popen(
                command,
                cwd=str(ws.code_dir),
                env=env,
                stdout=agent_out,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # its own process group, so we can kill it all
            )
        except (OSError, ValueError) as exc:
            harness_error = f"failed to launch agent: {exc}"
            log(f"[mlea] {harness_error}")
            wall = time.monotonic() - start
            _write_metadata(ws, task, config, 127, wall, False, [], harness_error)
            return RunResult(task, run_dir, 127, wall, False,
                             ws.submission_path.exists(), (), harness_error)

        checkpointer.start()
        try:
            proc.wait(timeout=config.time_cap_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_group(proc, config.grace_seconds, log)
        finally:
            checkpointer.stop()
            checkpointer.join(timeout=5)

        exit_code = proc.returncode if proc.returncode is not None else -1
        wall = time.monotonic() - start
        log(f"[mlea] exit={exit_code} wall={wall:.1f}s timed_out={timed_out}")

    if config.submission_glob:
        # Final mirror: the agent's last write may land after the last mark.
        mirror_submission(ws, config.submission_glob)
    has_submission = ws.submission_path.exists()
    checkpoints = tuple(checkpointer.checkpoints)
    _write_metadata(ws, task, config, exit_code, wall, timed_out,
                    checkpoints, harness_error)
    return RunResult(task, run_dir, exit_code, wall, timed_out,
                     has_submission, checkpoints, harness_error)


def _write_metadata(
    ws: Workspace,
    task: Task,
    config: RunConfig,
    exit_code: int,
    wall: float,
    timed_out: bool,
    checkpoints: Sequence[Checkpoint],
    harness_error: str | None,
) -> None:
    """Write the metadata triage reads.

    ``isolation`` is recorded so it can flow into the comparison fingerprint --
    an unsandboxed run must never be silently compared against a sandboxed one.
    """
    sub = ws.submission_path
    payload = {
        "competition_id": task.competition_id,
        "seed": task.seed,
        "exit_code": exit_code,
        "wall_clock_seconds": round(wall, 3),
        "time_cap_seconds": config.time_cap_seconds,
        "timed_out": timed_out,
        "isolation": config.isolation,
        "has_submission": sub.exists(),
        "submission_sha256": _sha256(sub) if sub.exists() else None,
        "checkpoints": [
            {
                "elapsed_seconds": c.elapsed_seconds,
                "sha256": c.sha256,
                "path": str(c.path.relative_to(ws.root)),
            }
            for c in checkpoints
        ],
    }
    if harness_error:
        # Surfaced to triage as an infra signature: a harness fault is our
        # fault, and must not be counted against the agent.
        payload["harness_error"] = harness_error
    ws.metadata_path.write_text(json.dumps(payload, indent=2))


def run_sweep(
    agent: Agent,
    tasks: Sequence[Task],
    config: RunConfig,
    *,
    on_result=None,
) -> list[RunResult]:
    """Run every task, in order.

    A harness fault on one task does not abandon the sweep -- it is recorded and
    the sweep continues, because losing 20 completed runs to the 21st failing is
    a far worse outcome than a partial sweep.
    """
    results: list[RunResult] = []
    for task in tasks:
        try:
            result = run_one(agent, task, config)
        except HarnessError as exc:
            result = RunResult(
                task, config.output_root / task.slug, -1, 0.0, False, False,
                (), str(exc),
            )
        results.append(result)
        if on_result is not None:
            on_result(result)
    write_submissions_jsonl(results, config.output_root / "submissions.jsonl")
    return results


def write_submissions_jsonl(results: Sequence[RunResult], path: Path) -> int:
    """Write the JSONL that upstream ``mlebench grade`` consumes.

    Schema verified against ``mlebench/grade.py``: only ``competition_id`` and
    ``submission_path`` are read, and extra keys are ignored. The path must end
    in ``.csv`` -- upstream scores any other suffix as "no submission" with only
    a warning, so a wrong extension fails silently rather than loudly.

    Only runs with a submission on disk are listed; there is nothing to grade
    otherwise. Returns the number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in results if r.submission_path.exists()]
    with path.open("w") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {
                        "competition_id": r.task.competition_id,
                        "submission_path": str(r.submission_path),
                        # Upstream reads only the two keys above and ignores the
                        # rest; these let our own grader map a score back to the
                        # run that produced it.
                        "seed": r.task.seed,
                        "run_dir": str(r.run_dir),
                    }
                )
                + "\n"
            )
    return len(rows)


__all__ = [
    "Agent",
    "answers_reachable",
    "resolve_competition_data_dir",
    "Checkpoint",
    "CommandAgent",
    "DEFAULT_CHECKPOINT_MARKS",
    "HarnessError",
    "RunConfig",
    "RunResult",
    "Task",
    "Workspace",
    "mirror_submission",
    "run_one",
    "run_sweep",
    "write_submissions_jsonl",
]
