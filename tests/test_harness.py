"""Harness tests use real subprocesses.

Process-group killing, SIGTERM grace, and mid-run snapshotting are exactly the
behaviours a mock would assert into existence while the real thing stayed
broken, so these spawn actual processes and check actual outcomes.
"""

import json
import os
import time
from pathlib import Path

import pytest

from mlea.harness import (
    CommandAgent,
    HarnessError,
    RunConfig,
    Task,
    Workspace,
    run_one,
    run_sweep,
    write_submissions_jsonl,
)
from mlea.triage import Outcome, classify, from_run_dir, triage_run_group

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="process-group termination is POSIX-only"
)

WRITE_SUB = 'printf "id,y\\n1,0\\n" > "$SUBMISSION_PATH"'


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data" / "spooky-author-identification"
    d.mkdir(parents=True)
    (d / "train.csv").write_text("id,text,author\n1,hi,EAP\n")
    return d


def cfg(tmp_path, **kw):
    kw.setdefault("time_cap_seconds", 10.0)
    kw.setdefault("checkpoint_marks", ())
    return RunConfig(output_root=tmp_path / "runs", **kw)


def task(data_dir, seed=0):
    return Task("spooky-author-identification", data_dir, seed=seed)


# --- the happy path, and the closed loop with triage ---


def test_successful_run_produces_a_triageable_directory(tmp_path, data_dir):
    r = run_one(CommandAgent("stub", WRITE_SUB), task(data_dir), cfg(tmp_path))
    assert r.exit_code == 0 and not r.timed_out and r.has_submission
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.VALID


def test_metadata_matches_what_triage_reads(tmp_path, data_dir):
    r = run_one(CommandAgent("stub", WRITE_SUB), task(data_dir, seed=3), cfg(tmp_path))
    meta = json.loads((r.run_dir / "metadata.json").read_text())
    assert meta["competition_id"] == "spooky-author-identification"
    assert meta["seed"] == 3 and meta["exit_code"] == 0
    assert meta["submission_sha256"] is not None
    a = from_run_dir(r.run_dir)
    assert a.seed == 3 and a.submission_rows == 1


def test_agent_receives_environment_and_substitutions(tmp_path, data_dir):
    agent = CommandAgent(
        "echoer",
        'echo "$COMPETITION_ID $SEED $TIME_CAP_SECONDS"; '
        'test -d "{data_dir}"; printf "id,y\\n1,0\\n" > "{submission_path}"',
    )
    r = run_one(agent, task(data_dir, seed=7), cfg(tmp_path, time_cap_seconds=30.0))
    log = (r.run_dir / "logs" / "agent.log").read_text()
    assert "spooky-author-identification 7 30" in log
    assert r.exit_code == 0


def test_agent_runs_inside_the_code_directory(tmp_path, data_dir):
    agent = CommandAgent("pwd", 'pwd > out.txt; ' + WRITE_SUB)
    r = run_one(agent, task(data_dir), cfg(tmp_path))
    assert (r.run_dir / "code" / "out.txt").exists()


# --- failures are results, not exceptions ---


def test_crashing_agent_is_recorded_not_raised(tmp_path, data_dir):
    r = run_one(CommandAgent("boom", "exit 3"), task(data_dir), cfg(tmp_path))
    assert r.exit_code == 3 and not r.has_submission
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.CRASH


def test_silent_agent_is_no_submission(tmp_path, data_dir):
    r = run_one(CommandAgent("quiet", "true"), task(data_dir), cfg(tmp_path))
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.NO_SUBMISSION


def test_unlaunchable_command_is_a_harness_error_not_an_agent_failure(tmp_path, data_dir):
    class Bad:
        name = "bad"

        def build_command(self, task, workspace):
            return ["/nonexistent/binary/xyz"]

    r = run_one(Bad(), task(data_dir), cfg(tmp_path))
    assert r.harness_error is not None
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.INFRA


def test_missing_data_dir_is_a_harness_error(tmp_path):
    t = Task("c", tmp_path / "absent")
    with pytest.raises(HarnessError, match="mlebench prepare"):
        run_one(CommandAgent("s", "true"), t, cfg(tmp_path))


# --- the time cap ---


def test_timeout_kills_the_agent(tmp_path, data_dir):
    r = run_one(
        CommandAgent("sleeper", "sleep 60"),
        task(data_dir),
        cfg(tmp_path, time_cap_seconds=1.0, grace_seconds=1.0),
    )
    assert r.timed_out
    assert r.wall_clock_seconds < 30
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.TIMEOUT


def test_submission_written_before_the_cap_survives_the_kill(tmp_path, data_dir):
    """The load-bearing rule: a truncated run that left a submission is a result."""
    r = run_one(
        CommandAgent("slow", WRITE_SUB + "; sleep 60"),
        task(data_dir),
        cfg(tmp_path, time_cap_seconds=2.0, grace_seconds=1.0),
    )
    assert r.timed_out and r.has_submission
    result = classify(from_run_dir(r.run_dir))
    assert result.outcome is Outcome.VALID
    assert result.truncated


def test_sigterm_grace_lets_an_agent_flush_its_submission(tmp_path, data_dir):
    """An agent that handles SIGTERM gets to write its best result."""
    agent = CommandAgent(
        "flusher",
        'trap \'printf "id,y\\n1,9\\n" > "$SUBMISSION_PATH"; exit 0\' TERM; '
        "sleep 60 & wait",
    )
    r = run_one(agent, task(data_dir), cfg(tmp_path, time_cap_seconds=1.0,
                                           grace_seconds=10.0))
    assert r.timed_out and r.has_submission
    assert "1,9" in (r.run_dir / "submission" / "submission.csv").read_text()


def test_child_processes_are_killed_with_the_parent(tmp_path, data_dir):
    """Signalling only the parent leaves training children holding the GPU."""
    marker = tmp_path / "child_still_alive.txt"
    agent = CommandAgent(
        "spawner",
        f"(sleep 8; touch {marker}) & sleep 60",
    )
    run_one(agent, task(data_dir),
            cfg(tmp_path, time_cap_seconds=1.0, grace_seconds=0.5))
    time.sleep(9)
    assert not marker.exists(), "orphaned child survived the process-group kill"


def test_time_cap_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="time_cap_seconds"):
        RunConfig(output_root=tmp_path, time_cap_seconds=0)


def test_isolation_is_validated(tmp_path):
    with pytest.raises(ValueError, match="isolation"):
        RunConfig(output_root=tmp_path, isolation="vm")


def test_unsandboxed_runs_warn_in_the_log(tmp_path, data_dir):
    r = run_one(CommandAgent("s", WRITE_SUB), task(data_dir), cfg(tmp_path))
    assert "no sandbox" in (r.run_dir / "logs" / "harness.log").read_text()


def test_isolation_is_recorded_for_the_fingerprint(tmp_path, data_dir):
    r = run_one(CommandAgent("s", WRITE_SUB), task(data_dir), cfg(tmp_path))
    assert json.loads((r.run_dir / "metadata.json").read_text())["isolation"] == "none"


# --- immutability ---


def test_completed_runs_are_not_silently_overwritten(tmp_path, data_dir):
    c = cfg(tmp_path)
    run_one(CommandAgent("s", WRITE_SUB), task(data_dir), c)
    with pytest.raises(HarnessError, match="already holds a completed run"):
        run_one(CommandAgent("s", WRITE_SUB), task(data_dir), c)


def test_force_overwrites_deliberately(tmp_path, data_dir):
    run_one(CommandAgent("s", WRITE_SUB), task(data_dir), cfg(tmp_path))
    r = run_one(CommandAgent("s2", 'printf "id,y\\n9,9\\n" > "$SUBMISSION_PATH"'),
                task(data_dir), cfg(tmp_path, force=True))
    assert "9,9" in r.submission_path.read_text()


def test_different_seeds_do_not_collide(tmp_path, data_dir):
    c = cfg(tmp_path)
    a = run_one(CommandAgent("s", WRITE_SUB), task(data_dir, seed=0), c)
    b = run_one(CommandAgent("s", WRITE_SUB), task(data_dir, seed=1), c)
    assert a.run_dir != b.run_dir


# --- checkpointing ---


def test_checkpoints_capture_the_submission_mid_run(tmp_path, data_dir):
    agent = CommandAgent("improver", WRITE_SUB + "; sleep 3; "
                         'printf "id,y\\n1,1\\n2,2\\n" > "$SUBMISSION_PATH"; sleep 3')
    r = run_one(agent, task(data_dir),
                cfg(tmp_path, time_cap_seconds=20.0, checkpoint_marks=(1.0, 5.0)))
    assert len(r.checkpoints) == 2
    assert r.checkpoints[0].sha256 != r.checkpoints[1].sha256
    assert (r.run_dir / "checkpoints" / "t=1" / "submission.csv").exists()


def test_unchanged_submissions_are_not_snapshotted_twice(tmp_path, data_dir):
    agent = CommandAgent("static", WRITE_SUB + "; sleep 4")
    r = run_one(agent, task(data_dir),
                cfg(tmp_path, time_cap_seconds=20.0, checkpoint_marks=(1.0, 2.0, 3.0)))
    assert len(r.checkpoints) == 1, "identical content must snapshot once"


def test_checkpointing_does_not_change_the_final_result(tmp_path, data_dir):
    """Snapshotting must be strictly additive or published numbers shift."""
    plain = run_one(CommandAgent("s", WRITE_SUB), task(data_dir),
                    RunConfig(output_root=tmp_path / "a", time_cap_seconds=10,
                              checkpoint_marks=()))
    snapped = run_one(CommandAgent("s", WRITE_SUB), task(data_dir),
                      RunConfig(output_root=tmp_path / "b", time_cap_seconds=10,
                                checkpoint_marks=(1.0, 2.0)))
    assert plain.submission_path.read_text() == snapped.submission_path.read_text()
    assert plain.exit_code == snapped.exit_code


def test_agent_that_never_writes_produces_no_checkpoints(tmp_path, data_dir):
    r = run_one(CommandAgent("late", "sleep 3"), task(data_dir),
                cfg(tmp_path, time_cap_seconds=10.0, checkpoint_marks=(1.0, 2.0)))
    assert r.checkpoints == ()


def test_marks_beyond_the_cap_are_ignored(tmp_path, data_dir):
    r = run_one(CommandAgent("s", WRITE_SUB), task(data_dir),
                cfg(tmp_path, time_cap_seconds=3.0,
                    checkpoint_marks=(1.0, 999.0)))
    assert all(c.elapsed_seconds < 3.0 for c in r.checkpoints)


def test_checkpoints_recorded_in_metadata(tmp_path, data_dir):
    r = run_one(CommandAgent("s", WRITE_SUB + "; sleep 3"), task(data_dir),
                cfg(tmp_path, time_cap_seconds=10.0, checkpoint_marks=(1.0,)))
    meta = json.loads((r.run_dir / "metadata.json").read_text())
    assert len(meta["checkpoints"]) == 1
    assert meta["checkpoints"][0]["path"] == "checkpoints/t=1/submission.csv"


# --- sweeps ---


def test_sweep_continues_past_a_harness_fault(tmp_path, data_dir):
    """Losing 20 good runs to the 21st failing is the worse outcome."""
    good = task(data_dir, seed=0)
    bad = Task("missing-comp", tmp_path / "absent", seed=0)
    results = run_sweep(CommandAgent("s", WRITE_SUB), [bad, good, task(data_dir, 1)],
                        cfg(tmp_path))
    assert len(results) == 3
    assert results[0].harness_error is not None
    assert results[1].exit_code == 0 and results[2].exit_code == 0


def test_sweep_writes_grade_ready_jsonl(tmp_path, data_dir):
    results = run_sweep(CommandAgent("s", WRITE_SUB),
                        [task(data_dir, 0), task(data_dir, 1)], cfg(tmp_path))
    lines = (tmp_path / "runs" / "submissions.jsonl").read_text().strip().split("\n")
    rows = [json.loads(x) for x in lines]
    assert len(rows) == 2
    assert {"competition_id", "submission_path"} <= set(rows[0])
    assert all(os.path.exists(r["submission_path"]) for r in rows)


def test_jsonl_omits_runs_without_a_submission(tmp_path, data_dir):
    results = run_sweep(CommandAgent("mix", "exit 1"), [task(data_dir)], cfg(tmp_path))
    assert write_submissions_jsonl(results, tmp_path / "x.jsonl") == 0


def test_sweep_output_is_triageable_as_a_group(tmp_path, data_dir):
    run_sweep(CommandAgent("s", WRITE_SUB), [task(data_dir, 0), task(data_dir, 1)],
              cfg(tmp_path))
    report = triage_run_group(tmp_path / "runs")
    assert report.total == 2 and len(report.gradeable) == 2


def test_on_result_callback_fires_per_run(tmp_path, data_dir):
    seen = []
    run_sweep(CommandAgent("s", WRITE_SUB), [task(data_dir, 0), task(data_dir, 1)],
              cfg(tmp_path), on_result=seen.append)
    assert len(seen) == 2


# --- regressions found by running the thing for real ---


def test_relative_output_root_still_reaches_the_agent(tmp_path, data_dir, monkeypatch):
    """The agent's cwd is its code dir, so a relative --out sent SUBMISSION_PATH
    somewhere that did not exist and every run silently produced nothing."""
    monkeypatch.chdir(tmp_path)
    r = run_one(
        CommandAgent("s", WRITE_SUB),
        Task("spooky-author-identification", data_dir),
        RunConfig(output_root="runs", time_cap_seconds=10, checkpoint_marks=()),
    )
    assert r.exit_code == 0 and r.has_submission
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.VALID


def test_relative_data_dir_is_resolved(tmp_path, data_dir, monkeypatch):
    monkeypatch.chdir(data_dir.parent)
    t = Task("spooky-author-identification", Path("spooky-author-identification"))
    r = run_one(CommandAgent("s", 'test -f "$DATA_DIR/train.csv" && ' + WRITE_SUB),
                t, RunConfig(output_root=tmp_path / "runs", time_cap_seconds=10,
                             checkpoint_marks=()))
    assert r.exit_code == 0


def test_the_agent_command_is_not_written_where_triage_scans_it(tmp_path, data_dir):
    """Logging the command into a trusted log made every run match its own
    signatures -- the command text became the evidence."""
    agent = CommandAgent("evil", 'echo "Spot Instance interruption notice"; exit 1')
    r = run_one(agent, task(data_dir), cfg(tmp_path))
    harness_log = (r.run_dir / "logs" / "harness.log").read_text()
    assert "Spot Instance interruption" not in harness_log
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.CRASH


def test_agent_stdout_and_harness_log_are_separate_files(tmp_path, data_dir):
    r = run_one(CommandAgent("s", 'echo hello; ' + WRITE_SUB), task(data_dir),
                cfg(tmp_path))
    assert "hello" in (r.run_dir / "logs" / "agent.log").read_text()
    assert "hello" not in (r.run_dir / "logs" / "harness.log").read_text()
    assert "[mlea]" in (r.run_dir / "logs" / "harness.log").read_text()


# --- upstream layout, verified against openai/mle-bench source ---


def test_prepared_layout_resolves_to_public_not_the_competition_dir(tmp_path):
    """`mlebench prepare` writes <id>/prepared/{public,private}. Handing an agent
    the competition dir would hand it prepared/private -- the answers."""
    from mlea.harness import resolve_competition_data_dir

    root = tmp_path / "data"
    (root / "leaf-classification" / "prepared" / "public").mkdir(parents=True)
    (root / "leaf-classification" / "prepared" / "private").mkdir(parents=True)
    (root / "leaf-classification" / "raw").mkdir(parents=True)
    resolved = resolve_competition_data_dir(root, "leaf-classification")
    assert resolved.name == "public"
    assert resolved.parent.name == "prepared"
    assert "private" not in str(resolved)


def test_flat_layout_still_works(tmp_path):
    root = tmp_path / "data"
    (root / "c1").mkdir(parents=True)
    from mlea.harness import resolve_competition_data_dir

    assert resolve_competition_data_dir(root, "c1") == (root / "c1").resolve() or \
        resolve_competition_data_dir(root, "c1") == root / "c1"


def test_agent_dir_is_exported(tmp_path, data_dir):
    """Upstream's agent images export AGENT_DIR alongside the other three."""
    r = run_one(CommandAgent("e", 'echo "AGENT_DIR=$AGENT_DIR"; ' + WRITE_SUB),
                task(data_dir), cfg(tmp_path))
    assert "AGENT_DIR=" in (r.run_dir / "logs" / "agent.log").read_text()


def test_submission_paths_in_jsonl_are_csv(tmp_path, data_dir):
    """Upstream silently scores a non-.csv submission_path as no-submission."""
    results = run_sweep(CommandAgent("s", WRITE_SUB), [task(data_dir)], cfg(tmp_path))
    rows = [
        json.loads(x)
        for x in (tmp_path / "runs" / "submissions.jsonl").read_text().splitlines()
    ]
    assert all(r["submission_path"].endswith(".csv") for r in rows)
    # Upstream reads only the first two and ignores extras; ours needs the rest
    # to map a score back to the run that produced it.
    assert all(
        {"competition_id", "submission_path"} <= set(r) for r in rows
    )
    assert all(set(r) == {"competition_id", "submission_path", "seed", "run_dir"}
               for r in rows)


# --- mirroring an agent's own submission path ---


def test_mirrors_the_agents_own_submission(tmp_path, data_dir):
    """AIDE writes workspaces/0-run/working/submission.csv and nothing else."""
    agent = CommandAgent(
        "aide-like",
        'mkdir -p workspaces/0-run/working && '
        'printf "id,y\\n1,7\\n" > workspaces/0-run/working/submission.csv',
    )
    r = run_one(agent, task(data_dir),
                cfg(tmp_path, submission_glob="workspaces/0-run/working/submission.csv"))
    assert r.has_submission
    assert "1,7" in r.submission_path.read_text()
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.VALID


def test_mirror_glob_matches_a_timestamped_directory(tmp_path, data_dir):
    """MLEvolve writes runs/<timestamp>_<id>/workspace/best_submission/."""
    agent = CommandAgent(
        "mlevolve-like",
        'd=runs/20260830_120000_c1/workspace/best_submission && mkdir -p "$d" && '
        'printf "id,y\\n2,2\\n" > "$d/submission.csv"',
    )
    r = run_one(agent, task(data_dir),
                cfg(tmp_path,
                    submission_glob="runs/*/workspace/best_submission/submission.csv"))
    assert "2,2" in r.submission_path.read_text()


def test_mirror_picks_the_newest_match(tmp_path, data_dir):
    agent = CommandAgent(
        "multi",
        'mkdir -p a b && printf "id,y\\n1,1\\n" > a/submission.csv && sleep 1 && '
        'printf "id,y\\n9,9\\n" > b/submission.csv',
    )
    r = run_one(agent, task(data_dir), cfg(tmp_path, submission_glob="*/submission.csv"))
    assert "9,9" in r.submission_path.read_text()


def test_mirror_builds_a_curve_from_the_agents_path(tmp_path, data_dir):
    agent = CommandAgent(
        "improving",
        'mkdir -p w && printf "id,y\\n1,1\\n" > w/submission.csv && sleep 3 && '
        'printf "id,y\\n1,1\\n2,2\\n" > w/submission.csv && sleep 3',
    )
    r = run_one(agent, task(data_dir),
                cfg(tmp_path, time_cap_seconds=20.0, checkpoint_marks=(1.0, 5.0),
                    submission_glob="w/submission.csv"))
    assert len(r.checkpoints) == 2
    assert r.checkpoints[0].sha256 != r.checkpoints[1].sha256


def test_missing_mirror_source_is_not_a_harness_error(tmp_path, data_dir):
    """An agent that produced nothing is a result, not a fault."""
    r = run_one(CommandAgent("silent", "true"), task(data_dir),
                cfg(tmp_path, submission_glob="nowhere/submission.csv"))
    assert r.harness_error is None
    assert not r.has_submission
    assert classify(from_run_dir(r.run_dir)).outcome is Outcome.NO_SUBMISSION


def test_empty_mirror_source_is_ignored(tmp_path, data_dir):
    r = run_one(CommandAgent("empty", "mkdir -p w && : > w/submission.csv"),
                task(data_dir), cfg(tmp_path, submission_glob="w/submission.csv"))
    assert not r.has_submission


def test_mirror_is_opt_in(tmp_path, data_dir):
    """Without the flag, only $SUBMISSION_PATH counts."""
    r = run_one(CommandAgent("aide-like",
                             'mkdir -p w && printf "a\\n1\\n" > w/submission.csv'),
                task(data_dir), cfg(tmp_path))
    assert not r.has_submission
