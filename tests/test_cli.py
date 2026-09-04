import json
import sys

import pytest

from mlea.cli import main


def write_runs(path, label, split, medals, seeds=3):
    path.write_text(
        json.dumps(
            {
                "label": label,
                "fingerprint": {"split_id": split},
                "runs": [
                    {"competition_id": c, "seed": s, "any_medal": s < k}
                    for c, k in medals.items()
                    for s in range(seeds)
                ],
            }
        )
    )


def test_power_preset_runs(capsys):
    assert main(["power", "--design", "live-gap"]) == 0
    out = capsys.readouterr().out
    assert "live-gap" in out and "48 runs" in out


def test_power_reports_effect(capsys):
    assert main(["power", "--design", "lite-regression", "--effect", "0.3"]) == 0
    assert "power" in capsys.readouterr().out


def test_power_rejects_unknown_design():
    with pytest.raises(SystemExit):
        main(["power", "--design", "nope"])


def test_compare_refuses_mismatched_splits(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_runs(a, "a", "split75", {"c1": 3})
    write_runs(b, "b", "low", {"c1": 3})
    assert main(["compare", str(a), str(b)]) == 2
    assert "refusing to compare" in capsys.readouterr().err


def test_compare_succeeds_on_matching_splits(tmp_path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    m = {f"c{i}": 2 for i in range(22)}
    write_runs(a, "a", "low", m)
    write_runs(b, "b", "low", m)
    assert main(["compare", str(a), str(b)]) == 0
    assert "difference" in capsys.readouterr().out


def test_regression_gate_fails_on_significant_drop(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    write_runs(a, "base", "low", {f"c{i}": 3 for i in range(22)})
    write_runs(b, "cand", "low", {f"c{i}": 0 for i in range(22)})
    assert main(["compare", str(a), str(b), "--fail-on-regression"]) == 1


def test_regression_gate_passes_on_neutral_change(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    m = {f"c{i}": 2 for i in range(22)}
    write_runs(a, "base", "low", m)
    write_runs(b, "cand", "low", m)
    assert main(["compare", str(a), str(b), "--fail-on-regression"]) == 0


def test_seeds_subcommand(capsys):
    assert main(["seeds", "--design", "full-regression", "--effect", "0.15"]) == 0
    assert "seed" in capsys.readouterr().out


def test_compare_with_pairs(tmp_path, capsys):
    a, b, p = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "p.json"
    write_runs(a, "pre", "pre-cutoff", {f"p{i}": 3 for i in range(8)})
    write_runs(b, "post", "post-cutoff", {f"q{i}": 0 for i in range(8)})
    p.write_text(json.dumps([[f"p{i}", f"q{i}"] for i in range(8)]))
    assert main(["compare", str(a), str(b), "--pairs", str(p)]) == 0
    assert "matched pairs" in capsys.readouterr().out


def _mkrun(root, name, meta, submission=None, log=""):
    d = root / name
    (d / "logs").mkdir(parents=True)
    d.joinpath("metadata.json").write_text(json.dumps(meta))
    if submission is not None:
        (d / "submission").mkdir()
        (d / "submission" / "submission.csv").write_text(submission)
    if log:
        # harness.log: the trusted log, where infra signatures are read from
        (d / "logs" / "harness.log").write_text(log)


def test_triage_cli(tmp_path, capsys):
    g = tmp_path / "group"
    g.mkdir()
    _mkrun(g, "c1", {"exit_code": 0}, submission="a,b\n1,2\n")
    _mkrun(g, "c2", {"exit_code": 1}, log="Spot instance interruption notice")
    assert main(["triage", str(g), "-v"]) == 0
    out = capsys.readouterr().out
    assert "capability signal" in out and "our fault" in out


def test_triage_cli_empty_group(tmp_path, capsys):
    g = tmp_path / "empty"
    g.mkdir()
    assert main(["triage", str(g)]) == 2
    assert "no run directories" in capsys.readouterr().err


def test_triage_emits_a_loadable_runset(tmp_path):
    from mlea.records import RunSet

    g = tmp_path / "group"
    g.mkdir()
    _mkrun(g, "c1", {"exit_code": 0}, submission="a,b\n1,2\n")
    _mkrun(g, "c2", {"exit_code": 1}, log="Spot ITN received. Interrupting.")
    out = tmp_path / "rs.json"
    assert main(["triage", str(g), "--emit-runset", str(out), "--split-id", "low"]) == 0
    rs = RunSet.from_json(out)
    assert rs.fingerprint.split_id == "low"
    assert rs.n_infra_failures == 1
    assert rs.competitions() == {"c1"}, "infra failures are excluded from capability"


# --- `mlea run`, and the whole pipeline end to end ---

import os as _os

import pytest as _pytest

posix_only = _pytest.mark.skipif(_os.name != "posix", reason="POSIX only")


def _data_root(tmp_path, *competitions):
    root = tmp_path / "data"
    for c in competitions:
        (root / c).mkdir(parents=True)
        (root / c / "train.csv").write_text("id,y\n1,0\n")
    return root


@posix_only
def test_run_executes_and_reports(tmp_path, capsys):
    root = _data_root(tmp_path, "leaf-classification")
    rc = main([
        "run", "--agent-cmd", 'printf "id,y\\n1,0\\n" > "$SUBMISSION_PATH"',
        "--data-root", str(root), "--competition", "leaf-classification",
        "--time-cap", "10", "--out", str(tmp_path / "runs"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1/1 run(s) produced a submission" in out
    assert "mlebench grade --submission" in out


@posix_only
def test_run_warns_about_missing_sandbox(tmp_path, capsys):
    root = _data_root(tmp_path, "c1")
    main(["run", "--agent-cmd", "true", "--data-root", str(root),
          "--competition", "c1", "--time-cap", "5", "--out", str(tmp_path / "r")])
    assert "isolation=none" in capsys.readouterr().err


def test_run_requires_competitions(tmp_path, capsys):
    assert main(["run", "--agent-cmd", "true", "--data-root", str(tmp_path),
                 "--out", str(tmp_path / "r")]) == 2
    assert "--competition" in capsys.readouterr().err


@posix_only
def test_run_reads_a_competition_set_file(tmp_path):
    root = _data_root(tmp_path, "c1", "c2")
    setfile = tmp_path / "split.txt"
    setfile.write_text("# comment\nc1\n\nc2\n")
    main(["run", "--agent-cmd", 'printf "a\\n1\\n" > "$SUBMISSION_PATH"',
          "--data-root", str(root), "--competition-set", str(setfile),
          "--time-cap", "10", "--out", str(tmp_path / "runs")])
    assert (tmp_path / "runs" / "c1__seed0" / "metadata.json").exists()
    assert (tmp_path / "runs" / "c2__seed0" / "metadata.json").exists()


@posix_only
def test_run_multiple_seeds(tmp_path):
    root = _data_root(tmp_path, "c1")
    main(["run", "--agent-cmd", 'printf "a\\n1\\n" > "$SUBMISSION_PATH"',
          "--data-root", str(root), "--competition", "c1", "--seeds", "3",
          "--time-cap", "10", "--out", str(tmp_path / "runs")])
    dirs = sorted(p.name for p in (tmp_path / "runs").iterdir() if p.is_dir())
    assert dirs == ["c1__seed0", "c1__seed1", "c1__seed2"]


@posix_only
def test_full_pipeline_run_triage_compare(tmp_path):
    """run -> triage -> runset -> compare, with no hand-written fixtures."""
    from mlea.records import RunSet

    root = _data_root(tmp_path, "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
    comps = [f"c{i}" for i in range(1, 9)]
    args = ["run", "--data-root", str(root), "--time-cap", "10", "--seeds", "2"]
    for c in comps:
        args += ["--competition", c]

    # Baseline writes a submission; candidate crashes on half the competitions.
    assert main(args + ["--agent-cmd", 'printf "a\\n1\\n" > "$SUBMISSION_PATH"',
                        "--out", str(tmp_path / "base")]) == 0
    assert main(args + [
        "--agent-cmd",
        'case "$COMPETITION_ID" in c1|c2|c3|c4) exit 1 ;; '
        'esac; printf "a\\n1\\n" > "$SUBMISSION_PATH"',
        "--out", str(tmp_path / "cand")]) == 0

    for name in ("base", "cand"):
        assert main(["triage", str(tmp_path / name),
                     "--emit-runset", str(tmp_path / f"{name}.json"),
                     "--split-id", "toy", "--label", name]) == 0

    base = RunSet.from_json(tmp_path / "base.json")
    cand = RunSet.from_json(tmp_path / "cand.json")
    # Isolation was carried from the harness into the fingerprint.
    assert base.fingerprint.container_config == "none"
    assert len(base.runs) == 16 and len(cand.runs) == 16
    assert main(["compare", str(tmp_path / "base.json"),
                 str(tmp_path / "cand.json")]) == 0


@posix_only
def test_sandboxed_and_unsandboxed_runs_cannot_be_compared(tmp_path, capsys):
    """The isolation level must survive all the way to the comparison guard."""
    root = _data_root(tmp_path, "c1")
    cmd = 'printf "a\\n1\\n" > "$SUBMISSION_PATH"'
    main(["run", "--agent-cmd", cmd, "--data-root", str(root), "--competition", "c1",
          "--time-cap", "10", "--isolation", "none", "--out", str(tmp_path / "a")])
    main(["run", "--agent-cmd", cmd, "--data-root", str(root), "--competition", "c1",
          "--time-cap", "10", "--isolation", "docker", "--out", str(tmp_path / "b")])
    for n in ("a", "b"):
        main(["triage", str(tmp_path / n), "--emit-runset",
              str(tmp_path / f"{n}.json"), "--split-id", "toy"])
    capsys.readouterr()
    assert main(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 2
    assert "container_config" in capsys.readouterr().err


@posix_only
def test_grade_hint_includes_required_output_dir(tmp_path, capsys):
    """`mlebench grade` requires --output-dir; without it the command exits."""
    root = _data_root(tmp_path, "c1")
    main(["run", "--agent-cmd", 'printf "a\\n1\\n" > "$SUBMISSION_PATH"',
          "--data-root", str(root), "--competition", "c1",
          "--time-cap", "10", "--out", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert "--submission" in out and "--output-dir" in out


@posix_only
def test_run_uses_the_prepared_public_dir(tmp_path):
    """A real mlebench data root must resolve past prepared/ to public/."""
    root = tmp_path / "data"
    pub = root / "c1" / "prepared" / "public"
    pub.mkdir(parents=True)
    (pub / "train.csv").write_text("id,y\n1,0\n")
    (root / "c1" / "prepared" / "private").mkdir(parents=True)
    (root / "c1" / "prepared" / "private" / "test.csv").write_text("secret\n")
    rc = main(["run", "--agent-cmd",
               'test -f "$DATA_DIR/train.csv" && echo "at:$DATA_DIR" '
               '&& printf "a\\n1\\n" > "$SUBMISSION_PATH"',
               "--data-root", str(root), "--competition", "c1",
               "--time-cap", "10", "--out", str(tmp_path / "runs")])
    assert rc == 0
    run_dir = tmp_path / "runs" / "c1__seed0"
    assert json.loads((run_dir / "metadata.json").read_text())["exit_code"] == 0
    assert "prepared/public" in (run_dir / "logs" / "agent.log").read_text()


@posix_only
def test_unsandboxed_run_warns_that_answers_are_reachable(tmp_path):
    """prepared/private is a sibling of public; without a container the agent
    can simply read it. Upstream mounts it elsewhere, mode 700."""
    root = tmp_path / "data"
    pub = root / "c1" / "prepared" / "public"
    pub.mkdir(parents=True)
    (pub / "train.csv").write_text("id,y\n1,0\n")
    (root / "c1" / "prepared" / "private").mkdir(parents=True)
    main(["run", "--agent-cmd", 'printf "a\\n1\\n" > "$SUBMISSION_PATH"',
          "--data-root", str(root), "--competition", "c1",
          "--time-cap", "10", "--out", str(tmp_path / "runs")])
    log = (tmp_path / "runs" / "c1__seed0" / "logs" / "harness.log").read_text()
    assert "prepared/private is a sibling" in log
    assert "void" in log


@posix_only
def test_no_answers_warning_for_a_flat_data_dir(tmp_path):
    root = _data_root(tmp_path, "c1")
    main(["run", "--agent-cmd", 'printf "a\\n1\\n" > "$SUBMISSION_PATH"',
          "--data-root", str(root), "--competition", "c1",
          "--time-cap", "10", "--out", str(tmp_path / "runs")])
    log = (tmp_path / "runs" / "c1__seed0" / "logs" / "harness.log").read_text()
    assert "prepared/private" not in log


# --- the whole pipeline, against real gradeable competitions ---


@posix_only
def test_selftest_passes_end_to_end(tmp_path, capsys):
    """The only test here that runs real models against real scoring.

    Everything else in this suite uses stub agents or synthetic fixtures; this
    generates competitions, fits models, grades scores, classifies failures and
    checks that all four stages agree.
    """
    rc = main(["selftest", "--out", str(tmp_path / "st"), "--competitions", "6",
               "--seeds", "3", "--time-cap", "180"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "SELFTEST PASSED" in out
    assert "FAIL" not in out
    assert (tmp_path / "st" / "report.html").exists()


@posix_only
def test_grading_feeds_back_into_triage(tmp_path):
    """A submission with the right rows and a wrong column name looks valid on
    disk. Triage can only know otherwise if grading writes its verdict back."""
    from mlea.triage import Outcome, triage_run_group

    main(["bench", "--out", str(tmp_path / "data"), "--count", "1"])
    runs = tmp_path / "runs"
    main(["run", "--agent-name", "broken", "--data-root", str(tmp_path / "data"),
          "--competition", "synth-easy-binary", "--time-cap", "120",
          "--agent-cmd", f"{sys.executable} -m mlea.baseline --strategy broken",
          "--out", str(runs)])

    before = triage_run_group(runs)
    assert before.results[0].outcome is Outcome.VALID, "looks fine on disk"

    main(["grade", "--submission", str(runs / "submissions.jsonl"),
          "--data-root", str(tmp_path / "data"),
          "--output-dir", str(tmp_path / "grades")])

    after = triage_run_group(runs)
    assert after.results[0].outcome is Outcome.INVALID_SUBMISSION


@posix_only
def test_grade_records_medals_for_a_real_model(tmp_path, capsys):
    main(["bench", "--out", str(tmp_path / "data"), "--count", "2"])
    runs = tmp_path / "runs"
    main(["run", "--agent-name", "tuned", "--data-root", str(tmp_path / "data"),
          "--competition", "synth-easy-binary", "--competition", "synth-hard-binary",
          "--time-cap", "180",
          "--agent-cmd", f"{sys.executable} -m mlea.baseline --strategy tuned",
          "--out", str(runs)])
    capsys.readouterr()
    rc = main(["grade", "--submission", str(runs / "submissions.jsonl"),
               "--data-root", str(tmp_path / "data"),
               "--output-dir", str(tmp_path / "grades")])
    assert rc == 0
    summary = json.loads((tmp_path / "grades" / "grading_report.json").read_text())
    assert summary["n_valid"] == 2
    assert summary["n_any_medal"] >= 1, "a tuned ridge should medal on these"


@posix_only
def test_runset_carries_medals_through_to_compare(tmp_path):
    from mlea.records import RunSet

    main(["bench", "--out", str(tmp_path / "data"), "--count", "2"])
    runs = tmp_path / "runs"
    main(["run", "--agent-name", "tuned", "--data-root", str(tmp_path / "data"),
          "--competition", "synth-easy-binary", "--competition", "synth-hard-binary",
          "--time-cap", "180",
          "--agent-cmd", f"{sys.executable} -m mlea.baseline --strategy tuned",
          "--out", str(runs)])
    main(["grade", "--submission", str(runs / "submissions.jsonl"),
          "--data-root", str(tmp_path / "data"),
          "--output-dir", str(tmp_path / "grades")])
    main(["triage", str(runs), "--emit-runset", str(tmp_path / "rs.json"),
          "--split-id", "synth", "--grades", str(tmp_path / "grades" / "medals.json")])
    rs = RunSet.from_json(tmp_path / "rs.json")
    assert rs.any_medal_rate() > 0, "medals must survive the trip into the run set"


def test_bench_cli_reports_thresholds(tmp_path, capsys):
    assert main(["bench", "--out", str(tmp_path / "d"), "--count", "2"]) == 0
    out = capsys.readouterr().out
    assert "oracle=" in out and "gold=" in out
