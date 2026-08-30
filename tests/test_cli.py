import json

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
        (d / "logs" / "run.log").write_text(log)


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
    _mkrun(g, "c2", {"exit_code": 1}, log="preempted")
    out = tmp_path / "rs.json"
    assert main(["triage", str(g), "--emit-runset", str(out), "--split-id", "low"]) == 0
    rs = RunSet.from_json(out)
    assert rs.fingerprint.split_id == "low"
    assert rs.n_infra_failures == 1
    assert rs.competitions() == {"c1"}, "infra failures are excluded from capability"
