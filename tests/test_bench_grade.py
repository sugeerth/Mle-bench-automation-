import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlea.bench import SUITE, CompetitionSpec, make_competition, make_suite
from mlea.grade import (
    grade_submission,
    medal_ranks,
    read_submission,
    thresholds_from_leaderboard,
)
from mlea.metrics import InvalidSubmission, get_metric


@pytest.fixture(scope="module")
def comp(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    return make_competition(
        CompetitionSpec("t-binary", "binary", n_train=600, n_test=300,
                        difficulty=0.4, n_teams=200, seed=7),
        root,
    )


@pytest.fixture(scope="module")
def reg(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    return make_competition(
        CompetitionSpec("t-reg", "regression", n_train=600, n_test=300,
                        difficulty=0.4, n_teams=200, seed=8),
        root,
    )


# --- medal rank tiers, verified against upstream's rank_score ---


@pytest.mark.parametrize(
    "n,expected_gold",
    [(10, 1), (50, 5), (99, 9), (100, 10), (249, 10), (250, 10), (999, 11),
     (1000, 12), (4000, 18)],
)
def test_medal_rank_tiers_match_upstream(n, expected_gold):
    assert medal_ranks(n)[0] == expected_gold


def test_medal_ranks_never_invert():
    for n in range(1, 3000, 7):
        g, s, b = medal_ranks(n)
        assert 1 <= g <= s <= b <= n


def test_thresholds_respect_metric_direction():
    scores = [0.1, 0.5, 0.9]
    hi = thresholds_from_leaderboard(scores, greater_is_better=True)
    lo = thresholds_from_leaderboard(scores, greater_is_better=False)
    assert hi.gold == 0.9 and lo.gold == 0.1


# --- generated competitions ---


def test_layout_matches_the_mlebench_prepared_shape(comp):
    for rel in ("prepared/public/train.csv", "prepared/public/test.csv",
                "prepared/public/sample_submission.csv",
                "prepared/public/description.md",
                "prepared/private/answers.json", "competition.json"):
        assert (comp / rel).exists(), rel


def test_answers_are_not_in_the_public_directory(comp):
    """The target column must not leak into test.csv."""
    header = (comp / "prepared" / "public" / "test.csv").read_text().split("\n")[0]
    assert "target" not in header
    train_header = (comp / "prepared" / "public" / "train.csv").read_text().split("\n")[0]
    assert train_header.endswith("target")


def test_generation_is_deterministic(tmp_path):
    spec = CompetitionSpec("d", "binary", n_train=200, n_test=100, n_teams=50, seed=3)
    a = make_competition(spec, tmp_path / "a")
    b = make_competition(spec, tmp_path / "b")
    assert (a / "prepared/public/train.csv").read_bytes() == \
           (b / "prepared/public/train.csv").read_bytes()
    assert json.loads((a / "competition.json").read_text())["thresholds"] == \
           json.loads((b / "competition.json").read_text())["thresholds"]


@pytest.mark.parametrize("fixture", ["comp", "reg"])
def test_gold_is_achievable_but_behind_the_oracle(fixture, request):
    """The core sanity property. Gold above the oracle would be unwinnable;
    gold far below it would make the competition trivial."""
    c = request.getfixturevalue(fixture)
    meta = json.loads((c / "competition.json").read_text())
    metric = get_metric(meta["metric"])
    gold, oracle = meta["thresholds"]["gold"], meta["oracle_score"]
    if metric.greater_is_better:
        assert gold < oracle
    else:
        assert gold > oracle


def test_harder_competitions_have_worse_thresholds(tmp_path):
    scores = []
    for i, d in enumerate((0.2, 0.8)):
        c = make_competition(
            CompetitionSpec(f"d{i}", "binary", n_train=600, n_test=300,
                            difficulty=d, n_teams=100, seed=11), tmp_path)
        scores.append(json.loads((c / "competition.json").read_text())["thresholds"]["gold"])
    assert scores[0] > scores[1]


def test_leaderboard_has_real_spread(comp):
    lb = json.loads((comp / "leaderboard.json").read_text())
    assert len(lb) >= 100
    assert max(lb) - min(lb) > 0.01, "a collapsed leaderboard makes medals meaningless"


def test_suite_covers_both_metric_directions():
    metrics = {get_metric(s.metric).greater_is_better for s in SUITE}
    assert metrics == {True, False}


def test_make_suite_creates_every_competition(tmp_path):
    assert len(make_suite(tmp_path)) == len(SUITE)


def test_invalid_spec_rejected():
    with pytest.raises(ValueError, match="task must be"):
        CompetitionSpec("x", "not-a-task")
    with pytest.raises(ValueError, match="difficulty"):
        CompetitionSpec("x", "binary", difficulty=1.5)


# --- grading ---


def _write(path, rows, header=("id", "target")):
    path.write_text(",".join(header) + "\n" + "".join(f"{a},{b}\n" for a, b in rows))


def _test_ids(comp):
    return list(json.loads((comp / "prepared/private/answers.json").read_text()))


def test_oracle_submission_medals(comp, tmp_path):
    """Grade the answers themselves: must be valid and win gold."""
    answers = json.loads((comp / "prepared/private/answers.json").read_text())
    p = tmp_path / "s.csv"
    _write(p, answers.items())
    r = grade_submission(p, comp)
    assert r.valid_submission and r.gold_medal and r.any_medal


def test_constant_submission_does_not_medal(comp, tmp_path):
    p = tmp_path / "s.csv"
    _write(p, [(i, 0.5) for i in _test_ids(comp)])
    r = grade_submission(p, comp)
    assert r.valid_submission and not r.any_medal
    assert r.score == pytest.approx(0.5)


def test_missing_file_is_not_a_submission(comp, tmp_path):
    r = grade_submission(tmp_path / "absent.csv", comp)
    assert not r.submission_exists and not r.valid_submission


def test_non_csv_path_scores_as_no_submission(comp, tmp_path):
    """Upstream does this silently; we record the reason."""
    p = tmp_path / "s.txt"
    p.write_text("id,target\n")
    r = grade_submission(p, comp)
    assert not r.submission_exists and "not a .csv" in r.error


def test_wrong_column_name_is_invalid_not_zero(comp, tmp_path):
    p = tmp_path / "s.csv"
    _write(p, [(i, 0.5) for i in _test_ids(comp)], header=("id", "prediction"))
    r = grade_submission(p, comp)
    assert r.submission_exists and not r.valid_submission
    assert "missing column" in r.error


def test_missing_rows_are_invalid(comp, tmp_path):
    p = tmp_path / "s.csv"
    _write(p, [(i, 0.5) for i in _test_ids(comp)[:-5]])
    r = grade_submission(p, comp)
    assert not r.valid_submission and "missing" in r.error


def test_unknown_ids_are_invalid(comp, tmp_path):
    p = tmp_path / "s.csv"
    _write(p, [("ghost", 0.5)] + [(i, 0.5) for i in _test_ids(comp)])
    r = grade_submission(p, comp)
    assert not r.valid_submission and "unknown" in r.error


def test_duplicate_ids_are_invalid(comp, tmp_path):
    ids = _test_ids(comp)
    p = tmp_path / "s.csv"
    _write(p, [(ids[0], 0.5)] + [(i, 0.5) for i in ids])
    r = grade_submission(p, comp)
    assert not r.valid_submission and "duplicate" in r.error


def test_non_numeric_prediction_is_invalid(comp, tmp_path):
    p = tmp_path / "s.csv"
    _write(p, [(i, "banana") for i in _test_ids(comp)])
    r = grade_submission(p, comp)
    assert not r.valid_submission and "not a number" in r.error


def test_empty_submission_is_invalid(comp, tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,target\n")
    r = grade_submission(p, comp)
    assert not r.valid_submission and "no rows" in r.error


def test_medal_tiers_are_nested(comp, tmp_path):
    """Gold implies silver implies bronze; anything else is a threshold bug."""
    answers = json.loads((comp / "prepared/private/answers.json").read_text())
    p = tmp_path / "s.csv"
    _write(p, answers.items())
    r = grade_submission(p, comp)
    assert not r.gold_medal or (r.silver_medal and r.bronze_medal)
    assert not r.silver_medal or r.bronze_medal
    assert not r.any_medal or r.above_median


def test_read_submission_reports_the_line_number(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,target\na,1\nb,oops\n")
    with pytest.raises(InvalidSubmission, match="line 3"):
        read_submission(p, "id", "target")


# --- reference agents ---


@pytest.mark.skipif(os.name != "posix", reason="POSIX only")
@pytest.mark.parametrize(
    "strategy,expect_valid",
    [("constant", True), ("linear", True), ("tuned", True), ("broken", False)],
)
def test_reference_agents_run_and_grade(comp, tmp_path, strategy, expect_valid):
    sub = tmp_path / f"{strategy}.csv"
    env = dict(os.environ, DATA_DIR=str(comp / "prepared" / "public"),
               SUBMISSION_PATH=str(sub))
    r = subprocess.run(
        [sys.executable, "-m", "mlea.baseline", "--strategy", strategy],
        env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    assert grade_submission(sub, comp).valid_submission is expect_valid


def _score(comp, tmp_path, strategy):
    sub = tmp_path / f"{strategy}-{comp.name}.csv"
    env = dict(os.environ, DATA_DIR=str(comp / "prepared" / "public"),
               SUBMISSION_PATH=str(sub))
    subprocess.run([sys.executable, "-m", "mlea.baseline", "--strategy", strategy],
                   env=env, capture_output=True, text=True, timeout=300, check=True)
    return grade_submission(sub, comp).score


@pytest.mark.skipif(os.name != "posix", reason="POSIX only")
def test_modelling_beats_a_constant_baseline(comp, tmp_path):
    assert _score(comp, tmp_path, "linear") > _score(comp, tmp_path, "constant")
    assert _score(comp, tmp_path, "tuned") > _score(comp, tmp_path, "constant")


@pytest.mark.skipif(os.name != "posix", reason="POSIX only")
def test_extra_capacity_pays_once_there_is_enough_data(tmp_path):
    """A real learning curve: the capacity search loses on a small training set
    (it overfits its own holdout) and wins on a larger one."""
    small = make_competition(
        CompetitionSpec("lc-small", "binary", n_train=600, n_test=500,
                        difficulty=0.4, n_teams=100, seed=7), tmp_path)
    large = make_competition(
        CompetitionSpec("lc-large", "binary", n_train=6000, n_test=500,
                        difficulty=0.4, n_teams=100, seed=7), tmp_path)
    small_delta = _score(small, tmp_path, "tuned") - _score(small, tmp_path, "linear")
    large_delta = _score(large, tmp_path, "tuned") - _score(large, tmp_path, "linear")
    assert large_delta > small_delta
    assert large_delta > 0.01


@pytest.mark.skipif(os.name != "posix", reason="POSIX only")
@pytest.mark.parametrize("strategy,code", [("silent", 0), ("crash", 1), ("hungry", 1)])
def test_failure_strategies_behave(comp, tmp_path, strategy, code):
    sub = tmp_path / "s.csv"
    env = dict(os.environ, DATA_DIR=str(comp / "prepared" / "public"),
               SUBMISSION_PATH=str(sub))
    r = subprocess.run([sys.executable, "-m", "mlea.baseline", "--strategy", strategy],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == code
    assert not sub.exists()
