import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mlea.baseline import clip_to, impute, suspicious_features
from mlea.bench import CHALLENGES, CompetitionSpec, make_competition
from mlea.skills import SkillCell, SkillProfile

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX only")


# --- the matched control is the whole design ---


def test_control_differs_only_by_the_challenge():
    spec = CompetitionSpec("c", "binary", seed=11, challenges=frozenset({"leakage"}))
    ctrl = spec.control()
    assert ctrl.challenges == frozenset()
    assert ctrl.seed == spec.seed and ctrl.difficulty == spec.difficulty
    assert ctrl.n_train == spec.n_train and ctrl.n_features == spec.n_features


def test_unknown_challenge_is_rejected():
    with pytest.raises(ValueError, match="unknown challenge"):
        CompetitionSpec("c", "binary", challenges=frozenset({"gremlins"}))


def test_every_challenge_is_documented():
    for name in CHALLENGES:
        assert CHALLENGES[name].strip(), name


# --- each pathology actually lands in the data ---


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("ch")
    out = {}
    for ch in list(CHALLENGES) + ["clean"]:
        challenges = frozenset() if ch == "clean" else frozenset({ch})
        out[ch] = make_competition(
            CompetitionSpec(f"c-{ch}", "binary", n_train=900, n_test=400,
                            difficulty=0.4, n_teams=80, seed=3,
                            challenges=challenges), root)
    return out


def test_leakage_adds_a_column_that_predicts_train_and_not_test(built):
    train = built["leakage"].joinpath("prepared/public/train.csv").read_text()
    assert "meta_score" in train.split("\n")[0]
    rows = [r.split(",") for r in train.strip().split("\n")[1:]]
    leak = np.array([float(r[-2]) for r in rows])
    y = np.array([float(r[-1]) for r in rows])
    corr = np.corrcoef(leak, y)[0, 1]
    assert abs(corr) > 0.95, "the leak must be near-perfect in train"


def test_missing_writes_empty_cells(built):
    assert ",," in built["missing"].joinpath("prepared/public/train.csv").read_text()
    assert ",," not in built["clean"].joinpath("prepared/public/train.csv").read_text()


def test_outliers_creates_extreme_values(built):
    def max_abs(comp):
        rows = comp.joinpath("prepared/public/train.csv").read_text().strip().split("\n")[1:]
        return max(abs(float(v)) for r in rows for v in r.split(",")[1:-1])

    assert max_abs(built["outliers"]) > 5 * max_abs(built["clean"])


def test_shift_moves_the_test_distribution(built):
    def mean_of(comp, fname):
        rows = comp.joinpath(f"prepared/public/{fname}").read_text().strip().split("\n")[1:]
        cols = [r.split(",")[1:] for r in rows]
        return float(np.mean([float(c[0]) for c in cols]))

    shifted = abs(mean_of(built["shift"], "test.csv") - mean_of(built["shift"], "train.csv"))
    clean = abs(mean_of(built["clean"], "test.csv") - mean_of(built["clean"], "train.csv"))
    assert shifted > clean + 0.3


def test_challenges_are_recorded_in_the_spec(built):
    spec = json.loads(built["leakage"].joinpath("competition.json").read_text())
    assert spec["challenges"] == ["leakage"]
    assert json.loads(built["clean"].joinpath("competition.json").read_text())[
        "challenges"] == []


def test_the_underlying_problem_is_unchanged(built):
    """Pathologies are applied to features after the target is fixed, so the
    oracle -- the best achievable score -- must not move."""
    oracles = {
        k: json.loads(v.joinpath("competition.json").read_text())["oracle_score"]
        for k, v in built.items()
    }
    assert len(set(round(v, 9) for v in oracles.values())) == 1


# --- the competences themselves ---


def test_imputation_uses_train_means_for_test():
    tr = np.array([[1.0, 2.0], [3.0, np.nan]])
    te = np.array([[np.nan, np.nan]])
    tr_i, means = impute(tr)
    te_i, _ = impute(te, means)
    assert not np.isnan(tr_i).any() and not np.isnan(te_i).any()
    assert te_i[0, 0] == pytest.approx(2.0)


def test_clipping_only_touches_columns_with_outliers():
    """Unconditional clipping discards signal on clean data and is actively
    harmful under covariate shift."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    _, bounds = clip_to(X)
    assert np.isinf(bounds[0]).all(), "clean columns must not be clipped"
    X[0, 1] = 400.0
    _, bounds2 = clip_to(X)
    assert np.isfinite(bounds2[0][1]), "the outlying column must be clipped"
    assert np.isinf(bounds2[0][0]), "its neighbours must not be"


def test_outlier_detection_is_not_masked_by_the_outlier():
    """Mean/std detection hides extreme values: they inflate the standard
    deviation, which shrinks their own z-score. MAD is unmoved by them."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 1))
    X[:5, 0] = 500.0
    _, bounds = clip_to(X)
    assert np.isfinite(bounds[0][0]), "five gross outliers must still be caught"


def test_suspicious_features_finds_a_leak_and_not_a_real_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400).astype(float)
    real = y + rng.normal(0, 1.2, 400)
    leak = y + rng.normal(0, 0.02, 400)
    X = np.column_stack([real, leak, rng.normal(size=400)])
    assert suspicious_features(X, y) == [1]


def test_suspicious_features_ignores_constant_columns():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    X = np.ones((4, 1))
    assert suspicious_features(X, y) == []


# --- profile aggregation ---


def _profile():
    return SkillProfile([
        SkillCell("naive", "leakage", 0.9, 0.4),
        SkillCell("naive", "missing", 0.9, None, "non-finite prediction(s)"),
        SkillCell("expert", "leakage", 0.9, 0.95),
        SkillCell("expert", "missing", 0.9, 0.89),
    ])


def test_delta_is_challenged_minus_control():
    assert _profile().cell("naive", "leakage").delta == pytest.approx(-0.5)


def test_a_broken_run_has_no_delta_but_is_flagged():
    c = _profile().cell("naive", "missing")
    assert c.delta is None and c.broke


def test_robustness_counts_a_break_as_a_total_loss():
    p = _profile()
    assert p.robustness("naive") == pytest.approx(-0.75)
    assert p.robustness("expert") > p.robustness("naive")


def test_robustness_ignores_gains():
    """Doing better than the control is not evidence of robustness."""
    p = SkillProfile([SkillCell("a", "x", 0.5, 0.9), SkillCell("a", "y", 0.5, 0.5)])
    assert p.robustness("a") == 0.0


def test_hardest_for_names_the_weakest_skill():
    assert _profile().hardest_for("naive") == "missing"
    assert _profile().hardest_for("expert") == "missing"


def test_no_agent_dominates_detects_a_split_field():
    assert _profile().no_agent_dominates() is False
    split = SkillProfile([
        SkillCell("a", "x", 0.5, 0.9), SkillCell("b", "x", 0.5, 0.5),
        SkillCell("a", "y", 0.5, 0.2), SkillCell("b", "y", 0.5, 0.6),
    ])
    assert split.no_agent_dominates()


# --- end to end ---


@posix_only
def test_skills_command_produces_a_discriminating_profile(tmp_path, capsys):
    from mlea.cli import main
    from mlea.skills import load

    rc = main(["skills", "--out", str(tmp_path / "s"), "--n-train", "1200",
               "--n-test", "500", "--time-cap", "300"])
    out = capsys.readouterr().out
    assert rc == 0, out
    p = load(tmp_path / "s" / "skills.json")
    assert set(p.challenges) == set(CHALLENGES)

    naive_missing = p.cell("naive", "missing")
    assert naive_missing.broke, "an agent that ignores NaN cannot be graded"
    assert p.cell("expert", "missing").delta is not None

    leak_naive = p.cell("naive", "leakage").delta
    leak_expert = p.cell("expert", "leakage").delta
    assert leak_expert > leak_naive + 0.2, "leak detection must be worth something"
    assert p.robustness("expert") > p.robustness("naive")
