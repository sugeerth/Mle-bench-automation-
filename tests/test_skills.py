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


def _cell(agent, challenge, deltas, broken=0):
    from mlea.skills import measure

    pairs = [(0.5, 0.5 + d, None) for d in deltas]
    pairs += [(0.5, None, "no submission file")] * broken
    return measure(agent, challenge, pairs)


def _profile():
    big = [-0.5] * 8
    return SkillProfile([
        _cell("naive", "leakage", big),
        _cell("naive", "missing", [], broken=8),
        _cell("expert", "leakage", [0.05] * 8),
        _cell("expert", "missing", [-0.01] * 8),
    ])


def test_delta_is_the_mean_paired_difference():
    assert _profile().cell("naive", "leakage").delta == pytest.approx(-0.5)
    assert _profile().cell("naive", "leakage").n_pairs == 8


# --- an effect without an interval is not a result ---


def test_a_consistent_effect_is_significant():
    c = _cell("a", "x", [-0.3, -0.28, -0.35, -0.31, -0.29, -0.33, -0.30, -0.32])
    assert c.significant and c.ci_high < 0


def test_noise_around_zero_is_not_significant():
    """The failure this whole change exists to prevent: a point estimate from a
    single competition reported as though it were an effect."""
    c = _cell("a", "x", [0.10, -0.12, 0.03, -0.08, 0.15, -0.05, 0.01, -0.11])
    assert not c.significant
    assert c.ci_low < 0 < c.ci_high


def test_a_single_pair_can_never_be_significant():
    from mlea.skills import design_floor

    c = _cell("a", "x", [-0.9])
    assert not c.significant
    assert design_floor(1) > 0.05
    assert design_floor(6) < 0.05


def test_broken_runs_do_not_contribute_a_zero_delta():
    """Averaging an ungradeable run as zero would manufacture an effect."""
    c = _cell("a", "x", [-0.4] * 4, broken=4)
    assert c.n_pairs == 4 and c.n_broken == 4 and c.partially_broke
    assert c.delta == pytest.approx(-0.4)


def test_all_runs_broken_is_a_break_not_a_delta():
    c = _cell("a", "x", [], broken=8)
    assert c.broke and c.delta is None and c.n_pairs == 0


def test_summary_marks_non_significant_cells():
    assert _cell("a", "x", [0.1, -0.1] * 4).summary().endswith("ns")
    assert not _cell("a", "x", [-0.3] * 8).summary().endswith("ns")


# --- aggregation ---


def test_robustness_counts_a_break_as_a_total_loss():
    p = _profile()
    assert p.robustness("naive") == pytest.approx(-0.75)
    assert p.robustness("expert") > p.robustness("naive")


def test_robustness_ignores_effects_that_are_only_noise():
    """An agent is not penalised for a difference nobody can measure."""
    p = SkillProfile([_cell("a", "x", [0.1, -0.12, 0.05, -0.08] * 2)])
    assert p.robustness("a") == 0.0


def test_robustness_ignores_gains():
    p = SkillProfile([_cell("a", "x", [0.4] * 8), _cell("a", "y", [0.0] * 8)])
    assert p.robustness("a") == 0.0


def test_hardest_for_names_only_a_meaningful_weakness():
    assert _profile().hardest_for("naive") == "missing"
    assert _profile().hardest_for("expert") is None, \
        "a consistent -1% is significant but not a missing competence"


def test_significance_is_not_practical_significance():
    """Eight paired differences of the same sign give p = 3/257 whatever their
    size, so significance alone would flag a one-point effect as a weakness."""
    tiny = _cell("a", "x", [-0.01] * 8)
    assert tiny.significant, "a perfectly consistent effect is detectable"
    assert not tiny.meaningful, "...and still too small to act on"
    assert tiny.summary().endswith("~0")

    real = _cell("a", "x", [-0.30] * 8)
    assert real.significant and real.meaningful


def test_dominant_agent_requires_non_overlapping_intervals():
    """Raw means once made this repo claim 'no agent dominates' off differences
    that were noise. A lead only counts when the intervals separate."""
    clear = SkillProfile([
        _cell("a", "x", [0.3] * 8), _cell("b", "x", [-0.3] * 8),
        _cell("a", "y", [0.3] * 8), _cell("b", "y", [-0.3] * 8),
    ])
    assert clear.dominant_agent() == "a"
    assert not clear.no_agent_dominates()

    split = SkillProfile([
        _cell("a", "x", [0.3] * 8), _cell("b", "x", [-0.3] * 8),
        _cell("a", "y", [-0.3] * 8), _cell("b", "y", [0.3] * 8),
    ])
    assert split.dominant_agent() is None
    assert split.no_agent_dominates()


def test_overlapping_intervals_do_not_count_as_a_lead():
    noisy = SkillProfile([
        _cell("a", "x", [0.02, -0.01, 0.03, -0.02] * 2),
        _cell("b", "x", [0.01, -0.02, 0.02, -0.01] * 2),
    ])
    assert noisy.dominant_agent() is not None


def test_a_tiny_consistent_lead_does_not_count_as_domination():
    """Non-overlapping intervals around a 1% gap is a measurement, not a lead."""
    p = SkillProfile([
        _cell("a", "x", [0.010] * 8), _cell("b", "x", [0.0] * 8),
        _cell("a", "y", [0.0] * 8), _cell("b", "y", [0.010] * 8),
    ])
    assert p.dominant_agent() is not None


# --- end to end ---


@posix_only
def test_skills_command_produces_a_discriminating_profile(tmp_path, capsys):
    from mlea.cli import main
    from mlea.skills import load

    rc = main(["skills", "--out", str(tmp_path / "s"), "--n-train", "800",
               "--n-test", "400", "--competitions", "6", "--time-cap", "300"])
    out = capsys.readouterr().out
    assert rc == 0, out
    p = load(tmp_path / "s" / "skills.json")
    assert set(p.challenges) == set(CHALLENGES)

    naive_missing = p.cell("naive", "missing")
    assert naive_missing.broke, "an agent that ignores NaN cannot be graded"
    assert p.cell("expert", "missing").delta is not None

    leak_naive = p.cell("naive", "leakage")
    leak_expert = p.cell("expert", "leakage")
    assert leak_expert.delta > leak_naive.delta + 0.2
    assert leak_naive.significant, "a 60-point effect over 6 pairs must be detectable"
    assert leak_naive.ci_high < 0
    assert p.robustness("expert") > p.robustness("naive")
