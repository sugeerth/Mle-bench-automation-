import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mlea import reference
from mlea.bench import DISCRIMINATING_SUITE, CompetitionSpec, make_competition
from mlea.instrument import (
    NotEnoughData,
    null_reliability,
    analyse,
    cost_frontier,
    cronbach_alpha,
    cross_validated_fidelity,
    kendall_tau_b,
    ranking_fidelity,
    select_items,
    split_half_reliability,
)


# --- reliability ----------------------------------------------------------


def test_alpha_of_pure_noise_is_centred_on_zero_but_wide():
    """Unbiased in the mean, and wide enough at small n to mislead on one draw.

    This is why :func:`null_reliability` exists. A single matrix of 40 unrelated
    items reached alpha 0.29 here; reported alone that reads as a real if modest
    common factor.
    """
    vals = [cronbach_alpha(np.random.default_rng(s).normal(size=(40, 30)))
            for s in range(40)]
    assert abs(np.mean(vals)) < 0.1
    assert max(vals) > 0.2


def test_alpha_is_high_when_one_ability_drives_every_item():
    rng = np.random.default_rng(1)
    ability = rng.normal(size=(40, 1))
    M = ability + 0.3 * rng.normal(size=(40, 30))
    assert cronbach_alpha(M) > 0.95


def test_alpha_rises_with_more_items_of_the_same_quality():
    """Spearman-Brown, observed rather than asserted."""
    rng = np.random.default_rng(2)
    ability = rng.normal(size=(60, 1))
    noise = rng.normal(size=(60, 40))
    short = cronbach_alpha(ability + 1.5 * noise[:, :8])
    long = cronbach_alpha(ability + 1.5 * noise)
    assert long > short


def test_alpha_rejects_degenerate_shapes():
    with pytest.raises(NotEnoughData):
        cronbach_alpha(np.zeros((5, 1)))
    with pytest.raises(NotEnoughData):
        cronbach_alpha(np.zeros((1, 5)))


def test_null_band_brackets_the_noise_alpha_it_is_there_to_catch():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(40, 30))
    band = null_reliability(M, n_draws=200, rng=np.random.default_rng(1))
    assert band > 0.15
    ability = rng.normal(size=(40, 1))
    real = ability + 0.4 * rng.normal(size=(40, 30))
    assert cronbach_alpha(real) > null_reliability(
        real, n_draws=200, rng=np.random.default_rng(1))


def test_analyse_warns_when_reliability_is_inside_the_null_band():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(30, 12))
    rep = analyse(M, [f"c{i}" for i in range(12)], [f"a{i}" for i in range(30)])
    assert not rep.reliability_is_significant
    assert any("not measuring a common ability" in n for n in rep.notes)


def test_split_half_tracks_alpha():
    rng = np.random.default_rng(3)
    ability = rng.normal(size=(30, 1))
    M = ability + rng.normal(size=(30, 24))
    a = cronbach_alpha(M)
    h = split_half_reliability(M, n_splits=100, rng=np.random.default_rng(4))
    assert abs(a - h) < 0.1


def test_identical_total_variance_gives_zero_reliability():
    """Every subject scoring the same total leaves nothing to be reliable about."""
    M = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    assert cronbach_alpha(M) <= 0.0


# --- item statistics ------------------------------------------------------


def _one_factor(n_subjects=20, n_items=10, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    ability = rng.normal(size=(n_subjects, 1))
    return ability + noise * rng.normal(size=(n_subjects, n_items)), ability.ravel()


def test_dead_item_is_flagged_and_discriminates_zero():
    M, _ = _one_factor()
    # 0.7 exactly, but the column's standard deviation comes back as 2e-16.
    # An equality test against zero would not flag this.
    M[:, 0] = 0.7
    rep = analyse(M, [f"c{i}" for i in range(M.shape[1])],
                  [f"a{i}" for i in range(M.shape[0])])
    dead = rep.items[0]
    assert 0 < dead.sd < 1e-12  # floating point, not an exact constant
    assert dead.dead and dead.discrimination == 0.0
    assert [i.item for i in rep.dead_items] == ["c0"]
    assert dead.saturation == "constant"


def test_saturation_names_the_end_an_item_is_stuck_at():
    M, _ = _one_factor()
    M[:, 0] = 0.0
    M[:, 1] = 1.0
    rep = analyse(M, [f"c{i}" for i in range(M.shape[1])],
                  [f"a{i}" for i in range(M.shape[0])])
    assert rep.items[0].saturation == "floor"
    assert rep.items[1].saturation == "ceiling"


def test_inverted_item_is_flagged_as_a_trap():
    """An item where better agents score worse subtracts from the ranking."""
    M, ability = _one_factor()
    M[:, 0] = -ability  # reversed on purpose
    rep = analyse(M, [f"c{i}" for i in range(M.shape[1])],
                  [f"a{i}" for i in range(M.shape[0])])
    assert rep.items[0].discrimination < 0
    assert rep.items[0].trap
    assert [i.item for i in rep.trap_items] == ["c0"]


def test_discrimination_is_corrected_for_the_item_itself():
    """A single item cannot correlate with a rest-score built without it.

    The uncorrected form -- item against a total that contains it -- would
    report a large positive value here purely from the shared term.
    """
    rng = np.random.default_rng(5)
    M = np.zeros((20, 4))
    M[:, 0] = rng.normal(size=20)  # unrelated to the rest
    M[:, 1:] = rng.normal(size=(20, 1)) + 0.1 * rng.normal(size=(20, 3))
    rep = analyse(M, list("abcd"), [f"s{i}" for i in range(20)])
    assert abs(rep.items[0].discrimination) < 0.5


def test_noise_correction_raises_discrimination_and_is_bounded():
    """A rate over 3 seeds is mostly seed draw; the correction says how much."""
    rng = np.random.default_rng(6)
    ability = rng.uniform(0.2, 0.8, size=(15, 1))
    truth = np.clip(ability + 0.1 * rng.normal(size=(15, 12)), 0.01, 0.99)
    seeds = np.full(truth.shape, 3)
    observed = rng.binomial(seeds, truth) / seeds
    rep = analyse(observed, [f"c{i}" for i in range(12)],
                  [f"a{i}" for i in range(15)], trials=seeds)
    for item in rep.items:
        assert 0.0 <= item.noise_share <= 1.0
        assert -1.0 <= item.discrimination_disattenuated <= 1.0
    mean_raw = np.mean([i.discrimination for i in rep.items])
    mean_adj = np.mean([i.discrimination_disattenuated for i in rep.items])
    assert mean_adj >= mean_raw


def test_noise_share_is_absent_without_trial_counts():
    M, _ = _one_factor()
    rep = analyse(M, [f"c{i}" for i in range(10)], [f"a{i}" for i in range(20)])
    assert all(i.noise_share is None for i in rep.items)
    assert all(i.discrimination_disattenuated is None for i in rep.items)


def test_analyse_rejects_a_mismatched_shape_and_nonfinite_scores():
    M, _ = _one_factor()
    with pytest.raises(ValueError):
        analyse(M, ["only-one"], [f"a{i}" for i in range(20)])
    bad = M.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        analyse(bad, [f"c{i}" for i in range(10)], [f"a{i}" for i in range(20)])


# --- separation, strata, prophecy -----------------------------------------


def test_strata_grow_with_reliability():
    low, _ = _one_factor(noise=3.0, seed=7)
    high, _ = _one_factor(noise=0.2, seed=7)
    names = [f"c{i}" for i in range(10)], [f"a{i}" for i in range(20)]
    a = analyse(low, *names)
    b = analyse(high, *names)
    assert b.reliability > a.reliability
    assert b.strata > a.strata


def test_items_needed_inverts_spearman_brown():
    M, _ = _one_factor(noise=1.5, seed=8)
    rep = analyse(M, [f"c{i}" for i in range(10)], [f"a{i}" for i in range(20)])
    need = rep.items_needed(0.95)
    assert need > rep.n_items
    # The prophecy applied forward to that length must reach the target.
    m = need / rep.n_items
    r = rep.reliability
    assert m * r / (1 + (m - 1) * r) >= 0.95 - 1e-9


def test_items_needed_returns_current_length_when_already_met():
    M, _ = _one_factor(noise=0.2, seed=9)
    rep = analyse(M, [f"c{i}" for i in range(10)], [f"a{i}" for i in range(20)])
    assert rep.items_needed(0.5) == rep.n_items


def test_items_needed_is_none_when_items_share_no_signal():
    rng = np.random.default_rng(10)
    M = rng.normal(size=(30, 12))
    M[:, 0] = -M[:, 1:].sum(axis=1)  # force the common signal negative
    rep = analyse(M, [f"c{i}" for i in range(12)], [f"a{i}" for i in range(30)])
    if rep.reliability <= 0:
        assert rep.items_needed(0.9) is None


def test_items_needed_validates_its_target():
    M, _ = _one_factor()
    rep = analyse(M, [f"c{i}" for i in range(10)], [f"a{i}" for i in range(20)])
    with pytest.raises(ValueError):
        rep.items_needed(1.0)


# --- rank agreement -------------------------------------------------------


def test_tau_b_is_one_for_identical_orders_and_minus_one_for_reversed():
    a = [1.0, 2.0, 3.0, 4.0]
    assert kendall_tau_b(a, a) == pytest.approx(1.0)
    assert kendall_tau_b(a, a[::-1]) == pytest.approx(-1.0)


def test_tau_b_discounts_ties_rather_than_guessing():
    """Tau-a would charge a tie as a disagreement; tau-b divides it out."""
    tied = [1.0, 1.0, 2.0, 3.0]
    other = [1.0, 2.0, 3.0, 4.0]
    tau = kendall_tau_b(tied, other)
    assert 0.8 < tau < 1.0


def test_tau_b_is_nan_when_one_side_has_no_order():
    assert np.isnan(kendall_tau_b([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


# --- subset selection -----------------------------------------------------


def test_selection_prefers_informative_items_over_dead_ones():
    M, _ = _one_factor(n_items=12, seed=11)
    M[:, :4] = 0.5  # four competitions nobody's score varies on
    chosen = select_items(M, 4)
    assert all(j >= 4 for j in chosen)


def test_selection_respects_a_budget():
    M, _ = _one_factor(n_items=10, seed=12)
    costs = [1.0] * 5 + [50.0] * 5
    chosen = select_items(M, 10, cost=costs, budget=6.0)
    assert sum(costs[j] for j in chosen) <= 6.0
    assert chosen  # and it still picks something


def test_selection_validates_k_and_costs():
    M, _ = _one_factor(n_items=6)
    with pytest.raises(ValueError):
        select_items(M, 0)
    with pytest.raises(ValueError):
        select_items(M, 3, cost=[1.0] * 5)
    with pytest.raises(ValueError):
        select_items(M, 3, cost=[0.0] * 6)


def test_full_subset_has_perfect_fidelity_with_itself():
    M, _ = _one_factor(n_items=8, seed=13)
    assert ranking_fidelity(M, range(8)) == pytest.approx(1.0)


def test_cross_validation_does_not_credit_a_selector_for_its_own_fit():
    """Selecting on every subject then scoring the same ones proves nothing.

    A selector handed a *reversed* ability signal scores well in-sample only
    because the rest-score is reversed too; held out on subjects it never saw,
    a selector that picks noise columns must not reach the fidelity of one that
    picks the informative ones.
    """
    rng = np.random.default_rng(14)
    ability = rng.normal(size=(20, 1))
    signal = ability + 0.2 * rng.normal(size=(20, 6))
    noise = rng.normal(size=(20, 6))
    M = np.hstack([signal, noise])
    good = cross_validated_fidelity(M, lambda tr: list(range(6)), n_splits=20,
                                    rng=np.random.default_rng(0))
    bad = cross_validated_fidelity(M, lambda tr: list(range(6, 12)), n_splits=20,
                                   rng=np.random.default_rng(0))
    assert good.mean > bad.mean


def test_cross_validation_needs_enough_subjects_to_split():
    M, _ = _one_factor(n_subjects=4, n_items=6)
    with pytest.raises(NotEnoughData):
        cross_validated_fidelity(M, lambda tr: [0, 1, 2], n_splits=5)


def test_cost_frontier_is_monotone_in_item_count():
    M, _ = _one_factor(n_subjects=20, n_items=12, seed=15)
    costs = [1.0] * 12
    rows = cost_frontier(M, costs, [2.0, 6.0, 12.0], n_splits=8,
                         rng=np.random.default_rng(0))
    counts = [n for _, n, _, _ in rows]
    assert counts == sorted(counts)


# --- the real MLE-bench matrix --------------------------------------------


def test_published_matrix_is_rectangular_and_complete():
    rates, seeds, exps, comps = reference.matrix()
    assert len(comps) == 75
    assert len(exps) >= 20
    assert all(len(r) == 75 for r in rates)
    assert all(len(s) == 75 for s in seeds)
    assert all(0.0 <= v <= 1.0 for row in rates for v in row)
    assert all(v >= 1 for row in seeds for v in row)


def test_every_competition_has_a_complexity_tier():
    _, _, _, comps = reference.matrix()
    tiers = reference.complexity()
    assert set(comps) <= set(tiers)
    assert set(tiers.values()) == {"low", "medium", "high"}


def test_the_lite_split_is_exactly_the_low_complexity_tier():
    """Two independently sourced lists of the same thing must agree."""
    tiers = reference.complexity()
    low = {c for c, t in tiers.items() if t == "low"}
    assert low == set(reference.LITE_COMPETITIONS)


def test_published_mlebench_has_competitions_no_agent_has_ever_solved():
    """The finding that motivates the module: dead items, paid for in full."""
    rates, seeds, exps, comps = reference.matrix()
    rep = analyse(np.array(rates), comps, exps, trials=np.array(seeds),
                  complexity=reference.complexity())
    assert len(rep.dead_items) >= 10
    assert all(i.saturation == "floor" for i in rep.dead_items)
    assert rep.dead_cost_share() > 0.15


def test_mlebench_reliability_falls_as_the_ability_band_narrows():
    """Reliability belongs to the instrument *and* the population it is used on."""
    rates, seeds, exps, comps = reference.matrix()
    M, S = np.array(rates), np.array(seeds)
    order = np.argsort(-M.mean(axis=1))
    wide = analyse(M, comps, exps, trials=S)
    rows = sorted(order[:8])
    narrow = analyse(M[rows], comps, [exps[i] for i in rows], trials=S[rows])
    assert narrow.reliability < wide.reliability
    assert len(narrow.dead_items) > len(wide.dead_items)


def test_mlebench_saturates_at_the_ceiling_only_among_strong_agents():
    """The erosion is not only "still too hard" -- half of it is "now too easy"."""
    rates, seeds, exps, comps = reference.matrix()
    M, S = np.array(rates), np.array(seeds)
    rows = sorted(np.argsort(-M.mean(axis=1))[:8])
    rep = analyse(M[rows], comps, [exps[i] for i in rows], trials=S[rows],
                  complexity=reference.complexity())
    ceiling = [i for i in rep.dead_items if i.saturation == "ceiling"]
    assert len(ceiling) >= 10


def test_a_random_subset_is_not_worse_than_lite_at_the_same_item_count():
    """Lite is chosen for human hours, not information. This says what that costs."""
    rates, _, exps, comps = reference.matrix()
    M = np.array(rates)
    lite = [j for j, c in enumerate(comps) if c in reference.LITE_COMPETITIONS]
    rng = np.random.default_rng(0)
    est_lite = cross_validated_fidelity(M, lambda tr: lite, n_splits=20,
                                        rng=np.random.default_rng(1))
    est_rand = cross_validated_fidelity(
        M, lambda tr: list(rng.choice(M.shape[1], len(lite), replace=False)),
        n_splits=20, rng=np.random.default_rng(1))
    assert abs(est_lite.mean - est_rand.mean) < 0.1


# --- the generated suite, after the instrument was pointed at it -----------


def test_latent_complexity_opens_headroom_between_the_field_and_the_oracle():
    """The fix that made a generated competition able to separate anything.

    With the original latent function the strongest model in the simulated field
    represents the target exactly, so the whole leaderboard sits at the oracle
    and percentile is noise.
    """
    gaps = []
    for lc in (0.0, 0.8):
        with tempfile.TemporaryDirectory() as tmp:
            spec = CompetitionSpec("hr", "binary", n_train=800, n_test=400,
                                   n_teams=60, difficulty=0.5, seed=1,
                                   latent_complexity=lc)
            comp = make_competition(spec, tmp)
            lb = np.array(json.loads((comp / "leaderboard.json").read_text()))
            oracle = json.loads((comp / "competition.json").read_text())["oracle_score"]
            gaps.append(oracle - float(np.median(lb)))
    assert gaps[1] > 1.4 * gaps[0]


def test_generation_is_unchanged_when_the_new_knobs_are_left_alone():
    """Old specs must still produce the same bytes, or every prior result moves."""
    spec = CompetitionSpec("compat", "binary", n_train=300, n_test=150,
                           n_teams=40, difficulty=0.5, seed=3)
    assert spec.latent_complexity == 0.0 and spec.field_strength == 0.0
    digests = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            comp = make_competition(spec, tmp)
            digests.append((
                (comp / "prepared/public/train.csv").read_bytes(),
                (comp / "leaderboard.json").read_text(),
            ))
    assert digests[0] == digests[1]


def test_the_new_knobs_are_validated():
    with pytest.raises(ValueError):
        CompetitionSpec("x", latent_complexity=1.5)
    with pytest.raises(ValueError):
        CompetitionSpec("x", field_strength=-0.1)


def test_discriminating_suite_leaves_out_the_items_a_constant_wins():
    ids = {s.id for s in DISCRIMINATING_SUITE}
    assert "disc-regression-shift" not in ids
    assert "disc-regression-leakage" not in ids
    assert "disc-binary-leakage" in ids
    assert all(s.latent_complexity > 0 and s.field_strength > 0
               for s in DISCRIMINATING_SUITE)
