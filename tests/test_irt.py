import numpy as np
import pytest

from mlea import reference
from mlea.irt import (
    MIN_IDENTIFIED_A,
    Attenuation,
    IRTModel,
    _sigmoid,
    calibrate_attenuation,
    fit,
    held_out_loglik,
)
import mlea.irt as irt_module


# --- parameter recovery: the test the rest of the module rests on ----------


def _simulate(n_subjects=25, n_items=75, seed=0, seed_counts=(3, 3, 3, 5, 9, 20, 39)):
    """Data generated at the published benchmark's exact shape.

    25 agents is a small number of *people*, which is what the usual
    rule-of-thumb about item-response models is counting. It is not a small
    amount of data: each cell carries 3 to 39 Bernoulli trials, about 9,000 in
    all, and the parameters being estimated are per item.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_subjects)
    a = np.exp(rng.normal(0.1, 0.4, n_items))
    b = rng.normal(0.3, 1.3, n_items)
    trials = rng.choice(seed_counts, size=(n_subjects, n_items)).astype(float)
    p = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    successes = rng.binomial(trials.astype(int), p).astype(float)
    return successes, trials, theta, a, b


def test_ability_is_recovered_almost_exactly_at_the_real_data_shape():
    S, N, theta, _, _ = _simulate()
    m = fit(S, N, [f"i{j}" for j in range(75)], [f"s{i}" for i in range(25)])
    est = np.array(m.ability)
    # Normalised, so compare on the standardised truth.
    truth = (theta - theta.mean()) / theta.std(ddof=1)
    assert np.corrcoef(truth, est)[0, 1] > 0.98
    assert np.sqrt(np.mean((est - truth) ** 2)) < 0.25


def test_difficulty_is_recovered_well_inside_the_measured_ability_range():
    """And badly outside it, which is the honest half of the claim."""
    S, N, theta, _, b = _simulate()
    m = fit(S, N, [f"i{j}" for j in range(75)], [f"s{i}" for i in range(25)])
    b_hat = np.array([i.b for i in m.items])
    scale = theta.std(ddof=1)
    truth = (b - theta.mean()) / scale
    near = np.abs(truth) <= 2.0
    assert np.corrcoef(truth[near], b_hat[near])[0, 1] > 0.90
    inside = np.sqrt(np.mean((b_hat[near] - truth[near]) ** 2))
    outside = np.sqrt(np.mean((b_hat[~near] - truth[~near]) ** 2))
    assert inside < outside  # an item far from every agent is barely located


def test_discrimination_is_attenuated_at_this_sample_size():
    """Documented rather than hidden: information goes as the square of this.

    The estimator is consistent -- the bias shrinks steadily as subjects are
    added -- but at 25 subjects the slopes come back roughly a quarter shallow,
    so every standard error derived from them is too wide, not too narrow.
    """
    S, N, theta, a, _ = _simulate()
    m = fit(S, N, [f"i{j}" for j in range(75)], [f"s{i}" for i in range(25)])
    a_hat = np.array([i.a for i in m.items])
    truth = a * theta.std(ddof=1)  # the normalised scale
    slope = float(np.polyfit(truth, a_hat, 1)[0])
    assert 0.5 < slope < 0.95
    assert np.corrcoef(truth, a_hat)[0, 1] > 0.7


def test_the_attenuation_shrinks_as_subjects_are_added():
    """Consistency, demonstrated rather than cited."""
    slopes = []
    for n in (25, 400):
        S, N, theta, a, _ = _simulate(n_subjects=n, n_items=40, seed=1)
        m = fit(S, N, [f"i{j}" for j in range(40)], [f"s{i}" for i in range(n)])
        a_hat = np.array([i.a for i in m.items])
        slopes.append(float(np.polyfit(a * theta.std(ddof=1), a_hat, 1)[0]))
    assert slopes[1] > slopes[0]
    assert slopes[1] > 0.85


# --- the quadrature grid --------------------------------------------------


def test_a_too_narrow_quadrature_grid_clips_the_strongest_agents():
    """The published agents ran off the end of a +-4 grid, and the fit was worse.

    Worth a test because the symptom is not an error -- it is a silently capped
    ability scale and a log-likelihood that looks fine in isolation.
    """
    rates, seeds, exps, comps = reference.matrix()
    S = np.rint(np.array(rates) * np.array(seeds))
    N = np.array(seeds, dtype=float)
    saved = irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE
    try:
        irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE = 61, 4.0
        narrow = fit(S, N, comps, exps, normalize=False)
        irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE = 241, 6.0
        wide = fit(S, N, comps, exps, normalize=False)
    finally:
        irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE = saved
    assert wide.log_likelihood > narrow.log_likelihood + 1.0
    assert max(wide.ability) > max(narrow.ability)


def test_the_fit_is_stable_once_the_grid_is_wide_enough():
    S, N, _, _, _ = _simulate(n_subjects=12, n_items=20, seed=2)
    names = [f"i{j}" for j in range(20)], [f"s{i}" for i in range(12)]
    saved = irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE
    try:
        irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE = 121, 6.0
        coarse = fit(S, N, *names, normalize=False)
        irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE = 481, 6.0
        fine = fit(S, N, *names, normalize=False)
    finally:
        irt_module.N_QUADRATURE, irt_module.QUADRATURE_RANGE = saved
    assert abs(coarse.log_likelihood - fine.log_likelihood) < 0.05


# --- the scale ------------------------------------------------------------


def test_normalising_the_scale_leaves_every_prediction_identical():
    """A 2PL is identified only up to a linear rescaling of ability.

    So normalising must change the numbers printed and nothing the model
    predicts. If it changes a probability, the transform is wrong.
    """
    S, N, _, _, _ = _simulate(n_subjects=15, n_items=25, seed=3)
    names = [f"i{j}" for j in range(25)], [f"s{i}" for i in range(15)]
    raw = fit(S, N, *names, normalize=False)
    norm = fit(S, N, *names, normalize=True)
    assert raw.log_likelihood == pytest.approx(norm.log_likelihood, abs=1e-9)
    mu = float(np.mean(raw.ability))
    sigma = float(np.std(raw.ability, ddof=1))
    for t in (-1.5, 0.0, 0.8, 2.0):
        raw_theta = t * sigma + mu
        for a, b in zip(raw.items, norm.items):
            p_raw = float(_sigmoid(np.array([a.a * (raw_theta - a.b)]))[0])
            p_norm = float(_sigmoid(np.array([b.a * (t - b.b)]))[0])
            assert p_raw == pytest.approx(p_norm, abs=1e-8)


def test_item_classification_does_not_depend_on_the_scale():
    """Identical data must not classify differently because of a rescaling.

    Testing ``|a|`` against a fixed threshold did exactly that: on the published
    data the same competition came back "too flat to place" unnormalised and
    "placeable but extrapolated" normalised, because ability SD was 2.5 and every
    slope moved with it. The threshold is now on ``|a| * SD(ability)``.
    """
    rates, seeds, exps, comps = reference.matrix()
    M, N = np.array(rates), np.array(seeds, dtype=float)
    raw = fit(np.rint(M * N), N, comps, exps, normalize=False)
    norm = fit(np.rint(M * N), N, comps, exps, normalize=True)
    for attr in ("located_items", "unidentified_items", "extrapolated_items"):
        assert [i.item for i in getattr(raw, attr)] == \
               [i.item for i in getattr(norm, attr)]


def test_normalised_ability_has_mean_zero_and_unit_spread():
    S, N, _, _, _ = _simulate(n_subjects=15, n_items=25, seed=4)
    m = fit(S, N, [f"i{j}" for j in range(25)], [f"s{i}" for i in range(15)])
    assert np.mean(m.ability) == pytest.approx(0.0, abs=1e-9)
    assert np.std(m.ability, ddof=1) == pytest.approx(1.0, abs=1e-9)


# --- information ----------------------------------------------------------


def test_item_information_peaks_at_its_own_difficulty():
    S, N, _, _, _ = _simulate(n_subjects=12, n_items=20, seed=5)
    m = fit(S, N, [f"i{j}" for j in range(20)], [f"s{i}" for i in range(12)])
    item = next(i for i in m.items if i.identified and abs(i.b) < 1.5)
    grid = np.linspace(item.b - 3, item.b + 3, 601)
    info = item.information(grid)
    assert grid[int(np.argmax(info))] == pytest.approx(item.b, abs=0.02)
    assert float(np.max(info)) == pytest.approx(item.peak_information, rel=1e-3)


def test_information_is_additive_in_seeds():
    """Three seeds of one competition triple information at that difficulty only."""
    S, N, _, _, _ = _simulate(n_subjects=12, n_items=20, seed=6)
    m = fit(S, N, [f"i{j}" for j in range(20)], [f"s{i}" for i in range(12)])
    grid = np.linspace(-2, 2, 9)
    one = m.information(grid, seeds=1.0)
    three = m.information(grid, seeds=3.0)
    assert np.allclose(three, 3.0 * one)
    assert np.allclose(m.standard_error(grid, seeds=4.0),
                       m.standard_error(grid, seeds=1.0) / 2.0)


def test_standard_error_is_lowest_where_the_items_are():
    S, N, _, _, b = _simulate(n_subjects=20, n_items=40, seed=7)
    m = fit(S, N, [f"i{j}" for j in range(40)], [f"s{i}" for i in range(20)])
    grid = np.linspace(-6, 6, 241)
    se = m.standard_error(grid)
    best = grid[int(np.nanargmin(se))]
    assert abs(best) < 2.0            # near the mass of the items
    assert se[0] > np.nanmin(se) * 2  # and much worse out at the edge


def test_resolving_range_tightens_as_the_bar_is_raised():
    S, N, _, _, _ = _simulate(n_subjects=20, n_items=40, seed=8)
    m = fit(S, N, [f"i{j}" for j in range(40)], [f"s{i}" for i in range(20)])
    loose = m.resolving_range(0.5)
    tight = m.resolving_range(0.2)
    assert loose is not None and tight is not None
    assert loose[0] <= tight[0] and tight[1] <= loose[1]
    assert m.blind_above(0.2) == tight[1]


def test_an_impossible_bar_resolves_nothing():
    S, N, _, _, _ = _simulate(n_subjects=12, n_items=20, seed=9)
    m = fit(S, N, [f"i{j}" for j in range(20)], [f"s{i}" for i in range(12)])
    assert m.resolving_range(1e-6) is None
    assert m.blind_above(1e-6) is None


def test_expected_rate_is_monotone_in_ability():
    S, N, _, _, _ = _simulate(n_subjects=15, n_items=30, seed=10)
    m = fit(S, N, [f"i{j}" for j in range(30)], [f"s{i}" for i in range(15)])
    rates = m.expected_rate(np.linspace(-4, 4, 60))
    assert np.all(np.diff(rates) >= -1e-12)


def test_corrected_information_needs_a_calibration_first():
    S, N, _, _, _ = _simulate(n_subjects=12, n_items=20, seed=11)
    m = fit(S, N, [f"i{j}" for j in range(20)], [f"s{i}" for i in range(12)])
    with pytest.raises(ValueError, match="attenuation"):
        m.information(0.0, corrected=True)


def test_correcting_for_attenuation_narrows_the_error_bars():
    import dataclasses

    S, N, _, _, _ = _simulate(n_subjects=12, n_items=20, seed=12)
    m = fit(S, N, [f"i{j}" for j in range(20)], [f"s{i}" for i in range(12)])
    m = dataclasses.replace(m, attenuation=0.75)
    raw = float(m.standard_error(0.0)[0])
    fixed = float(m.standard_error(0.0, corrected=True)[0])
    assert fixed < raw
    assert fixed == pytest.approx(raw * 0.75)


# --- what the model says about items it cannot place -----------------------


def test_a_competition_nobody_solves_is_flagged_rather_than_reported():
    """Its difficulty is where the prior stops, not a measurement.

    This is the distinction classical test theory cannot draw: a competition
    nobody solves and a competition that tells you nothing both have zero
    variance, and only one of them is interesting.
    """
    S, N, _, _, _ = _simulate(n_subjects=20, n_items=30, seed=13)
    S[:, 0] = 0.0          # never solved
    S[:, 1] = N[:, 1]      # always solved
    m = fit(S, N, [f"i{j}" for j in range(30)], [f"s{i}" for i in range(20)])
    assert not m.items[0].identified
    assert not m.items[1].identified
    assert m.items[0].b > max(i.b for i in m.items if i.identified)
    assert m.items[1].b < min(i.b for i in m.items if i.identified)
    assert {i.item for i in m.unidentified_items} == {"i0", "i1"}


def test_an_inverted_competition_gets_a_negative_slope():
    """Discrimination is left free in sign, so a trap item shows up as one."""
    S, N, theta, _, _ = _simulate(n_subjects=25, n_items=30, seed=14)
    strong = np.argsort(-theta)
    # Reverse the outcome: the best agents fail, the worst succeed.
    S[strong, 0] = N[strong, 0] * np.linspace(0.0, 1.0, len(strong))
    S[:, 0] = np.rint(S[:, 0])
    m = fit(S, N, [f"i{j}" for j in range(30)], [f"s{i}" for i in range(25)])
    assert m.items[0].a < 0
    assert "i0" in {i.item for i in m.inverted_items}


# --- validation -----------------------------------------------------------


def test_fit_validates_its_inputs():
    S, N, _, _, _ = _simulate(n_subjects=6, n_items=8, seed=15)
    names = [f"i{j}" for j in range(8)], [f"s{i}" for i in range(6)]
    with pytest.raises(ValueError):
        fit(S, N[:, :4], *names)
    with pytest.raises(ValueError):
        fit(S, N, ["too-few"], names[1])
    with pytest.raises(ValueError):
        fit(S, np.zeros_like(N), *names)
    with pytest.raises(ValueError):
        fit(N + 1.0, N, *names)          # successes above trials
    with pytest.raises(ValueError):
        fit(-S, N, *names)
    with pytest.raises(ValueError):
        fit(S[:1], N[:1], names[0], ["only-one"])


def test_item_lookup_by_name():
    S, N, _, _, _ = _simulate(n_subjects=6, n_items=8, seed=16)
    m = fit(S, N, [f"i{j}" for j in range(8)], [f"s{i}" for i in range(6)])
    assert m.item("i3").item == "i3"
    with pytest.raises(KeyError):
        m.item("nope")


def test_select_for_ability_targets_the_band_it_is_given():
    S, N, _, _, _ = _simulate(n_subjects=20, n_items=40, seed=17)
    m = fit(S, N, [f"i{j}" for j in range(40)], [f"s{i}" for i in range(20)])
    easy = m.select_for_ability(8, -1.5)
    hard = m.select_for_ability(8, 1.5)
    assert len(easy) == len(hard) == 8
    assert set(easy) != set(hard)
    # The set chosen for an ability must carry more information there.
    def total(names, theta):
        return sum(float(m.item(n).information(theta)) for n in names)
    assert total(easy, -1.5) > total(hard, -1.5)
    assert total(hard, 1.5) > total(easy, 1.5)
    with pytest.raises(ValueError):
        m.select_for_ability(0, 0.0)


def test_selection_can_exclude_named_items():
    S, N, _, _, _ = _simulate(n_subjects=12, n_items=20, seed=18)
    m = fit(S, N, [f"i{j}" for j in range(20)], [f"s{i}" for i in range(12)])
    first = m.select_for_ability(3, 0.0)
    again = m.select_for_ability(3, 0.0, exclude=first)
    assert not set(first) & set(again)


# --- does the second parameter earn its keep? -----------------------------


def test_the_two_parameter_model_beats_the_item_mean_out_of_sample():
    """An in-sample improvement from more parameters is not evidence of anything.

    The item-mean baseline is what classical test theory implicitly assumes --
    every agent is the average agent -- so a model that cannot beat it on cells
    it never saw has bought nothing.
    """
    S, N, _, _, _ = _simulate(n_subjects=20, n_items=30, seed=19)
    out = held_out_loglik(S, N, [f"i{j}" for j in range(30)],
                          [f"s{i}" for i in range(20)], n_folds=4,
                          rng=np.random.default_rng(0))
    assert out["2pl"] > out["item_mean"]
    assert out["rasch"] > out["item_mean"]
    assert set(out) == {"2pl", "rasch", "item_mean"}
    assert all(v < 0 for v in out.values())


def test_attenuation_calibration_reports_a_shallow_slope_and_good_ability():
    """The parametric bootstrap asks a question answerable without the truth.

    Not "are these parameters right" but "if they were, would this design
    recover them". The answer is the bias correction, and it also says which
    outputs to lean on.
    """
    S, N, _, _, _ = _simulate(n_subjects=20, n_items=30, seed=20)
    m = fit(S, N, [f"i{j}" for j in range(30)], [f"s{i}" for i in range(20)])
    att = calibrate_attenuation(m, N, n_draws=4, rng=np.random.default_rng(0))
    assert isinstance(att, Attenuation)
    assert 0.4 < att.slope < 1.1
    assert att.ability_recovery > 0.95
    assert att.n_draws == 4
    assert "discrimination slope" in str(att)


def test_calibration_refuses_a_model_with_nothing_identified():
    S, N, _, _, _ = _simulate(n_subjects=8, n_items=4, seed=21)
    S[:] = 0.0
    m = fit(S, N, [f"i{j}" for j in range(4)], [f"s{i}" for i in range(8)])
    with pytest.raises(ValueError, match="identified"):
        calibrate_attenuation(m, N, n_draws=2)


# --- the published benchmark ---------------------------------------------


def test_published_mlebench_fit_orders_agents_the_way_their_rates_do():
    rates, seeds, exps, comps = reference.matrix()
    M, N = np.array(rates), np.array(seeds, dtype=float)
    m = fit(np.rint(M * N), N, comps, exps, complexity=reference.complexity())
    assert m.converged
    ability = np.array(m.ability)
    # Ability and raw medal rate must agree on the order; they are different
    # quantities, so the agreement is a check rather than a tautology.
    assert np.corrcoef(ability, M.mean(axis=1))[0, 1] > 0.95
    # The 11 never-medalled competitions cannot be located on the scale.
    assert len(m.unidentified_items) == 11
    assert all(i.observed_rate == 0.0 for i in m.unidentified_items)


def test_published_mlebench_is_sharpest_in_its_middle_and_blurs_at_the_top():
    """The thing a single reliability number cannot say."""
    rates, seeds, exps, comps = reference.matrix()
    M, N = np.array(rates), np.array(seeds, dtype=float)
    m = fit(np.rint(M * N), N, comps, exps)
    mid = float(m.standard_error(0.3, seeds=3.0)[0])
    top = float(m.standard_error(2.2, seeds=3.0)[0])
    assert top > mid
    assert m.marginal_reliability() > 0.9


def test_never_solved_competitions_sit_above_every_measured_agent():
    rates, seeds, exps, comps = reference.matrix()
    M, N = np.array(rates), np.array(seeds, dtype=float)
    m = fit(np.rint(M * N), N, comps, exps)
    hardest = max(i.b for i in m.located_items)
    for item in m.unidentified_items:
        assert item.b > hardest
        assert item.b > max(m.ability)
    # All eleven land on the same value, because none of them is a measurement:
    # with no agent ever solving them, the difficulty is wherever the prior
    # stops. Reporting eleven identical numbers as eleven findings would be the
    # error this flag exists to prevent.
    bs = [i.b for i in m.unidentified_items]
    assert max(bs) - min(bs) < 0.02


def test_a_shallow_slope_is_separated_from_a_hard_competition():
    """``identified`` is not enough; a flat item can be pushed anywhere.

    On the published data ``tensorflow-speech-recognition-challenge`` fits at
    difficulty +10.4 with a slope of 0.33 -- the model saying "this barely tracks
    ability" in a form that reads as "this is impossibly hard". The flag keeps
    those two statements apart.
    """
    rates, seeds, exps, comps = reference.matrix()
    M, N = np.array(rates), np.array(seeds, dtype=float)
    m = fit(np.rint(M * N), N, comps, exps)
    lo, hi = m.ability_range
    for item in m.located_items:
        assert lo - 1.0 <= item.b <= hi + 1.0
    assert len(m.extrapolated_items) >= 1
    for item in m.extrapolated_items:
        assert abs(item.a) < 1.0  # extrapolated difficulty comes with a flat slope
    assert len(m.located_items) + len(m.unidentified_items) \
        + len(m.extrapolated_items) == len(m.items)
