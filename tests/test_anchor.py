import numpy as np
import pytest

from mlea import reference
from mlea.anchor import (
    SELECTORS,
    EstimationResult,
    _kmeans,
    estimate_ability,
    evaluate,
    naive_rate,
    predict_full_rate,
)
from mlea.irt import _sigmoid, fit


def _world(n_subjects=14, n_items=30, seed=0, n_seeds=5):
    """A 2PL world with difficulty spread wide enough for subsets to be biased."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_subjects)
    a = np.exp(rng.normal(0.1, 0.3, n_items))
    b = np.linspace(-2.2, 2.2, n_items)
    trials = np.full((n_subjects, n_items), float(n_seeds))
    p = _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))
    successes = rng.binomial(trials.astype(int), p).astype(float)
    names = [f"i{j}" for j in range(n_items)], [f"s{i}" for i in range(n_subjects)]
    model = fit(successes, trials, *names)
    return model, successes, trials, names


# --- ability from a partial observation ------------------------------------


def test_ability_from_every_item_matches_the_full_fit():
    model, S, N, _ = _world()
    allj = list(range(S.shape[1]))
    for i in range(S.shape[0]):
        assert estimate_ability(model, S[i], N[i], allj) == pytest.approx(
            model.ability[i], abs=0.12
        )


def test_ability_rises_with_observed_successes():
    model, S, N, _ = _world()
    sub = list(range(0, 30, 3))
    row = N[0].copy()
    low = estimate_ability(model, np.zeros_like(row), row, sub)
    mid = estimate_ability(model, row * 0.5, row, sub)
    high = estimate_ability(model, row, row, sub)
    assert low < mid < high


def test_ability_falls_back_to_the_prior_mean_with_nothing_to_go_on():
    model, S, N, _ = _world()
    assert estimate_ability(model, S[0], N[0], []) == pytest.approx(0.0, abs=1e-6)


def test_unlocatable_items_are_skipped_rather_than_trusted():
    """An item nobody ever solved carries a prior-set difficulty, not a reading."""
    model, S, N, names = _world()
    S2 = S.copy()
    S2[:, 0] = 0.0
    m2 = fit(S2, N, *names)
    assert not m2.items[0].identified
    with_dead = estimate_ability(m2, S2[3], N[3], [0, 5, 10])
    without = estimate_ability(m2, S2[3], N[3], [5, 10])
    assert with_dead == pytest.approx(without, abs=1e-9)


# --- the estimator --------------------------------------------------------


def test_the_subset_mean_is_biased_by_how_easy_the_subset_is():
    """The whole reason to estimate rather than average."""
    model, S, N, _ = _world()
    easy = list(range(0, 8))        # items are ordered by difficulty
    truth = (S / N).mean(axis=1)
    naive_err = np.array([naive_rate(S[i], N[i], easy) - truth[i]
                          for i in range(S.shape[0])])
    pirt_err = np.array([predict_full_rate(model, S[i], N[i], easy) - truth[i]
                         for i in range(S.shape[0])])
    assert naive_err.mean() > 0.15           # the easy subset flatters everyone
    assert abs(pirt_err.mean()) < 0.05       # the model corrects for it
    assert np.abs(pirt_err).mean() < np.abs(naive_err).mean()


def test_the_estimator_also_corrects_a_subset_that_is_too_hard():
    model, S, N, _ = _world()
    hard = list(range(22, 30))
    truth = (S / N).mean(axis=1)
    naive_err = np.mean([naive_rate(S[i], N[i], hard) - truth[i]
                         for i in range(S.shape[0])])
    pirt_err = np.mean([predict_full_rate(model, S[i], N[i], hard) - truth[i]
                        for i in range(S.shape[0])])
    assert naive_err < -0.15
    assert abs(pirt_err) < abs(naive_err)


def test_predicting_from_every_item_reproduces_the_observed_rate():
    """With nothing left to predict, the estimator must not invent anything."""
    model, S, N, _ = _world()
    allj = list(range(S.shape[1]))
    for i in range(S.shape[0]):
        assert predict_full_rate(model, S[i], N[i], allj) == pytest.approx(
            float((S[i] / N[i]).mean()), abs=1e-12
        )


def test_an_unlocatable_item_is_predicted_at_its_observed_rate():
    model, S, N, names = _world()
    S2 = S.copy()
    S2[:, 0] = 0.0
    m2 = fit(S2, N, *names)
    # Item 0 is never solved, so its contribution must be its observed 0.0 and
    # not a probability read off the prior-determined difficulty.
    pred = predict_full_rate(m2, S2[1], N[1], [5, 10, 15])
    by_hand = 0.0
    theta = estimate_ability(m2, S2[1], N[1], [5, 10, 15])
    for j, item in enumerate(m2.items):
        if j in (5, 10, 15):
            by_hand += S2[1, j] / N[1, j]
        elif item.located:
            by_hand += float(_sigmoid(np.array([item.a * (theta - item.b)]))[0])
        else:
            by_hand += item.observed_rate
    assert pred == pytest.approx(by_hand / len(m2.items))


# --- selectors ------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SELECTORS))
def test_every_selector_returns_k_distinct_valid_items(name):
    model, S, N, _ = _world()
    rng = np.random.default_rng(1)
    chosen = SELECTORS[name](model, S, N, 8, rng)
    assert len(chosen) == len(set(chosen)) == 8
    assert all(0 <= j < S.shape[1] for j in chosen)


def test_stratified_selection_spans_the_difficulty_range():
    model, S, N, _ = _world()
    chosen = SELECTORS["stratified"](model, S, N, 6, np.random.default_rng(2))
    rate = (S.sum(axis=0) / N.sum(axis=0))[chosen]
    assert rate.max() - rate.min() > 0.5


def test_max_information_selection_concentrates_rather_than_spreads():
    """Which is exactly why it is a poor choice for estimating a mean."""
    model, S, N, _ = _world()
    info = SELECTORS["max_information"](model, S, N, 6, np.random.default_rng(3))
    strat = SELECTORS["stratified"](model, S, N, 6, np.random.default_rng(3))
    spread = lambda idx: float(np.std([model.items[j].b for j in idx]))
    assert spread(info) < spread(strat)


def test_anchor_selection_skips_items_the_model_cannot_place():
    model, S, N, names = _world()
    S2 = S.copy()
    S2[:, :4] = 0.0
    m2 = fit(S2, N, *names)
    chosen = SELECTORS["anchor"](m2, S2, N, 6, np.random.default_rng(4))
    assert not set(chosen) & {0, 1, 2, 3}


def test_anchor_selection_returns_what_it_can_when_asked_for_too_much():
    """Asked for more items than the model can place, it returns the placeable."""
    model, S, N, names = _world(n_items=10)
    S2 = S.copy()
    S2[:, 6:] = 0.0  # the hardest four become never-solved
    m2 = fit(S2, N, *names)
    placeable = {j for j, i in enumerate(m2.items) if i.located}
    assert 0 < len(placeable) < 10
    chosen = SELECTORS["anchor"](m2, S2, N, 10, np.random.default_rng(5))
    assert set(chosen) <= placeable
    assert len(chosen) == len(placeable)


def test_kmeans_returns_the_requested_number_of_centres():
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(m, 0.05, size=(12, 4)) for m in (-2.0, 0.0, 2.0)])
    C = _kmeans(X, 3, rng)
    assert C.shape == (3, 4)
    assert sorted(round(float(c.mean()), 1) for c in C) == [-2.0, 0.0, 2.0]


def test_kmeans_survives_identical_rows():
    """Duplicate response curves make every distance zero, which divides by it."""
    X = np.ones((10, 3))
    C = _kmeans(X, 4, np.random.default_rng(7))
    assert C.shape == (4, 3)
    assert np.isfinite(C).all()


# --- the head-to-head -----------------------------------------------------


def test_evaluate_holds_out_each_agent_and_scores_every_pairing():
    model, S, N, names = _world(n_subjects=8, n_items=16, seed=8)
    res = evaluate(S, N, names[0], names[1], ks=(6,),
                   selectors=("stratified", "max_information"),
                   n_repeats=1, rng=np.random.default_rng(0))
    assert {(r.selector, r.estimator) for r in res} == {
        ("stratified", "pirt"), ("stratified", "naive"),
        ("max_information", "pirt"), ("max_information", "naive"),
    }
    assert all(r.n_agents == 8 and r.k == 6 for r in res)
    assert all(0.0 <= r.mae <= 1.0 and r.max_error >= r.mae for r in res)
    assert "MAE" in str(res[0])


def test_the_model_estimator_beats_the_subset_mean_out_of_sample():
    """The headline, on held-out agents rather than in sample."""
    model, S, N, names = _world(n_subjects=10, n_items=20, seed=9)
    res = evaluate(S, N, names[0], names[1], ks=(6,),
                   selectors=("max_information",), n_repeats=1,
                   rng=np.random.default_rng(0))
    by = {r.estimator: r for r in res}
    assert by["pirt"].mae < by["naive"].mae
    assert abs(by["pirt"].bias) < abs(by["naive"].bias)


def test_evaluate_validates_its_arguments():
    model, S, N, names = _world(n_subjects=6, n_items=10, seed=10)
    with pytest.raises(ValueError, match="selector"):
        evaluate(S, N, names[0], names[1], selectors=("nope",))
    with pytest.raises(ValueError, match="every k"):
        evaluate(S, N, names[0], names[1], ks=(99,))
    with pytest.raises(ValueError, match="leave-one-out"):
        evaluate(S[:2], N[:2], names[0], names[1][:2], ks=(3,))


def test_result_is_hashable_and_carries_its_identity():
    r = EstimationResult("anchor", "pirt", 20, 0.02, 0.1, -0.001, 25)
    assert {r}
    assert r.selector == "anchor" and r.k == 20


# --- against the published benchmark --------------------------------------


def test_published_benchmark_subset_estimates_land_within_a_few_points():
    """Twenty of the 75, estimated rather than averaged, on held-out agents.

    Kept to one selector and one k so it runs in the test suite; the full
    five-selector sweep is in ``docs/BENCHMARK-QUALITY.md`` with the command to
    reproduce it.
    """
    rates, seeds, exps, comps = reference.matrix()
    M, N = np.array(rates), np.array(seeds, dtype=float)
    S = np.rint(M * N)
    keep = list(range(0, len(exps), 2))  # half the agents, for runtime
    res = evaluate(S[keep], N[keep], comps, [exps[i] for i in keep],
                   ks=(20,), selectors=("stratified",), n_repeats=1,
                   rng=np.random.default_rng(0))
    by = {r.estimator: r for r in res}
    assert by["pirt"].mae < 0.05
    assert by["pirt"].mae < by["naive"].mae
    assert abs(by["pirt"].bias) < abs(by["naive"].bias)
