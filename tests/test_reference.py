import pytest

from mlea.power import DESIGNS, Design, minimum_detectable_effect, power_for_effect
from mlea.reference import (
    LITE_COMPETITIONS,
    UnknownExperiment,
    experiments,
    reference_rates,
    summarise,
)


def test_data_ships_with_the_package():
    assert len(experiments()) > 20


def test_reproduces_the_papers_headline_number():
    """The shipped table must aggregate to the published 16.9%, or it is wrong."""
    rates = reference_rates("models-o1-preview-aide")
    assert len(rates) == 75
    assert sum(rates) / len(rates) == pytest.approx(0.169, abs=0.002)


def test_lite_split_has_22_competitions():
    assert len(LITE_COMPETITIONS) == 22


def test_lite_filter_restricts_the_pool():
    full = reference_rates("models-o1-preview-aide")
    lite = reference_rates("models-o1-preview-aide", competitions=LITE_COMPETITIONS)
    assert len(full) == 75 and len(lite) == 22
    assert sum(lite) / len(lite) > sum(full) / len(full), "lite is the easy split"


def test_distribution_is_u_shaped_not_uniform():
    """The finding that invalidated the old Beta assumption."""
    s = summarise("models-o1-preview-aide")
    assert s["never"] == 42, "42 of 75 competitions never medalled in ~21 seeds"
    assert s["always"] == 2


def test_unknown_experiment_lists_the_options():
    with pytest.raises(UnknownExperiment, match="Available"):
        reference_rates("no-such-agent")


def test_unknown_split_combination_raises():
    with pytest.raises(UnknownExperiment, match="no competitions"):
        reference_rates("models-o1-preview-aide", competitions=frozenset({"nope"}))


def test_rates_are_probabilities():
    for exp in experiments():
        assert all(0.0 <= r <= 1.0 for r in reference_rates(exp))


# --- the empirical path in the power model ---


def test_presets_resample_real_data_rather_than_assuming_a_shape():
    for name, design in DESIGNS.items():
        pool = design.baseline_pool()
        assert pool is not None, f"{name} still assumes a Beta"
        assert len(pool) >= 20


def test_beta_fallback_still_works_for_hypotheticals():
    d = Design(name="hypothetical", n_units=30, n_seeds=3, base_rate=0.5)
    assert d.baseline_pool() is None
    assert power_for_effect(d, -0.3, n_sims=800, seed=0).power > 0


def test_fitted_beta_and_empirical_resample_agree():
    """Cross-check: the corrected parametric model and the data-driven one must
    land in the same place, or one of them is wrong."""
    empirical = DESIGNS["lite-regression"]
    fitted = Design(name="fitted", n_units=22, n_seeds=3, base_rate=0.773,
                    heterogeneity=0.7)
    a = minimum_detectable_effect(empirical, n_sims=3000, seed=0)
    b = minimum_detectable_effect(fitted, n_sims=3000, seed=0)
    assert a is not None and b is not None
    assert abs(a - b) < 0.04


def test_old_assumption_was_optimistic_on_the_contamination_design():
    """conc=3.0 understated heterogeneity; the 8-pair design is worse than it said."""
    from dataclasses import replace

    empirical = DESIGNS["live-gap"]
    old = replace(empirical, reference=None, reference_split=None,
                  base_rate=0.773, heterogeneity=3.0)
    mde_new = minimum_detectable_effect(empirical, n_sims=3000, seed=0)
    mde_old = minimum_detectable_effect(old, n_sims=3000, seed=0)
    assert mde_new is not None and mde_old is not None
    assert abs(mde_new) > abs(mde_old)


def test_default_heterogeneity_matches_the_fit():
    assert Design(name="d", n_units=10, n_seeds=3).heterogeneity == 0.7
