import pytest

from mlea.power import (
    DESIGNS,
    Design,
    minimum_detectable_effect,
    power_for_effect,
    seeds_needed,
)


def d(**kw):
    base = dict(name="t", n_units=22, n_seeds=3, base_rate=0.5, heterogeneity=3.0)
    base.update(kw)
    return Design(**base)


def test_zero_effect_gives_power_near_alpha():
    r = power_for_effect(d(), 0.0, alpha=0.05, n_sims=3000, seed=1)
    assert r.power < 0.08, "false positive rate must be near alpha under the null"


def test_power_increases_with_effect():
    powers = [
        power_for_effect(d(), e, n_sims=1500, seed=2).power
        for e in (0.0, 0.1, 0.2, 0.4)
    ]
    assert powers == sorted(powers)


def test_power_increases_with_units():
    small = power_for_effect(d(n_units=8), 0.15, n_sims=1500, seed=3).power
    large = power_for_effect(d(n_units=75), 0.15, n_sims=1500, seed=3).power
    assert large > small


def test_power_increases_with_seeds():
    few = power_for_effect(d(n_seeds=1), 0.15, n_sims=1500, seed=4).power
    many = power_for_effect(d(n_seeds=10), 0.15, n_sims=1500, seed=4).power
    assert many > few


def test_worse_matching_reduces_power():
    good = power_for_effect(d(matching_sd=0.0), 0.2, n_sims=1500, seed=5).power
    bad = power_for_effect(d(matching_sd=0.35), 0.2, n_sims=1500, seed=5).power
    assert good > bad


def test_tiny_design_has_exactly_zero_power():
    r = power_for_effect(d(n_units=4), 0.9, n_sims=500, seed=6)
    assert r.power == 0.0
    assert r.impossible
    assert "IMPOSSIBLE" in r.summary()


def test_mde_returns_none_for_impossible_design():
    assert minimum_detectable_effect(d(n_units=4), n_sims=500, seed=7) is None


def test_mde_is_an_effect_that_actually_reaches_target_power():
    design = d(n_units=22, n_seeds=3)
    mde = minimum_detectable_effect(design, target_power=0.8, n_sims=2000, seed=8)
    assert mde is not None
    at_mde = power_for_effect(design, mde, n_sims=2000, seed=8).power
    assert at_mde >= 0.78
    below = power_for_effect(design, mde * 0.5, n_sims=2000, seed=8).power
    assert below < at_mde
    assert mde < 0, "default direction is 'decrease'"


def test_mde_shrinks_as_units_grow():
    small = minimum_detectable_effect(d(n_units=10), n_sims=1500, seed=9)
    large = minimum_detectable_effect(d(n_units=75), n_sims=1500, seed=9)
    assert small is not None and large is not None
    assert abs(large) < abs(small)


def test_seeds_needed_returns_none_when_units_are_the_bottleneck():
    """Seeds cannot rescue a design with too few competitions."""
    assert seeds_needed(d(n_units=6), 0.02, max_seeds=8, n_sims=800, seed=10) is None


def test_seeds_needed_finds_a_workable_count():
    n = seeds_needed(d(n_units=75), 0.20, max_seeds=10, n_sims=1200, seed=11)
    assert n is not None and 1 <= n <= 10


def test_cost_counts_both_arms():
    assert d(n_units=8, n_seeds=3).cost_runs() == 48


def test_presets_are_well_formed():
    for name, design in DESIGNS.items():
        assert design.name == name
        assert design.n_units > 0 and design.n_seeds > 0
        assert 0.0 <= design.base_rate <= 1.0
        assert design.description


def test_direction_matters_near_the_ceiling():
    """At a high base rate the cap squashes gains but not drops."""
    design = d(n_units=22, base_rate=0.803)
    down = minimum_detectable_effect(
        design, direction="decrease", n_sims=1500, seed=12
    )
    up = minimum_detectable_effect(design, direction="increase", n_sims=1500, seed=12)
    assert down is not None
    assert up is None or abs(down) < abs(up)


def test_direction_is_validated():
    with pytest.raises(ValueError, match="direction"):
        minimum_detectable_effect(d(), direction="sideways", n_sims=200)


def test_mde_sign_follows_direction():
    down = minimum_detectable_effect(
        d(n_units=75), direction="decrease", n_sims=1200, seed=13
    )
    up = minimum_detectable_effect(
        d(n_units=75), direction="increase", n_sims=1200, seed=13
    )
    assert down is not None and down < 0
    assert up is not None and up > 0
