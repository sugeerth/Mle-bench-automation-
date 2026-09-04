import numpy as np
import pytest

from mlea.stats import (
    cluster_bootstrap_ci,
    permutation_pvalues,
    sign_matrix,
    smallest_detectable_pvalue,
)


def test_sign_matrix_is_exact_enumeration_when_small():
    m = sign_matrix(4, n_perm=999, rng=np.random.default_rng(0))
    assert m.shape == (16, 4)
    assert {tuple(r) for r in m}.__len__() == 16


def test_sign_matrix_samples_when_large():
    m = sign_matrix(20, n_perm=101, rng=np.random.default_rng(0))
    assert m.shape == (101, 20)


def test_pvalue_floor_accounts_for_two_sided_tie():
    """|mean| ties each pattern with its negation; with +1 correction the floor is 3/(2**n+1)."""
    assert smallest_detectable_pvalue(4, 999) == pytest.approx(3 / 17)
    assert smallest_detectable_pvalue(8, 999) == pytest.approx(3 / 257)
    assert smallest_detectable_pvalue(30, 999) == pytest.approx(1 / 1000)


def test_six_units_is_the_minimum_viable_paired_design():
    assert smallest_detectable_pvalue(5, 999) > 0.05
    assert smallest_detectable_pvalue(6, 999) < 0.05


@pytest.mark.parametrize("n_units", [3, 4, 5])
def test_tiny_designs_cannot_reach_significance(n_units):
    """Up to 5 units, no effect however large produces p<0.05."""
    rng = np.random.default_rng(0)
    diffs = np.ones((1, n_units))  # maximally extreme
    p = permutation_pvalues(diffs, sign_matrix(n_units, 999, rng))[0]
    assert p > 0.05


def test_all_positive_differences_give_minimum_pvalue():
    rng = np.random.default_rng(0)
    diffs = np.ones((1, 10))
    p = permutation_pvalues(diffs, sign_matrix(10, 999, rng))[0]
    assert p == pytest.approx(smallest_detectable_pvalue(10, 999))


def test_zero_effect_is_not_significant():
    rng = np.random.default_rng(0)
    diffs = np.zeros((1, 12))
    p = permutation_pvalues(diffs, sign_matrix(12, 999, rng))[0]
    assert p == pytest.approx(1.0)


def test_pvalue_is_symmetric_under_sign_flip():
    """Two-sided: a uniform drop is as significant as a uniform gain."""
    rng = np.random.default_rng(0)
    signs = sign_matrix(10, 999, rng)
    d = np.array([[0.3, 0.1, 0.6, 0.2, 0.4, 0.5, 0.1, 0.3, 0.2, 0.4]])
    assert permutation_pvalues(d, signs)[0] == pytest.approx(
        permutation_pvalues(-d, signs)[0]
    )


def test_false_positive_rate_is_controlled():
    """Under the null the test must reject at about alpha, not more."""
    rng = np.random.default_rng(7)
    n_sims, n_units = 3000, 22
    # Null: symmetric differences with zero mean.
    diffs = rng.normal(0, 0.3, size=(n_sims, n_units))
    pvals = permutation_pvalues(diffs, sign_matrix(n_units, 999, rng))
    assert (pvals < 0.05).mean() < 0.075


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    diffs = np.full(20, 0.25)
    lo, hi = cluster_bootstrap_ci(diffs, n_boot=2000, alpha=0.05, rng=rng)
    assert lo == pytest.approx(0.25) and hi == pytest.approx(0.25)


def test_bootstrap_ci_widens_with_spread():
    rng = np.random.default_rng(0)
    tight = cluster_bootstrap_ci(
        np.array([0.1, 0.1, 0.12, 0.08] * 5), n_boot=2000, alpha=0.05, rng=rng
    )
    wide = cluster_bootstrap_ci(
        np.array([-0.8, 0.9, -0.7, 1.0] * 5), n_boot=2000, alpha=0.05, rng=rng
    )
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])
