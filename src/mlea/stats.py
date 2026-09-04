"""Paired inference primitives, shared by the comparison and power modules.

Both modules use the *same* test function, so a power calculation describes the
analysis that will actually be run rather than a faster approximation of it.
"""

from __future__ import annotations

import itertools

import numpy as np

#: Above this many units, exact sign-flip enumeration is replaced by sampling.
EXACT_ENUMERATION_LIMIT = 15


def sign_matrix(n_units: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Sign-flip patterns for a paired randomisation test, shape ``(n, n_units)``.

    Under the null of no condition effect, the label assignment within a unit is
    exchangeable, so negating a unit's difference is an equally likely outcome.
    For small designs every pattern is enumerated, which makes the resulting
    p-value exact rather than simulated -- the pre/post-cutoff design has ~8
    pairs, so it lands in the exact regime.
    """
    if n_units <= EXACT_ENUMERATION_LIMIT:
        return np.array(list(itertools.product([1.0, -1.0], repeat=n_units)))
    return rng.choice([1.0, -1.0], size=(n_perm, n_units))


def permutation_pvalues(diffs: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Two-sided paired sign-flip p-values.

    ``diffs`` is ``(n_sims, n_units)``; returns ``(n_sims,)``. Uses the
    add-one correction, so a p-value is never 0 -- with B patterns the smallest
    attainable value is ``1/(B+1)``, which is what makes tiny designs honest
    about the resolution they actually have.
    """
    diffs = np.atleast_2d(diffs)
    observed = np.abs(diffs.mean(axis=1))
    # (n_sims, n_perm): the mean difference under each relabelling.
    permuted = np.abs(diffs @ signs.T) / diffs.shape[1]
    n_perm = signs.shape[0]
    at_least_as_extreme = (permuted >= observed[:, None] - 1e-12).sum(axis=1)
    return (at_least_as_extreme + 1.0) / (n_perm + 1.0)


def cluster_bootstrap_ci(
    diffs: np.ndarray,
    *,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile CI, resampling *units* (competitions), not runs.

    Competitions are the cluster: resampling individual runs would treat seeds
    of the same competition as independent and understate the interval badly,
    because between-competition variance dominates on this benchmark.
    """
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def smallest_detectable_pvalue(n_units: int, n_perm: int) -> float:
    """The smallest p-value a design can ever produce.

    Under exact enumeration the two-sided statistic ``|mean|`` is invariant to
    negating a sign pattern, so every pattern is tied with its negation and the
    extreme count is never below 2. With the add-one correction the numerator is
    therefore never below 3, giving a floor of ``3/(2**n + 1)``.

    The practical consequence: a paired design needs **at least 6 units** to be
    able to reach p<0.05 at all (n=5 floors at 3/33 = 0.091, n=6 at 3/65 =
    0.046). A smaller design cannot produce a significant result however large
    the true effect -- worth knowing before running it, not after.
    """
    if n_units <= EXACT_ENUMERATION_LIMIT:
        return 3.0 / (2**n_units + 1.0)
    return 1.0 / (n_perm + 1.0)


__all__ = [
    "EXACT_ENUMERATION_LIMIT",
    "cluster_bootstrap_ci",
    "permutation_pvalues",
    "sign_matrix",
    "smallest_detectable_pvalue",
]
