"""Estimating a full benchmark score from a few competitions.

:mod:`mlea.instrument` answered "which subset preserves the *ranking*". That is
the weaker question. The one practitioners actually have is **"if I run 20 of
the 75, what is my number on all 75, and how wrong is it?"** -- because a
ranking is only useful against agents you have also run, while a score is
comparable to a published one.

The current best answer to that question is the anchor-point approach from the
`tinyBenchmarks` line of work: fit an item-response model on existing results,
run a new system on a small anchor set, estimate its ability from those items
alone, and *predict* the items you skipped. This module implements it, and --
the part that matters -- measures it against the alternatives on this data
rather than assuming it wins:

* ``random`` -- a uniform draw, the honest baseline that is better than most
  hand-picked sets.
* ``stratified`` -- spread across observed difficulty. The usual practitioner
  heuristic, and the one a lite split approximates.
* ``greedy_alpha`` -- this package's own earlier method: maximise internal
  consistency. It was built for ranking, so this is a fair test of whether it
  transfers to estimation.
* ``max_information`` -- the Fisher-information-optimal set at a target ability,
  which is the textbook adaptive-testing choice.
* ``anchor`` -- cluster items by their fitted response curves and take one
  representative per cluster, which is the tinyBenchmarks construction.

Two estimators are compared on every subset:

* ``naive`` -- the mean over the items you ran. What everyone reports. Biased by
  exactly how much easier or harder the subset is than the whole.
* ``pirt`` -- estimate ability from the observed items, then add back the model's
  prediction for the unobserved ones. Unbiased when the model is right, and the
  bias when it is wrong is the thing measured below.

Everything is scored leave-one-agent-out: the model is fitted, and the subset
chosen, without ever seeing the agent whose score is then predicted. Selecting a
subset on the same results it is scored against measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .irt import IRTModel, _normal_weights, _sigmoid, fit

#: Ability grid for estimating theta from a partial observation.
_GRID = np.linspace(-6.0, 6.0, 481)


def estimate_ability(
    model: IRTModel,
    successes: np.ndarray,
    trials: np.ndarray,
    subset: Sequence[int],
) -> float:
    """Expected a-posteriori ability from a subset of items only.

    The same posterior the full fit computes, restricted to the items that were
    actually run. An agent that does well on a hard anchor set lands high even
    if the raw fraction looks unremarkable, which is the entire point of
    estimating ability rather than averaging scores.
    """
    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    ll = np.zeros_like(_GRID)
    for j in subset:
        item = model.items[j]
        # ``located``, not ``isfinite(b)``. An item nobody ever solved comes back
        # with a perfectly finite difficulty that the *prior* chose, and an item
        # with a flat slope comes back with an extrapolated one; believing either
        # curve imports a number the data never supported. Guarding on finiteness
        # instead silently let both through.
        if not item.located:
            continue
        z = item.a * (_GRID - item.b)
        p = np.clip(_sigmoid(z), 1e-9, 1 - 1e-9)
        ll += s[j] * np.log(p) + (n[j] - s[j]) * np.log1p(-p)
    w = _normal_weights(_GRID)
    post = w * np.exp(ll - ll.max())
    total = post.sum()
    if total <= 0 or not np.isfinite(total):
        return 0.0
    return float((post / total) @ _GRID)


def predict_full_rate(
    model: IRTModel,
    successes: np.ndarray,
    trials: np.ndarray,
    subset: Sequence[int],
) -> float:
    """Predicted mean rate over **all** items, from the subset that was run.

    Observed items contribute what was observed; skipped items contribute the
    model's prediction at the estimated ability. This is the estimator the
    anchor-point literature calls p-IRT, and the reason it beats a subset mean
    is that it corrects for the subset being unrepresentative in difficulty --
    which every cheap subset is.
    """
    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    theta = estimate_ability(model, s, n, subset)
    chosen = set(subset)
    total = 0.0
    for j, item in enumerate(model.items):
        if j in chosen:
            total += s[j] / n[j]
        elif item.located:
            total += float(_sigmoid(np.array([item.a * (theta - item.b)]))[0])
        else:
            # An item the model cannot locate -- never solved by anyone, or with
            # a slope too flat to place -- is predicted at its observed rate
            # rather than from ability, because its difficulty is a prior bound
            # and not a measurement. Pretending otherwise would import the prior
            # into the headline number.
            total += item.observed_rate
    return total / len(model.items)


def naive_rate(successes: np.ndarray, trials: np.ndarray, subset: Sequence[int]) -> float:
    """The mean over the items that were run. What a lite split reports."""
    s = np.asarray(successes, dtype=float)[list(subset)]
    n = np.asarray(trials, dtype=float)[list(subset)]
    return float(np.mean(s / n))


# --- subset selection strategies ------------------------------------------
#
# Every selector sees only the training matrix and the training fit. None of
# them is allowed near the held-out agent.


def select_random(model, S, N, k, rng) -> list[int]:
    return sorted(rng.choice(S.shape[1], size=k, replace=False).tolist())


def select_stratified(model, S, N, k, rng) -> list[int]:
    """Spread evenly across observed difficulty -- the practitioner's heuristic."""
    rate = S.sum(axis=0) / N.sum(axis=0)
    order = np.argsort(rate)
    bins = np.array_split(order, k)
    return sorted(int(rng.choice(b)) for b in bins if len(b))


def select_greedy_alpha(model, S, N, k, rng) -> list[int]:
    """This package's earlier ranking-oriented method, tested outside its purpose."""
    from .instrument import select_items

    return sorted(select_items(S / N, k))


def select_max_information(model, S, N, k, rng, *, theta: float = 0.0) -> list[int]:
    """Fisher-information-optimal at a target ability (adaptive-testing's rule)."""
    info = np.array([
        float(i.information(theta)) if i.located else 0.0 for i in model.items
    ])
    return sorted(np.argsort(-info)[:k].tolist())


def select_anchor(model, S, N, k, rng) -> list[int]:
    """Cluster items by their fitted response curves; take one per cluster.

    The tinyBenchmarks construction. Representing the whole suite matters more
    than each item being individually informative: a set of twenty items that
    are all informative at the same difficulty measures that difficulty twenty
    times and the rest of the scale not at all.
    """
    grid = np.linspace(-3.0, 3.0, 25)
    curves = []
    usable = []
    for j, item in enumerate(model.items):
        if not item.located:
            continue
        curves.append(_sigmoid(item.a * (grid - item.b)))
        usable.append(j)
    if len(usable) <= k:
        return sorted(usable)
    X = np.array(curves)
    centres = _kmeans(X, k, rng)
    chosen = []
    for c in centres:
        d = ((X - c) ** 2).sum(axis=1)
        for cand in np.argsort(d):
            if usable[int(cand)] not in chosen:
                chosen.append(usable[int(cand)])
                break
    return sorted(chosen)


def _kmeans(X: np.ndarray, k: int, rng, *, n_iter: int = 60) -> np.ndarray:
    """k-means with k-means++ seeding. numpy only, like everything else here."""
    n = X.shape[0]
    centres = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        total = d.sum()
        probs = np.full(n, 1.0 / n) if total <= 0 else d / total
        centres.append(X[rng.choice(n, p=probs)])
    C = np.array(centres)
    for _ in range(n_iter):
        dist = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        labels = dist.argmin(axis=1)
        new = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else C[i]
            for i in range(k)
        ])
        if np.allclose(new, C):
            break
        C = new
    return C


SELECTORS: dict[str, Callable] = {
    "random": select_random,
    "stratified": select_stratified,
    "greedy_alpha": select_greedy_alpha,
    "max_information": select_max_information,
    "anchor": select_anchor,
}


# --- the head-to-head -----------------------------------------------------


@dataclass(frozen=True)
class EstimationResult:
    selector: str
    estimator: str
    k: int
    #: Mean absolute error in predicting the full-suite rate, in rate units.
    mae: float
    #: Worst single agent, which is what actually burns a reader.
    max_error: float
    #: Signed mean error: does this combination systematically over- or
    #: under-state? A cheap subset that is biased by 4 points is worse than one
    #: that is noisy by 4, because the bias does not average out across papers.
    bias: float
    n_agents: int

    def __str__(self) -> str:
        return (
            f"{self.selector:16s} {self.estimator:5s} k={self.k:3d}  "
            f"MAE {self.mae * 100:5.2f}pp  max {self.max_error * 100:5.2f}pp  "
            f"bias {self.bias * 100:+5.2f}pp"
        )


def evaluate(
    successes: np.ndarray,
    trials: np.ndarray,
    items: Sequence[str],
    subjects: Sequence[str],
    *,
    ks: Sequence[int] = (10, 20, 30),
    selectors: Sequence[str] | None = None,
    n_repeats: int = 3,
    rng: np.random.Generator | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[EstimationResult]:
    """Leave-one-agent-out comparison of every selector-estimator pair.

    For each held-out agent: fit the model on the other agents, choose a subset
    without seeing the held-out one, pretend only that subset was run, and
    predict its full-suite rate. The error is against the agent's true rate over
    all items.

    This is the only version of the claim that can fail. A subset chosen with
    the held-out agent's results in hand will reproduce them, and an estimator
    tuned on the same agents it is scored on will look unbiased.
    """
    S = np.asarray(successes, dtype=float)
    N = np.asarray(trials, dtype=float)
    rng = rng or np.random.default_rng(0)
    names = list(selectors) if selectors else list(SELECTORS)
    unknown = set(names) - set(SELECTORS)
    if unknown:
        raise ValueError(f"unknown selector(s): {sorted(unknown)}")
    n_agents, n_items = S.shape
    if n_agents < 3:
        raise ValueError("leave-one-out needs at least 3 agents")
    if any(not 1 <= k <= n_items for k in ks):
        raise ValueError(f"every k must be in 1..{n_items}")

    truth = (S / N).mean(axis=1)
    errors: dict[tuple[str, str, int], list[float]] = {}

    for held in range(n_agents):
        keep = [i for i in range(n_agents) if i != held]
        model = fit(S[keep], N[keep], items, [subjects[i] for i in keep])
        if progress:
            progress(f"   fold {held + 1}/{n_agents}: {subjects[held]}")
        for name in names:
            selector = SELECTORS[name]
            repeats = n_repeats if name in ("random", "stratified", "anchor") else 1
            for k in ks:
                for rep in range(repeats):
                    sub = selector(model, S[keep], N[keep], k,
                                   np.random.default_rng(rng.integers(1 << 32)))
                    if not sub:
                        continue
                    for est, value in (
                        ("pirt", predict_full_rate(model, S[held], N[held], sub)),
                        ("naive", naive_rate(S[held], N[held], sub)),
                    ):
                        errors.setdefault((name, est, k), []).append(
                            value - truth[held]
                        )

    out = []
    for (name, est, k), errs in sorted(errors.items()):
        e = np.array(errs)
        out.append(EstimationResult(
            selector=name, estimator=est, k=k,
            mae=float(np.mean(np.abs(e))),
            max_error=float(np.max(np.abs(e))),
            bias=float(np.mean(e)),
            n_agents=n_agents,
        ))
    return out


__all__ = [
    "EstimationResult",
    "SELECTORS",
    "estimate_ability",
    "evaluate",
    "naive_rate",
    "predict_full_rate",
]
