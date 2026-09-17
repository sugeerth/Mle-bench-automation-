"""Measure the benchmark, not the agent.

Everything else in this package asks *how good is this agent*. This module asks
the prior question: **is this benchmark capable of telling agents apart, and at
what price?** A benchmark is a measuring instrument, and an instrument that has
never been calibrated is a number generator with good manners.

The vocabulary is classical test theory, because an agent-by-competition score
matrix is exactly a respondent-by-item matrix and the century of work on what
makes such a matrix informative applies unchanged:

* **Item discrimination** -- how strongly one competition's outcome tracks the
  agent's overall ability. A competition nobody solves, or everybody solves,
  discriminates zero: it costs full price and moves no ranking. Computed
  *corrected*, against the mean of the **other** items, because correlating an
  item with a total that contains it inflates every value and inflates short
  suites most.
* **Reliability** (Cronbach's alpha) -- what share of the observed spread in
  ability is real rather than item-sampling noise. This is the number that
  decides whether "agent A beat agent B on this benchmark" survives re-running
  it on a different set of competitions.
* **Separation and strata** -- reliability restated as *how many distinct
  ability levels the instrument can actually resolve*. A benchmark with
  reliability 0.5 sorts agents into about two bins no matter how many decimal
  places the leaderboard prints.
* **Attenuation correction** -- a medal rate measured over 3 seeds is mostly
  binomial noise, and noise drags every correlation toward zero. Given the trial
  counts, the share of each item's variance that is noise is computable, and the
  discrimination can be reported both as measured and as it would be with seeds
  to spare.

One caution stated up front, because it governs every number here: reliability
is a property of an instrument **and the population it is used on**, not of the
instrument alone. A benchmark that separates a 1%-medal agent from a 78%-medal
agent effortlessly may be near-useless between two frontier agents. So
:func:`analyse` is meant to be run on the ability band you actually care about,
and :meth:`InstrumentReport.summary` names the band it was given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

#: Relative cost of one competition, by upstream's complexity tier.
#:
#: Provenance: MLE-bench's own tiering (``experiments/splits/{low,medium,high}.txt``)
#: is defined by how long an experienced ML engineer would need -- under 2 hours,
#: 2 to 10 hours, over 10 hours. These weights are the midpoints of those bands
#: (the open-ended top band taken at 15). They are a proxy for *human* effort,
#: which is what upstream measured; agent wall-clock and GPU spend correlate with
#: it but are not the same thing, so treat a cost figure here as an ordering, not
#: a bill. Pass an explicit cost vector to :func:`cost_frontier` for a real one.
COMPLEXITY_COST: dict[str, float] = {"low": 1.0, "medium": 6.0, "high": 15.0}

#: Below this, an item's contribution to ranking agents is not worth its slot.
#: Convention from item analysis: 0.2 is the usual floor for "keep", 0.3 for
#: "good". Reported, never enforced -- dropping items is the caller's call.
WEAK_DISCRIMINATION = 0.20

#: An item whose across-subject spread is within this of zero is constant.
#:
#: Not a stylistic nicety. Scores reach this module through division (a medal
#: rate) or interpolation (a leaderboard percentile), so a competition every
#: agent scored identically on arrives with a standard deviation of 2e-16 rather
#: than 0. Testing ``sd == 0`` silently fails to flag exactly the items the
#: module exists to find. Relative to the item's own scale, so it holds for
#: scores that are not rates.
DEAD_ITEM_TOL = 1e-9


def _is_constant(sd: float, mean: float) -> bool:
    return sd <= DEAD_ITEM_TOL * max(1.0, abs(mean))


class NotEnoughData(ValueError):
    """Raised when a matrix is too small for the statistic being asked for."""


@dataclass(frozen=True)
class ItemStats:
    """One competition, viewed as a test item."""

    item: str
    mean: float
    sd: float
    #: Corrected item-rest correlation: this item against the mean of the others.
    discrimination: float
    #: The same, corrected for measurement noise in both terms. ``None`` when no
    #: trial counts were supplied, so no noise estimate exists.
    discrimination_disattenuated: float | None
    #: Share of this item's across-agent variance attributable to finite seeds.
    noise_share: float | None
    complexity: str | None = None

    @property
    def dead(self) -> bool:
        """Every agent got the same score. Carries no information at all."""
        return _is_constant(self.sd, self.mean)

    @property
    def weak(self) -> bool:
        return self.dead or self.discrimination < WEAK_DISCRIMINATION

    @property
    def trap(self) -> bool:
        """Discriminates in the wrong direction: better agents score worse.

        Not a curiosity. A competition where covariate shift is severe enough
        that predicting the training mean beats modelling is measuring "declined
        to extrapolate", and it subtracts from the ranking it is averaged into.
        An item like this is worth keeping as a diagnostic and worth removing
        from a score.
        """
        return self.discrimination < -WEAK_DISCRIMINATION

    @property
    def saturation(self) -> str | None:
        """Which end a dead item is stuck at -- floor, ceiling, or neither."""
        if not self.dead:
            return None
        if self.mean <= 0.0:
            return "floor"
        if self.mean >= 1.0:
            return "ceiling"
        return "constant"


@dataclass(frozen=True)
class InstrumentReport:
    items: tuple[ItemStats, ...]
    subjects: tuple[str, ...]
    ability: tuple[float, ...]
    #: Cronbach's alpha over items.
    reliability: float
    #: Split-half reliability, Spearman-Brown corrected, averaged over splits.
    split_half: float
    #: 95th percentile of reliability when the items are made independent by
    #: permutation. Observed reliability below this is not evidence of anything.
    reliability_null_p95: float = 0.0
    notes: tuple[str, ...] = ()

    # --- derived ----------------------------------------------------------

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def dead_items(self) -> tuple[ItemStats, ...]:
        return tuple(i for i in self.items if i.dead)

    @property
    def weak_items(self) -> tuple[ItemStats, ...]:
        return tuple(i for i in self.items if i.weak)

    @property
    def trap_items(self) -> tuple[ItemStats, ...]:
        return tuple(i for i in self.items if i.trap)

    @property
    def reliability_is_significant(self) -> bool:
        """Is the measured reliability above what independent items would give?

        Alpha is unbiased in the mean under independence but very wide when
        subjects are few: with 25 agents, unrelated items reach 0.25 one time in
        twenty by chance alone. A reliability reported without that reference is
        not interpretable, and benchmark papers essentially never report it.
        """
        return self.reliability > self.reliability_null_p95

    @property
    def separation(self) -> float:
        """Wright's separation index ``G = sqrt(r / (1 - r))``.

        The ratio of true ability spread to measurement error: how many error
        widths fit inside the spread of the population measured.
        """
        r = min(max(self.reliability, 0.0), 0.999999)
        return float(np.sqrt(r / (1.0 - r)))

    @property
    def strata(self) -> float:
        """Number of ability levels this instrument can statistically resolve.

        ``(4G + 1) / 3`` -- the count of levels separated by about three standard
        errors, i.e. distinguishable rather than merely differently printed. A
        leaderboard with more entries than strata is reporting an order it cannot
        support.
        """
        return (4.0 * self.separation + 1.0) / 3.0

    @property
    def ability_band(self) -> tuple[float, float]:
        return (float(min(self.ability)), float(max(self.ability)))

    def items_needed(self, target_reliability: float) -> int | None:
        """How many items of this average quality reach a target reliability.

        The Spearman-Brown prophecy formula: lengthening a test by a factor ``m``
        takes reliability ``r`` to ``m*r / (1 + (m-1)*r)``. Inverted, it answers
        the question a benchmark designer actually has -- *is 22 competitions
        enough, or do I need 60?* -- without running anything.

        ``None`` when the target is unreachable by lengthening (reliability at or
        below zero means the items carry no common signal to accumulate).
        """
        if not 0.0 < target_reliability < 1.0:
            raise ValueError("target_reliability must be in (0, 1)")
        r = self.reliability
        if r <= 0.0:
            return None
        if r >= target_reliability:
            return self.n_items
        m = (target_reliability * (1.0 - r)) / (r * (1.0 - target_reliability))
        return int(np.ceil(m * self.n_items))

    def cost(self, weights: dict[str, float] | None = None) -> float | None:
        """Total cost under a complexity-tier weighting, if tiers are known."""
        w = weights or COMPLEXITY_COST
        if any(i.complexity is None for i in self.items):
            return None
        return float(sum(w[i.complexity] for i in self.items))  # type: ignore[index]

    def dead_cost_share(self, weights: dict[str, float] | None = None) -> float | None:
        """Fraction of the suite's cost spent on items that move no ranking."""
        total = self.cost(weights)
        if total is None or total == 0:
            return None
        w = weights or COMPLEXITY_COST
        dead = sum(w[i.complexity] for i in self.dead_items)  # type: ignore[index]
        return float(dead / total)

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_subjects": self.n_subjects,
            "ability_band": list(self.ability_band),
            "reliability": self.reliability,
            "reliability_null_p95": self.reliability_null_p95,
            "split_half": self.split_half,
            "separation": self.separation,
            "strata": self.strata,
            "dead_items": [i.item for i in self.dead_items],
            "weak_items": [i.item for i in self.weak_items],
            "trap_items": [i.item for i in self.trap_items],
            "items": [
                {
                    "item": i.item,
                    "mean": i.mean,
                    "sd": i.sd,
                    "discrimination": i.discrimination,
                    "discrimination_disattenuated": i.discrimination_disattenuated,
                    "noise_share": i.noise_share,
                    "complexity": i.complexity,
                }
                for i in self.items
            ],
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lo, hi = self.ability_band
        lines = [
            f"{self.n_items} items x {self.n_subjects} agents, "
            f"ability {lo:.3f}-{hi:.3f}",
            f"reliability (alpha)   {self.reliability:.3f}"
            + f"   (independent items would reach "
              f"{self.reliability_null_p95:.3f} at this size)",
            f"split-half            {self.split_half:.3f}",
            f"separation G          {self.separation:.2f}",
            f"distinguishable levels{self.strata:6.1f}",
        ]
        if self.trap_items:
            lines.append(
                "inverted items       "
                + f" {len(self.trap_items)}: "
                + ", ".join(i.item for i in self.trap_items)
            )
        if self.dead_items:
            share = self.dead_cost_share()
            cost = "" if share is None else f", {share:.0%} of suite cost"
            lines.append(
                f"dead items            {len(self.dead_items)}{cost}"
            )
        lines.extend(self.notes)
        return "\n".join(lines)


# --- core statistics ------------------------------------------------------


def cronbach_alpha(matrix: np.ndarray) -> float:
    """Internal-consistency reliability of a subject-by-item score matrix.

    ``k/(k-1) * (1 - sum(item variances) / variance of the total)``. Interpreted
    here as: of the spread in agent ability this benchmark reports, what fraction
    would survive swapping these competitions for a different sample of
    comparable ones.
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise NotEnoughData("expected a 2-D subject-by-item matrix")
    n_subjects, k = M.shape
    if k < 2:
        raise NotEnoughData("alpha needs at least 2 items")
    if n_subjects < 2:
        raise NotEnoughData("alpha needs at least 2 subjects")
    total_var = M.sum(axis=1).var(ddof=1)
    if total_var <= 0:
        return 0.0
    return float(k / (k - 1) * (1.0 - M.var(axis=0, ddof=1).sum() / total_var))


def split_half_reliability(
    matrix: np.ndarray, *, n_splits: int = 200, rng: np.random.Generator | None = None
) -> float:
    """Mean Spearman-Brown-corrected correlation between random half-suites.

    Alpha is the expectation of this over all splits, so the two should agree;
    reporting both is a cheap check that neither is being driven by one item.
    Averaging over random splits rather than taking odd-versus-even matters: with
    items ordered by anything at all, a single split is a coin flip.
    """
    M = np.asarray(matrix, dtype=float)
    n_subjects, k = M.shape
    if k < 4 or n_subjects < 3:
        raise NotEnoughData("split-half needs at least 4 items and 3 subjects")
    rng = rng or np.random.default_rng(0)
    vals = []
    half = k // 2
    for _ in range(n_splits):
        perm = rng.permutation(k)
        a = M[:, perm[:half]].mean(axis=1)
        b = M[:, perm[half : 2 * half]].mean(axis=1)
        if a.std() == 0 or b.std() == 0:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        vals.append(2 * r / (1 + r) if r > -1 else -1.0)  # Spearman-Brown
    if not vals:
        return 0.0
    return float(np.mean(vals))


def null_reliability(
    matrix: np.ndarray,
    *,
    n_draws: int = 400,
    quantile: float = 0.95,
    rng: np.random.Generator | None = None,
) -> float:
    """Reliability reachable when the items share nothing -- the null band.

    Each column is shuffled independently, which destroys the alignment between
    items while preserving every item's own distribution. That is a stronger
    null than simulating Gaussian noise: it holds the marginals, the ceilings and
    the floors fixed, so what is left is exactly the question asked -- how much
    apparent common signal does a matrix this shape produce by accident?
    """
    M = np.asarray(matrix, dtype=float)
    rng = rng or np.random.default_rng(0)
    vals = []
    for _ in range(n_draws):
        shuffled = np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])])
        try:
            vals.append(cronbach_alpha(shuffled))
        except NotEnoughData:
            raise
    return float(np.quantile(vals, quantile))


def kendall_tau_b(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation with a tie correction.

    Tau-b rather than tau-a because benchmark scores tie constantly -- two agents
    with the same medal count are a tie, not a coin flip, and tau-a would charge
    for it. Implemented directly rather than pulled from scipy: this package has
    a numpy-only floor on purpose, so it runs where scipy does not.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("inputs must be the same length")
    n = x.size
    if n < 2:
        raise NotEnoughData("tau needs at least 2 items")
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(n, k=1)
    sx, sy = dx[iu], dy[iu]
    concordant = float(np.sum((sx * sy) > 0))
    discordant = float(np.sum((sx * sy) < 0))
    tie_x = float(np.sum((sx == 0) & (sy != 0)))
    tie_y = float(np.sum((sy == 0) & (sx != 0)))
    denom = np.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def _noise_variance(rates: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """Per-item binomial sampling variance of a rate measured over finite seeds.

    ``p(1-p)/n`` averaged over subjects. This is the part of an item's spread
    that is the seed draw rather than the agent, and it is the reason a
    3-seed medal rate must not be read as a measurement.
    """
    p = np.clip(np.asarray(rates, dtype=float), 0.0, 1.0)
    n = np.maximum(np.asarray(trials, dtype=float), 1.0)
    return (p * (1.0 - p) / n).mean(axis=0)


def analyse(
    matrix: np.ndarray,
    items: Sequence[str],
    subjects: Sequence[str],
    *,
    trials: np.ndarray | None = None,
    complexity: dict[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> InstrumentReport:
    """Full instrument analysis of a subject-by-item score matrix.

    ``matrix[i, j]`` is subject ``i``'s score on item ``j``, in any unit that is
    comparable across items -- a medal rate, a leaderboard percentile, a resolved
    fraction. Raw metric values are *not* comparable across items (an AUC and an
    RMSE do not average), and mixing them silently produces an alpha that means
    nothing; convert first.

    ``trials`` (same shape) is the number of seeds behind each cell, when scores
    are rates. Supplying it turns on the attenuation correction, which is the
    difference between "this competition does not separate agents" and "we did
    not run enough seeds to tell".
    """
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise NotEnoughData("expected a 2-D subject-by-item matrix")
    if M.shape != (len(subjects), len(items)):
        raise ValueError("matrix shape must be (len(subjects), len(items))")
    if M.shape[0] < 2:
        raise NotEnoughData("instrument analysis needs at least 2 subjects")
    if not np.isfinite(M).all():
        raise ValueError("matrix contains non-finite scores")

    rng = rng or np.random.default_rng(0)
    noise = _noise_variance(M, trials) if trials is not None else None
    item_sd = M.std(axis=0, ddof=1)
    item_var = item_sd**2

    stats: list[ItemStats] = []
    for j, name in enumerate(items):
        rest = np.delete(M, j, axis=1).mean(axis=1)
        col = M[:, j]
        # The same tolerance as :attr:`ItemStats.dead`, and for the same reason:
        # a constant column arrives with a standard deviation of 2e-16, and
        # feeding it to a correlation returns a meaningless 1e-17 instead of the
        # zero the item deserves.
        if (_is_constant(float(col.std(ddof=1)), float(col.mean()))
                or _is_constant(float(rest.std(ddof=1)), float(rest.mean()))):
            disc = 0.0
        else:
            disc = float(np.corrcoef(col, rest)[0, 1])
        share = None
        adjusted = None
        if noise is not None:
            share = float(noise[j] / item_var[j]) if item_var[j] > 0 else 1.0
            share = min(max(share, 0.0), 1.0)
            # Correction for attenuation. The rest-score averages many items, so
            # its own noise is already small; both terms are corrected anyway,
            # and the result is clamped -- an estimated reliability can exceed
            # the correlation and push the ratio past 1, which is an artefact of
            # estimating two variances, not a correlation above unity.
            rest_rel = 1.0
            if item_var.sum() > 0:
                rest_noise = float(noise.sum() - noise[j]) / max(M.shape[1] - 1, 1) ** 2
                rest_var = float(rest.var(ddof=1))
                rest_rel = max(1.0 - rest_noise / rest_var, 1e-6) if rest_var > 0 else 1.0
            denom = np.sqrt(max(1.0 - share, 1e-6) * rest_rel)
            adjusted = float(np.clip(disc / denom, -1.0, 1.0))
        stats.append(
            ItemStats(
                item=name,
                mean=float(col.mean()),
                sd=float(item_sd[j]),
                discrimination=disc,
                discrimination_disattenuated=adjusted,
                noise_share=share,
                complexity=(complexity or {}).get(name),
            )
        )

    reliability = cronbach_alpha(M)
    null_p95 = null_reliability(M, rng=rng)
    try:
        half = split_half_reliability(M, rng=rng)
    except NotEnoughData:
        half = float("nan")

    notes: list[str] = []
    if reliability <= null_p95:
        notes.append(
            f"WARNING: reliability {reliability:.3f} is within the range "
            f"independent items reach by chance at this size "
            f"(95th percentile {null_p95:.3f}). This suite is not measuring a "
            f"common ability."
        )
    if M.shape[0] < 10:
        notes.append(
            f"note: {M.shape[0]} subjects. Reliability is itself estimated, and "
            f"below ~10 subjects it is estimated badly."
        )
    if noise is not None:
        overall = float(noise.sum() / item_var.sum()) if item_var.sum() > 0 else 1.0
        notes.append(f"note: {overall:.1%} of item variance is finite-seed noise.")

    return InstrumentReport(
        items=tuple(stats),
        subjects=tuple(subjects),
        ability=tuple(float(v) for v in M.mean(axis=1)),
        reliability=reliability,
        split_half=half,
        reliability_null_p95=null_p95,
        notes=tuple(notes),
    )


# --- choosing a cheaper suite ---------------------------------------------


def select_items(
    matrix: np.ndarray,
    k: int,
    *,
    cost: Sequence[float] | None = None,
    budget: float | None = None,
) -> list[int]:
    """Greedily pick items that make the most reliable suite of size ``k``.

    Reliability, not discrimination, is the objective: two items that each track
    ability well but track *each other* perfectly are one item's worth of
    information at two items' price, and only a joint criterion notices. Greedy
    is not optimal, but the alternative is a subset search over ``C(75, 22)``.

    With ``cost`` supplied the gain is per unit cost, and ``budget`` caps the
    total -- which is the question a benchmark maintainer actually faces: not
    *which 22*, but *what is the best suite I can afford*.
    """
    M = np.asarray(matrix, dtype=float)
    n_items = M.shape[1]
    if not 1 <= k <= n_items:
        raise ValueError(f"k must be in 1..{n_items}")
    costs = np.ones(n_items) if cost is None else np.asarray(cost, dtype=float)
    if costs.shape != (n_items,):
        raise ValueError("cost must have one entry per item")
    if (costs <= 0).any():
        raise ValueError("costs must be positive")

    chosen: list[int] = []
    spent = 0.0
    while len(chosen) < k:
        best, best_gain = None, -np.inf
        for j in range(n_items):
            if j in chosen:
                continue
            if budget is not None and spent + costs[j] > budget:
                continue
            trial = chosen + [j]
            if len(trial) < 2:
                # Alpha is undefined for one item; seed on raw spread instead.
                gain = float(M[:, j].std(ddof=1))
            else:
                try:
                    gain = cronbach_alpha(M[:, trial])
                except NotEnoughData:
                    continue
                if not np.isfinite(gain):
                    continue
            gain = gain / costs[j] if cost is not None else gain
            if gain > best_gain:
                best, best_gain = j, gain
        if best is None:
            break  # budget exhausted
        chosen.append(best)
        spent += float(costs[best])
    return chosen


def ranking_fidelity(
    matrix: np.ndarray, subset: Sequence[int], reference: Sequence[int] | None = None
) -> float:
    """Tau-b between the ranking a subset produces and the full suite's.

    The practical question behind every "lite" split: if I run a quarter of the
    benchmark, do I get the same answer? Note this compares *rankings*, not
    scores -- a subset of easier competitions inflates every score without
    necessarily changing who is on top.
    """
    M = np.asarray(matrix, dtype=float)
    ref = list(range(M.shape[1])) if reference is None else list(reference)
    return kendall_tau_b(M[:, list(subset)].mean(axis=1), M[:, ref].mean(axis=1))


@dataclass(frozen=True)
class FidelityEstimate:
    """Held-out fidelity of a selection rule, with the spread over splits."""

    mean: float
    sd: float
    n_splits: int
    selection_counts: dict[int, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.mean:.3f} +- {self.sd:.3f} (n={self.n_splits})"


def cross_validated_fidelity(
    matrix: np.ndarray,
    selector: Callable[[np.ndarray], Sequence[int]],
    *,
    n_splits: int = 40,
    rng: np.random.Generator | None = None,
) -> FidelityEstimate:
    """Fidelity of a *selection rule*, measured on subjects it did not see.

    This is the guard that makes subset selection honest. Choosing the 22 most
    informative competitions using every agent's results and then reporting how
    well those 22 rank the same agents measures nothing but the fitting: any
    subset picked to reproduce a ranking will reproduce that ranking. Splitting
    the agents, selecting on one half and scoring the ranking of the other half
    is the only version of the claim that can fail.
    """
    M = np.asarray(matrix, dtype=float)
    n_subjects = M.shape[0]
    if n_subjects < 6:
        raise NotEnoughData("cross-validated fidelity needs at least 6 subjects")
    rng = rng or np.random.default_rng(0)
    vals: list[float] = []
    counts: dict[int, int] = {}
    cut = n_subjects // 2
    for _ in range(n_splits):
        perm = rng.permutation(n_subjects)
        train, test = perm[:cut], perm[cut:]
        subset = list(selector(M[train]))
        if not subset:
            continue
        for j in subset:
            counts[j] = counts.get(j, 0) + 1
        tau = ranking_fidelity(M[test], subset)
        if np.isfinite(tau):
            vals.append(tau)
    if not vals:
        raise NotEnoughData("no split produced a usable ranking")
    return FidelityEstimate(
        mean=float(np.mean(vals)),
        sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        n_splits=len(vals),
        selection_counts=counts,
    )


def cost_frontier(
    matrix: np.ndarray,
    costs: Sequence[float],
    budgets: Sequence[float],
    *,
    rng: np.random.Generator | None = None,
    n_splits: int = 20,
) -> list[tuple[float, int, float, float]]:
    """``(budget, n_items, held-out tau, sd)`` for each budget.

    The curve a benchmark maintainer should publish alongside the benchmark:
    how much ranking fidelity a given spend buys. Without it, "lite" splits get
    chosen on intuition about which competitions look cheap.
    """
    M = np.asarray(matrix, dtype=float)
    cost_arr = np.asarray(costs, dtype=float)
    out = []
    for b in budgets:
        def sel(train: np.ndarray, _b: float = float(b)) -> list[int]:
            return select_items(train, M.shape[1], cost=cost_arr, budget=_b)

        try:
            est = cross_validated_fidelity(M, sel, n_splits=n_splits, rng=rng)
        except NotEnoughData:
            continue
        chosen = sel(M)
        out.append((float(b), len(chosen), est.mean, est.sd))
    return out


__all__ = [
    "COMPLEXITY_COST",
    "FidelityEstimate",
    "InstrumentReport",
    "ItemStats",
    "NotEnoughData",
    "WEAK_DISCRIMINATION",
    "analyse",
    "cost_frontier",
    "DEAD_ITEM_TOL",
    "cronbach_alpha",
    "cross_validated_fidelity",
    "kendall_tau_b",
    "null_reliability",
    "ranking_fidelity",
    "select_items",
    "split_half_reliability",
]
