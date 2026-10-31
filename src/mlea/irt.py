"""A two-parameter item-response model of a benchmark.

:mod:`mlea.instrument` applies classical test theory, which answers "how
reliable is this suite" with a single number for the whole population. That
single number is the thing wrong with it. A benchmark is not uniformly sharp:
it resolves ability well in some range and not at all outside it, and the range
moves as agents improve. Classical test theory cannot see that, because it has
no ability scale -- only a total score, whose meaning depends on which items
happen to be in the suite.

``docs/BENCHMARK-QUALITY.md`` originally said a full item-response model was
"not identifiable" on 25 agents. That was wrong, and wrong in a way worth
naming: the parameters being estimated are **per item**, not per agent, and
each of the 25x75 cells carries 3 to 39 Bernoulli trials -- about 9,000 in all.
25 is a small number of *people*, which is what the usual rule of thumb is about;
it is not a small amount of data. Parameter recovery on data simulated at
exactly this shape is in the tests, and the fitted model beats both a Rasch model
and the classical item-mean baseline on held-out cells.

The model is the standard 2PL::

    P(medal | agent i, competition j) = sigmoid(a_j * (theta_i - b_j))

where ``theta`` is agent ability on a scale fixed by the prior ``theta ~ N(0, 1)``,
``b_j`` is a competition's difficulty **on that same scale**, and ``a_j`` is how
sharply it separates agents near its difficulty. Fitted by marginal maximum
likelihood with EM over a quadrature grid, so ability is integrated out rather
than estimated jointly -- joint estimation biases item parameters, and the bias
does not vanish with more agents.

Three things fall out that classical test theory cannot give:

* **Difficulty and discrimination separate.** A competition nobody solves and a
  competition that tells you nothing are the same item under CTT (both have zero
  variance). Here the first gets a large ``b`` and the second a small ``a``.
* **Information is a function of ability.** Each competition is informative in a
  band around its own difficulty and nearly useless far from it. Summing those
  curves gives the benchmark's standard error **at every ability level** -- and
  therefore the level at which it goes blind.
* **A suite can be chosen for a target ability.** "Which 20 competitions best
  separate agents near the current frontier" is a different question from "which
  20 are most reliable overall", and only this model can be asked it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Quadrature grid for integrating ability out.
#:
#: The span is not decoration. At +-4 the published agents ran into the edge of
#: the grid -- the strongest sat at +3.97 and the log-likelihood was 4.7 nats
#: worse than it should have been, because the grid, not the data, was deciding
#: where the top of the scale was. At +-6 the fit is identical at 121, 241, 321
#: and 481 nodes, which is what "the quadrature is not the limiting
#: approximation" has to mean before it can be claimed.
N_QUADRATURE = 241
QUADRATURE_RANGE = 6.0

#: Weakly informative priors, which exist to make dead items finite rather than
#: to shrink live ones. An item nobody has ever solved has a likelihood that
#: pushes its difficulty to infinity; the prior stops at a stated place instead
#: of at whatever the optimiser's step limit happens to be. Discrimination is
#: left free in sign -- forcing it positive would hide the inverted items that
#: are worth finding.
PRIOR_A_MEAN, PRIOR_A_SD = 1.0, 0.6
PRIOR_B_SD = 4.0

#: Below this, ``b = -d/a`` is not a difficulty, it is a division by zero.
#:
#: Compared against ``|a| * SD(ability)`` rather than ``|a|`` alone, because a
#: slope only means something relative to the ability scale it is a slope over.
#: Testing the raw value made the classification depend on whether the fit had
#: been normalised: the same competition came back "too flat to place" on one
#: scale and "placeable but extrapolated" on the other, from identical data.
MIN_IDENTIFIED_A = 0.15


class FitFailed(RuntimeError):
    """Raised when EM did not converge to anything usable."""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branchless and overflow-free: exp(-|z|) never overflows.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass(frozen=True)
class ItemParameters:
    """One competition's place on the ability scale."""

    item: str
    #: Discrimination: the slope of the response curve at its steepest.
    a: float
    #: Difficulty: the ability at which a medal is a coin flip.
    b: float
    #: False when the data cannot locate this item -- no agent has ever solved
    #: it, or every agent always does. Its difficulty is then a bound set by the
    #: prior, not a measurement, and saying so is the whole point of the flag.
    identified: bool
    #: Observed medal rate, for the reader who wants the raw number.
    observed_rate: float
    #: True when the fitted difficulty falls outside the ability range actually
    #: measured, so it is an extrapolation from the tail of the response curve
    #: rather than a reading off its steep part.
    #:
    #: This catches a failure that ``identified`` alone does not. An item with a
    #: shallow slope can be pushed to an absurd difficulty while still varying
    #: across agents: on the published data,
    #: ``tensorflow-speech-recognition-challenge`` lands at difficulty +10.4 with
    #: a slope of 0.33, which is the model saying "this competition barely tracks
    #: ability" in a form that looks like "this competition is impossibly hard".
    extrapolated: bool = False
    complexity: str | None = None

    def information(self, theta: np.ndarray | float) -> np.ndarray:
        """Fisher information this item carries at each ability.

        ``a^2 * p * (1 - p)``: peaks at ``theta = b`` and falls off like a
        logistic bump. This is why a benchmark cannot be summarised by one
        difficulty -- an item is informative *somewhere*, and the question is
        always where.
        """
        theta = np.asarray(theta, dtype=float)
        p = _sigmoid(self.a * (theta - self.b))
        return (self.a**2) * p * (1.0 - p)

    @property
    def peak_information(self) -> float:
        return float(self.a**2 / 4.0)

    @property
    def located(self) -> bool:
        """The difficulty is a measurement, not a bound or an extrapolation."""
        return self.identified and not self.extrapolated


@dataclass(frozen=True)
class IRTModel:
    items: tuple[ItemParameters, ...]
    subjects: tuple[str, ...]
    #: Expected a-posteriori ability for each subject.
    ability: tuple[float, ...]
    #: Posterior SD of each subject's ability -- the error bar the leaderboard
    #: does not print.
    ability_se: tuple[float, ...]
    log_likelihood: float
    n_iterations: int
    converged: bool
    #: Seeds per competition assumed when reporting test information.
    seeds: float = 3.0
    #: Attenuation of discrimination at this design, if it has been calibrated.
    #: ``None`` until :func:`calibrate_attenuation` has been run.
    attenuation: float | None = None

    # --- the test as a measuring device ----------------------------------

    def information(
        self,
        theta: np.ndarray | float,
        *,
        seeds: float | None = None,
        corrected: bool = False,
    ):
        """Total Fisher information at each ability, over all items.

        Information adds across items and across independent seeds, which is
        exactly why running the same competition three times and running three
        different competitions are not the same purchase: three seeds of one
        item triple information *at that item's difficulty only*.

        ``corrected`` divides out the attenuation measured by
        :func:`calibrate_attenuation`. Discrimination is estimated with a
        downward bias at small subject counts, and information goes as its
        square, so the uncorrected curve **understates** how sharp the benchmark
        is. Both are reported rather than one silently chosen: the raw curve is
        what the data says, the corrected one is what it would say with enough
        agents to estimate the slopes cleanly.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        n = self.seeds if seeds is None else seeds
        total = np.zeros_like(theta)
        for item in self.items:
            total += n * item.information(theta)
        if corrected:
            if self.attenuation is None:
                raise ValueError(
                    "no attenuation calibration on this model; run "
                    "calibrate_attenuation() first"
                )
            total = total / (self.attenuation**2)
        return total

    def standard_error(
        self, theta: np.ndarray | float, *, seeds: float | None = None,
        corrected: bool = False,
    ):
        """``1 / sqrt(information)`` -- the resolution of the benchmark at each ability."""
        info = self.information(theta, seeds=seeds, corrected=corrected)
        with np.errstate(divide="ignore"):
            return 1.0 / np.sqrt(np.where(info > 0, info, np.nan))

    def expected_rate(self, theta: np.ndarray | float):
        """Mean medal rate across the suite at a given ability.

        The bridge back to the number people actually quote. An ability of
        ``+1.2`` means nothing to a reader; "the ability at which an agent
        medals on 71% of the suite" does.
        """
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        out = np.zeros_like(theta)
        for item in self.items:
            out += _sigmoid(item.a * (theta - item.b))
        return out / len(self.items)

    def resolving_range(
        self, target_se: float = 0.3, *, seeds: float | None = None, grid=None,
        corrected: bool = False,
    ) -> tuple[float, float] | None:
        """The ability interval the benchmark resolves to ``target_se`` or better.

        Outside it, two agents separated by less than the printed difference are
        not distinguishable by this benchmark however many decimal places the
        leaderboard carries. ``None`` when no ability is resolved that well.
        """
        grid = _default_grid() if grid is None else np.asarray(grid, dtype=float)
        se = self.standard_error(grid, seeds=seeds, corrected=corrected)
        ok = np.isfinite(se) & (se <= target_se)
        if not ok.any():
            return None
        idx = np.flatnonzero(ok)
        return float(grid[idx[0]]), float(grid[idx[-1]])

    def blind_above(
        self, target_se: float = 0.3, *, seeds: float | None = None,
        corrected: bool = False,
    ) -> float | None:
        """Ability above which the benchmark stops resolving to ``target_se``.

        The number a benchmark maintainer should watch. When the strongest agents
        pass it, the leaderboard is still printing an order it can no longer
        support.
        """
        rng = self.resolving_range(target_se, seeds=seeds, corrected=corrected)
        return None if rng is None else rng[1]

    def marginal_reliability(self, *, seeds: float | None = None, grid=None) -> float:
        """Reliability as classical test theory would report it, done properly.

        ``1 - E[SE^2] / Var(theta)`` with the expectation taken over the prior
        ability distribution rather than over whichever agents happened to be
        measured. That removes the dependence on the sample that makes a plain
        alpha incomparable between papers.
        """
        grid = _default_grid() if grid is None else np.asarray(grid, dtype=float)
        w = _normal_weights(grid)
        se = self.standard_error(grid, seeds=seeds)
        err = float(np.nansum(w * se**2))
        return float(1.0 - err / 1.0)

    # --- choosing a suite -------------------------------------------------

    def select_for_ability(
        self, k: int, theta: float, *, exclude: Sequence[str] = ()
    ) -> list[str]:
        """The ``k`` competitions carrying most information at a target ability.

        A different question from "the ``k`` most reliable overall", and the one
        worth asking when the agents being compared are all near the frontier.
        """
        if not 1 <= k <= len(self.items):
            raise ValueError(f"k must be in 1..{len(self.items)}")
        blocked = set(exclude)
        ranked = sorted(
            (i for i in self.items if i.item not in blocked),
            key=lambda i: -float(i.information(theta)),
        )
        return [i.item for i in ranked[:k]]

    @property
    def unidentified_items(self) -> tuple[ItemParameters, ...]:
        return tuple(i for i in self.items if not i.identified)

    @property
    def located_items(self) -> tuple[ItemParameters, ...]:
        """Items whose difficulty the measured agents actually pin down."""
        return tuple(i for i in self.items if i.located)

    @property
    def extrapolated_items(self) -> tuple[ItemParameters, ...]:
        return tuple(i for i in self.items if i.identified and i.extrapolated)

    @property
    def ability_range(self) -> tuple[float, float]:
        return (float(min(self.ability)), float(max(self.ability)))

    @property
    def inverted_items(self) -> tuple[ItemParameters, ...]:
        """Items whose fitted discrimination is negative: better agents do worse."""
        return tuple(i for i in self.items if i.identified and i.a < 0)

    def item(self, name: str) -> ItemParameters:
        for i in self.items:
            if i.item == name:
                return i
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "log_likelihood": self.log_likelihood,
            "seeds": self.seeds,
            "marginal_reliability": self.marginal_reliability(),
            "subjects": [
                {"subject": s, "ability": a, "ability_se": e}
                for s, a, e in zip(self.subjects, self.ability, self.ability_se)
            ],
            "items": [
                {
                    "item": i.item, "a": i.a, "b": i.b,
                    "identified": i.identified,
                    "extrapolated": i.extrapolated,
                    "observed_rate": i.observed_rate,
                    "complexity": i.complexity,
                }
                for i in self.items
            ],
        }


def _default_grid(n: int = 241, span: float = 6.0) -> np.ndarray:
    return np.linspace(-span, span, n)


def _normal_weights(grid: np.ndarray) -> np.ndarray:
    w = np.exp(-0.5 * grid**2)
    return w / w.sum()


def _quadrature() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.linspace(-QUADRATURE_RANGE, QUADRATURE_RANGE, N_QUADRATURE)
    return nodes, _normal_weights(nodes)


def _log_p(a: np.ndarray, d: np.ndarray, nodes: np.ndarray):
    """Stable ``log p`` and ``log(1-p)`` on the quadrature grid, shape (items, nodes)."""
    z = np.outer(a, nodes) + d[:, None]
    # log(sigmoid(z)) = -log1p(exp(-z)), computed on the stable branch.
    log_p = np.where(z >= 0, -np.log1p(np.exp(-np.abs(z))), z - np.log1p(np.exp(-np.abs(z))))
    log_q = log_p - z
    return log_p, log_q


def _m_step_item(
    r: np.ndarray, f: np.ndarray, nodes: np.ndarray, a0: float, d0: float,
    *, n_newton: int = 25, ridge: float = 1e-6,
) -> tuple[float, float]:
    """One item's weighted logistic fit against expected counts, with priors.

    Parametrised by slope ``a`` and intercept ``d``; difficulty is ``-d/a``.
    Fitting ``b`` directly makes the Hessian singular exactly where ``a`` is
    small, which is precisely the items that need the most care.
    """
    a, d = a0, d0
    for _ in range(n_newton):
        z = a * nodes + d
        p = _sigmoid(z)
        resid = r - f * p
        w = f * p * (1.0 - p)
        # Priors: a ~ N(PRIOR_A_MEAN, PRIOR_A_SD^2), b = -d/a handled through a
        # prior on d, which is the parametrisation the Newton step is stable in.
        g_a = float(np.sum(nodes * resid) - (a - PRIOR_A_MEAN) / PRIOR_A_SD**2)
        g_d = float(np.sum(resid) - d / PRIOR_B_SD**2)
        h_aa = float(-np.sum(w * nodes**2) - 1.0 / PRIOR_A_SD**2)
        h_ad = float(-np.sum(w * nodes))
        h_dd = float(-np.sum(w) - 1.0 / PRIOR_B_SD**2)
        H = np.array([[h_aa - ridge, h_ad], [h_ad, h_dd - ridge]])
        g = np.array([g_a, g_d])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        # Newton on a concave objective moves against the gradient through the
        # Hessian; damped, because an item with almost no information can
        # otherwise take a step the size of the prior.
        a_new, d_new = a - step[0], d - step[1]
        if not np.isfinite(a_new) or not np.isfinite(d_new):
            break
        move = max(abs(a_new - a), abs(d_new - d))
        if move > 1.0:  # damping
            scale = 1.0 / move
            a_new, d_new = a + (a_new - a) * scale, d + (d_new - d) * scale
        a, d = a_new, d_new
        if move < 1e-8:
            break
    return a, d


def fit(
    successes: np.ndarray,
    trials: np.ndarray,
    items: Sequence[str],
    subjects: Sequence[str],
    *,
    complexity: dict[str, str] | None = None,
    seeds: float | None = None,
    max_iter: int = 1500,
    tol: float = 1e-6,
    normalize: bool = True,
) -> IRTModel:
    """Fit a 2PL model by marginal maximum likelihood (EM over quadrature).

    ``successes[i, j]`` is how many seeds of agent ``i`` medalled on competition
    ``j``, out of ``trials[i, j]``. Ability is integrated out against a standard
    normal prior, which both identifies the scale and avoids the bias that joint
    estimation of ability and item parameters carries.

    ``normalize`` rescales the fitted solution so ability has mean 0 and SD 1
    **across the agents actually measured**, carrying ``a`` and ``b`` along. A
    2PL is only identified up to a linear rescaling of ability, and with 75
    items the likelihood overwhelms the ``N(0, 1)`` prior that nominally fixes
    it -- on the published data the fitted ability SD comes out at 2.5, so the
    unnormalised scale is set by how hard the priors happen to pull against each
    other. After normalising, one unit means one standard deviation of published
    agent ability, which is a quantity a reader can hold.
    """
    S = np.asarray(successes, dtype=float)
    N = np.asarray(trials, dtype=float)
    if S.shape != N.shape:
        raise ValueError("successes and trials must have the same shape")
    if S.shape != (len(subjects), len(items)):
        raise ValueError("matrix shape must be (len(subjects), len(items))")
    if S.ndim != 2 or S.shape[0] < 2 or S.shape[1] < 2:
        raise ValueError("need at least 2 subjects and 2 items")
    if (N <= 0).any():
        raise ValueError("every cell needs at least one trial")
    if (S < 0).any() or (S > N).any():
        raise ValueError("successes must lie in [0, trials]")

    nodes, weights = _quadrature()
    n_items = S.shape[1]
    a = np.full(n_items, PRIOR_A_MEAN)
    # Start difficulty at the observed logit, which is where it would be if every
    # agent were average. Starting everything at zero makes EM crawl.
    rate = np.clip(S.sum(axis=0) / N.sum(axis=0), 1e-3, 1 - 1e-3)
    d = np.log(rate / (1 - rate))

    log_lik = -np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        log_p, log_q = _log_p(a, d, nodes)             # (items, nodes)
        # (subjects, nodes): the log-likelihood of each agent at each ability.
        ll_iq = S @ log_p + (N - S) @ log_q
        shift = ll_iq.max(axis=1, keepdims=True)
        unnorm = np.exp(ll_iq - shift) * weights[None, :]
        denom = unnorm.sum(axis=1, keepdims=True)
        post = unnorm / denom
        new_ll = float(np.sum(np.log(denom) + shift))

        r = S.T @ post          # (items, nodes) expected successes
        f = N.T @ post          # (items, nodes) expected trials
        prev = np.concatenate([a, d])
        for j in range(n_items):
            a[j], d[j] = _m_step_item(r[j], f[j], nodes, a[j], d[j])
        if abs(new_ll - log_lik) < tol and np.max(np.abs(np.concatenate([a, d]) - prev)) < tol:
            log_lik = new_ll
            converged = True
            break
        log_lik = new_ll

    if not np.isfinite(log_lik):
        raise FitFailed("EM produced a non-finite log-likelihood")

    log_p, log_q = _log_p(a, d, nodes)
    ll_iq = S @ log_p + (N - S) @ log_q
    shift = ll_iq.max(axis=1, keepdims=True)
    unnorm = np.exp(ll_iq - shift) * weights[None, :]
    post = unnorm / unnorm.sum(axis=1, keepdims=True)
    theta = post @ nodes
    theta_sd = np.sqrt(np.maximum(post @ (nodes**2) - theta**2, 0.0))

    if normalize and len(theta) > 1:
        # A 2PL is identified only up to theta -> (theta - mu) / sigma. Writing
        # the response as logit = a*theta + d, the substitution theta = sigma*t + mu
        # gives logit = (a*sigma)*t + (a*mu + d), so a and d move together and
        # b = -d/a lands exactly at (b - mu) / sigma. Information scales by
        # sigma^2 and every curve follows, so nothing below needs to know.
        mu = float(theta.mean())
        sigma = float(theta.std(ddof=1))
        if sigma > 0:
            d = a * mu + d          # must use the old a
            a = a * sigma
            theta = (theta - mu) / sigma
            theta_sd = theta_sd / sigma

    obs = S.sum(axis=0) / N.sum(axis=0)
    # An item every agent got right, or every agent got wrong, has no interior
    # information: its difficulty is wherever the prior stops, not a measurement.
    unvarying = (S.sum(axis=0) == 0) | (S.sum(axis=0) == N.sum(axis=0))
    # A difficulty outside the ability actually measured, plus a margin, is read
    # off the flat tail of the response curve. The margin is one ability SD,
    # taken from the data rather than written as 1.0: that happens to be correct
    # after normalising and is an arbitrary number of units without it.
    spread = float(theta.std(ddof=1)) if len(theta) > 1 else 1.0
    lo, hi = float(theta.min()) - spread, float(theta.max()) + spread
    # Scale-free: a slope is only steep relative to the ability range it spans.
    steep = np.abs(a) * spread >= MIN_IDENTIFIED_A
    b_hat = np.where(steep, -d / np.where(a == 0, 1e-12, a), np.inf)
    params = tuple(
        ItemParameters(
            item=name,
            a=float(a[j]),
            b=float(b_hat[j]),
            identified=bool(not unvarying[j] and steep[j]),
            observed_rate=float(obs[j]),
            extrapolated=bool(not (lo <= b_hat[j] <= hi)),
            complexity=(complexity or {}).get(name),
        )
        for j, name in enumerate(items)
    )
    return IRTModel(
        items=params,
        subjects=tuple(subjects),
        ability=tuple(float(v) for v in theta),
        ability_se=tuple(float(v) for v in theta_sd),
        log_likelihood=log_lik,
        n_iterations=it,
        converged=converged,
        seeds=float(seeds if seeds is not None else np.median(N)),
    )


# --- how much to trust the slopes -----------------------------------------


@dataclass(frozen=True)
class Attenuation:
    """How much this design shrinks estimated discrimination toward the prior."""

    slope: float
    slope_sd: float
    ability_recovery: float
    difficulty_recovery: float
    n_draws: int

    def __str__(self) -> str:
        return (
            f"discrimination slope {self.slope:.3f} +- {self.slope_sd:.3f}; "
            f"ability recovered at r={self.ability_recovery:.3f}, "
            f"difficulty at r={self.difficulty_recovery:.3f} ({self.n_draws} draws)"
        )


def calibrate_attenuation(
    model: IRTModel,
    trials: np.ndarray,
    *,
    n_draws: int = 24,
    rng: np.random.Generator | None = None,
) -> Attenuation:
    """Measure this design's own bias by refitting data it generated itself.

    Marginal maximum likelihood is consistent but not unbiased, and with 25
    agents the estimated discriminations come back systematically shallow. That
    matters more than it sounds: information goes as the *square* of
    discrimination, so a 30% attenuation understates the benchmark's sharpness
    by half, and every standard error downstream is too wide.

    The calibration is a parametric bootstrap. Data is generated from the fitted
    model's own parameters at exactly the real trial counts, the model is
    refitted on it, and the estimated discriminations are regressed on the ones
    that generated them. The slope is this design's attenuation. Ability and
    difficulty recovery come along free, and are the numbers that say which of
    the model's outputs can be leaned on and which cannot.

    Self-consistent rather than circular: the question asked is not "are the
    fitted parameters right" but "if they were, would this design recover them",
    which is answerable without knowing the truth.
    """
    rng = rng or np.random.default_rng(0)
    N = np.asarray(trials, dtype=float)
    a_true = np.array([i.a for i in model.items])
    b_true = np.array([i.b for i in model.items])
    theta_true = np.array(model.ability)
    usable = np.isfinite(b_true) & np.array([i.identified for i in model.items])
    if usable.sum() < 3:
        raise ValueError("too few identified items to calibrate against")
    P = _sigmoid(a_true[None, :] * (theta_true[:, None] - b_true[None, :]))
    P = np.nan_to_num(P, nan=0.0, posinf=1.0, neginf=0.0)
    names = [i.item for i in model.items]

    slopes, th_r, b_r = [], [], []
    for _ in range(n_draws):
        S = rng.binomial(np.rint(N).astype(int), P).astype(float)
        try:
            refit = fit(S, N, names, model.subjects, normalize=True)
        except (FitFailed, ValueError):
            continue
        a_hat = np.array([i.a for i in refit.items])
        b_hat = np.array([i.b for i in refit.items])
        ok = usable & np.isfinite(b_hat) & np.isfinite(a_hat)
        if ok.sum() < 3:
            continue
        slopes.append(float(np.polyfit(a_true[ok], a_hat[ok], 1)[0]))
        th_r.append(float(np.corrcoef(theta_true, np.array(refit.ability))[0, 1]))
        b_r.append(float(np.corrcoef(b_true[ok], b_hat[ok])[0, 1]))
    if not slopes:
        raise FitFailed("no bootstrap draw produced a usable refit")
    return Attenuation(
        slope=float(np.mean(slopes)),
        slope_sd=float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
        ability_recovery=float(np.mean(th_r)),
        difficulty_recovery=float(np.mean(b_r)),
        n_draws=len(slopes),
    )


# --- is the model worth its parameters? -----------------------------------


def _fit_fixed_a(S, N, items, subjects, **kw) -> tuple[np.ndarray, np.ndarray]:
    """A Rasch (1PL) fit: every item equally discriminating.

    Returns ``(a, d)`` rather than a model, because a Rasch fit is only wanted
    here for the held-out comparison that decides whether the second parameter
    earns its keep -- nothing downstream needs information curves from it.

    Discrimination is pinned at the mean of the free fit, which is the standard
    way to put a Rasch solution on a scale comparable to the 2PL's rather than
    on its own arbitrary one.
    """
    model = fit(S, N, items, subjects, **kw)
    mean_a = float(np.mean([i.a for i in model.items]))
    nodes, weights = _quadrature()
    a = np.full(len(items), mean_a)
    rate = np.clip(np.asarray(S).sum(0) / np.asarray(N).sum(0), 1e-3, 1 - 1e-3)
    d = np.log(rate / (1 - rate))
    for _ in range(200):
        log_p, log_q = _log_p(a, d, nodes)
        ll_iq = np.asarray(S) @ log_p + (np.asarray(N) - np.asarray(S)) @ log_q
        shift = ll_iq.max(axis=1, keepdims=True)
        unnorm = np.exp(ll_iq - shift) * weights[None, :]
        post = unnorm / unnorm.sum(axis=1, keepdims=True)
        r, f = np.asarray(S).T @ post, np.asarray(N).T @ post
        prev = d.copy()
        p = _sigmoid(np.outer(a, nodes) + d[:, None])
        grad = (r - f * p).sum(axis=1) - d / PRIOR_B_SD**2
        hess = -(f * p * (1 - p)).sum(axis=1) - 1.0 / PRIOR_B_SD**2
        d = d - grad / hess
        if np.max(np.abs(d - prev)) < 1e-7:
            break
    return a, d


def held_out_loglik(
    successes: np.ndarray,
    trials: np.ndarray,
    items: Sequence[str],
    subjects: Sequence[str],
    *,
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Predictive log-likelihood per held-out trial, for 2PL, Rasch and item-mean.

    A model with more parameters always fits better in sample, so an in-sample
    improvement is not evidence of anything. Cells are held out at random, the
    model is refitted on the rest, and each approach is scored on the trials it
    never saw. The item-mean baseline is what classical test theory implicitly
    assumes -- every agent is the average agent -- and a model that cannot beat
    it has not earned its complexity.
    """
    S = np.asarray(successes, dtype=float)
    N = np.asarray(trials, dtype=float)
    rng = rng or np.random.default_rng(0)
    cells = np.array([(i, j) for i in range(S.shape[0]) for j in range(S.shape[1])])
    order = rng.permutation(len(cells))
    folds = np.array_split(order, n_folds)
    totals = {"2pl": 0.0, "rasch": 0.0, "item_mean": 0.0}
    n_trials = 0.0

    for fold in folds:
        held = cells[fold]
        mask = np.ones_like(N, dtype=bool)
        mask[held[:, 0], held[:, 1]] = False
        # A held-out cell is made weightless rather than removed, so the matrix
        # stays rectangular and the fit sees the same items and agents.
        S_tr = np.where(mask, S, 0.0)
        N_tr = np.where(mask, N, 1e-9)
        model = fit(S_tr, N_tr, items, subjects)
        theta = np.array(model.ability)
        a = np.array([i.a for i in model.items])
        d = np.array([-i.a * i.b if np.isfinite(i.b) else 0.0 for i in model.items])

        a_r, d_r = _fit_fixed_a(S_tr, N_tr, items, subjects)
        # Item mean from the training cells only.
        tr_rate = np.clip(
            (S * mask).sum(axis=0) / np.maximum((N * mask).sum(axis=0), 1e-9),
            1e-4, 1 - 1e-4,
        )
        for i, j in held:
            s, n = S[i, j], N[i, j]
            n_trials += n
            for key, p in (
                ("2pl", _sigmoid(np.array([a[j] * theta[i] + d[j]]))[0]),
                ("rasch", _sigmoid(np.array([a_r[j] * theta[i] + d_r[j]]))[0]),
                ("item_mean", tr_rate[j]),
            ):
                p = float(np.clip(p, 1e-6, 1 - 1e-6))
                totals[key] += s * np.log(p) + (n - s) * np.log(1 - p)
    return {k: v / n_trials for k, v in totals.items()}


__all__ = [
    "Attenuation",
    "FitFailed",
    "IRTModel",
    "ItemParameters",
    "MIN_IDENTIFIED_A",
    "calibrate_attenuation",
    "fit",
    "held_out_loglik",
]
