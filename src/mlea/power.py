"""Power and minimum detectable effect for MLE-bench sweep designs.

The question this answers: *before* spending five figures on a sweep, could that
sweep have detected the effect you care about? On this benchmark the answer is
often no, because n is the number of competitions (22 on lite, 8 or so matched
pairs on a contamination experiment) and per-competition outcomes are binary.

The simulation uses the same paired sign-flip test as :mod:`mlea.compare`, so a
reported power describes the analysis that will actually be run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import permutation_pvalues, sign_matrix, smallest_detectable_pvalue


@dataclass(frozen=True)
class Design:
    """A sweep design to be evaluated for power.

    ``base_rate`` and ``heterogeneity`` describe the baseline arm as a Beta
    distribution over per-competition medal probabilities. Heterogeneity is the
    Beta concentration: low values mean competitions are near-always or
    near-never won (which is what MLE-bench actually looks like), high values
    mean they are all of similar difficulty. It is the parameter that most
    affects power, and it is an assumption -- vary it.

    ``matching_sd`` is 0 for a same-competition paired design. For the matched
    pre/post-cutoff design it is the residual sd in baseline difficulty between
    paired competitions after matching; larger means worse matching and less
    variance removed.
    """

    name: str
    n_units: int
    n_seeds: int
    base_rate: float
    heterogeneity: float = 3.0
    matching_sd: float = 0.0
    description: str = ""

    def cost_runs(self) -> int:
        """Total agent runs: both arms, every unit, every seed."""
        return 2 * self.n_units * self.n_seeds


@dataclass(frozen=True)
class PowerResult:
    design: Design
    effect: float
    power: float
    alpha: float
    n_sims: int
    p_floor: float

    @property
    def impossible(self) -> bool:
        return self.p_floor >= self.alpha

    def summary(self) -> str:
        line = (
            f"{self.design.name}: {self.design.n_units} units x "
            f"{self.design.n_seeds} seeds ({self.design.cost_runs()} runs)  "
            f"effect {self.effect:+.0%} -> power {self.power:.0%}"
        )
        if self.impossible:
            line += f"  [IMPOSSIBLE: p floor {self.p_floor:.3f} >= alpha {self.alpha}]"
        return line


def _simulate_diffs(
    design: Design, effect: float, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-unit medal-rate differences under a true ``effect``.

    Shape ``(n_sims, n_units)``. Baseline probabilities are drawn per simulation
    so the reported power averages over which competitions you happen to draw,
    rather than conditioning on one lucky set.
    """
    conc = max(design.heterogeneity, 1e-6)
    a = max(design.base_rate * conc, 1e-6)
    b = max((1.0 - design.base_rate) * conc, 1e-6)
    p_base = rng.beta(a, b, size=(n_sims, design.n_units))

    p_a = p_base
    p_b = p_base + effect
    if design.matching_sd > 0:
        # Imperfect matching: the paired unit is a *different* competition whose
        # baseline difficulty only approximately matches its partner's.
        p_b = p_b + rng.normal(0.0, design.matching_sd, size=p_base.shape)
    p_a = np.clip(p_a, 0.0, 1.0)
    p_b = np.clip(p_b, 0.0, 1.0)

    k_a = rng.binomial(design.n_seeds, p_a)
    k_b = rng.binomial(design.n_seeds, p_b)
    return (k_b - k_a) / design.n_seeds


def power_for_effect(
    design: Design,
    effect: float,
    *,
    alpha: float = 0.05,
    n_sims: int = 4_000,
    n_perm: int = 999,
    seed: int = 0,
) -> PowerResult:
    """Probability the design detects a true ``effect`` at level ``alpha``."""
    rng = np.random.default_rng(seed)
    p_floor = smallest_detectable_pvalue(design.n_units, n_perm)
    if p_floor >= alpha:
        # No relabelling can produce a significant result; power is exactly 0.
        return PowerResult(design, effect, 0.0, alpha, n_sims, p_floor)

    diffs = _simulate_diffs(design, effect, n_sims, rng)
    signs = sign_matrix(design.n_units, n_perm, rng)
    pvals = permutation_pvalues(diffs, signs)
    return PowerResult(
        design=design,
        effect=effect,
        power=float((pvals < alpha).mean()),
        alpha=alpha,
        n_sims=n_sims,
        p_floor=p_floor,
    )


def minimum_detectable_effect(
    design: Design,
    *,
    direction: str = "decrease",
    target_power: float = 0.80,
    alpha: float = 0.05,
    n_sims: int = 4_000,
    n_perm: int = 999,
    seed: int = 0,
    tolerance: float = 0.005,
    max_effect: float = 1.0,
) -> float | None:
    """Smallest effect this design detects with ``target_power``.

    ``direction`` matters and defaults to ``"decrease"``. Medal probability is
    bounded at 1, so near a high base rate an *improvement* is squashed by the
    ceiling while an equally-sized *drop* is fully visible. At the 80.3% lite
    base rate the two differ by a large factor, and both regression gating and
    the contamination gap are drops -- searching upward by default would give a
    badly pessimistic answer for the question actually being asked.

    Returns ``None`` when no effect up to ``max_effect`` reaches the target --
    the honest answer for several designs people actually run, and the reason
    this function exists.
    """
    if direction not in ("decrease", "increase"):
        raise ValueError("direction must be 'decrease' or 'increase'")
    if smallest_detectable_pvalue(design.n_units, n_perm) >= alpha:
        return None
    sign = -1.0 if direction == "decrease" else 1.0

    def power_at(magnitude: float) -> float:
        return power_for_effect(
            design,
            sign * magnitude,
            alpha=alpha,
            n_sims=n_sims,
            n_perm=n_perm,
            seed=seed,
        ).power

    lo, hi = 0.0, max_effect
    if power_at(hi) < target_power:
        return None
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        if power_at(mid) >= target_power:
            hi = mid
        else:
            lo = mid
    return sign * hi


def seeds_needed(
    design: Design,
    effect: float,
    *,
    target_power: float = 0.80,
    alpha: float = 0.05,
    max_seeds: int = 20,
    n_sims: int = 4_000,
    n_perm: int = 999,
    seed: int = 0,
) -> int | None:
    """Fewest seeds per unit that reach ``target_power`` for ``effect``.

    Returns ``None`` if ``max_seeds`` is not enough -- seeds cannot fix a design
    whose unit count is too small, because between-competition variance and the
    permutation floor both depend on units, not seeds.
    """
    from dataclasses import replace

    for s in range(1, max_seeds + 1):
        candidate = replace(design, n_seeds=s)
        if (
            power_for_effect(
                candidate, effect, alpha=alpha, n_sims=n_sims, n_perm=n_perm, seed=seed
            ).power
            >= target_power
        ):
            return s
    return None


#: Designs grounded in real numbers, for the CLI. Base rates come from MLEvolve
#: (65.3% full set, 80.3% low/lite); the SWE-bench-Live gap (~55 points) is the
#: reference effect size for the contamination experiment.
DESIGNS: dict[str, Design] = {
    "lite-regression": Design(
        name="lite-regression",
        n_units=22,
        n_seeds=3,
        base_rate=0.803,
        description="Detect an agent regression on the 22-competition lite split.",
    ),
    "full-regression": Design(
        name="full-regression",
        n_units=75,
        n_seeds=3,
        base_rate=0.653,
        description="Detect an agent regression on the full 75-competition split.",
    ),
    "live-gap": Design(
        name="live-gap",
        n_units=8,
        n_seeds=3,
        base_rate=0.803,
        matching_sd=0.15,
        description=(
            "Pre/post-cutoff contamination gap, 8 matched pairs "
            "(the design costed in PROPOSAL-mle-bench-live.md)."
        ),
    ),
    "live-gap-16": Design(
        name="live-gap-16",
        n_units=16,
        n_seeds=3,
        base_rate=0.803,
        matching_sd=0.15,
        description="Pre/post-cutoff contamination gap, 16 matched pairs.",
    ),
}


__all__ = [
    "DESIGNS",
    "Design",
    "PowerResult",
    "minimum_detectable_effect",
    "power_for_effect",
    "seeds_needed",
]
