"""Skill profiling: which ML competence does an agent actually have?

MLE-bench reports one number. At 65% and rising it is close to saturating, and
a single number cannot say *what* an agent is missing -- which is what anyone
improving an agent needs to know.

A generated benchmark can do better, because a challenged competition and an
otherwise identical clean control can be produced from the same seed and the
same latent function. The pair differs by exactly one pathology, so the score
difference isolates one skill instead of reporting an aggregate that hides which
one is absent. That pairing is impossible on real competitions -- it is the same
difficulty-matching problem that made a rolling Kaggle split unworkable.

Measured in leaderboard percentile, never medal rate: medal rate saturates, and
a competence gap that does not flip a medal is invisible to it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .stats import (
    cluster_bootstrap_ci,
    permutation_pvalues,
    sign_matrix,
    smallest_detectable_pvalue,
)


#: Smallest effect worth calling a weakness, in percentile points.
#:
#: Statistical significance is not practical significance. A perfectly
#: consistent one-point effect is significant -- eight paired differences of the
#: same sign give p = 3/257 whatever their size -- and reporting it as a missing
#: competence would bury the real ones in trivia. A cell must clear both bars.
MIN_MEANINGFUL = 0.02


@dataclass(frozen=True)
class SkillCell:
    """One agent against one challenge, measured over several competitions.

    ``deltas`` holds one paired difference per competition: challenged minus
    control, both run on the same seed and the same latent function. A single
    pair is a point estimate with no uncertainty, and this repository has spent
    a lot of effort establishing that a benchmark difference without an interval
    is not a result. So the cell carries the whole sample and reports an
    interval and a p-value over it.
    """

    agent: str
    challenge: str
    #: One paired difference per competition, in percentile points.
    deltas: tuple[float, ...] = ()
    #: Competitions where the agent produced no gradeable submission.
    n_broken: int = 0
    #: The first such failure, for the tooltip.
    failure: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    alpha: float = 0.05

    @property
    def delta(self) -> float | None:
        """Mean paired difference. Negative means the pathology cost something."""
        return float(np.mean(self.deltas)) if self.deltas else None

    @property
    def n_pairs(self) -> int:
        return len(self.deltas)

    @property
    def broke(self) -> bool:
        """True when the agent could not be graded on *any* challenged run."""
        return self.n_pairs == 0 and self.n_broken > 0

    @property
    def partially_broke(self) -> bool:
        return self.n_broken > 0 and self.n_pairs > 0

    @property
    def significant(self) -> bool:
        """Is the effect distinguishable from zero at ``alpha``?"""
        return self.p_value is not None and self.p_value < self.alpha

    @property
    def meaningful(self) -> bool:
        """Distinguishable from zero *and* big enough to act on."""
        return (
            self.significant
            and self.delta is not None
            and abs(self.delta) >= MIN_MEANINGFUL
        )

    def summary(self) -> str:
        if self.broke:
            return f"BROKE ({self.n_broken}/{self.n_broken})"
        if self.delta is None:
            return "—"
        txt = f"{self.delta:+.0%}"
        if not self.significant:
            txt += " ns"
        elif not self.meaningful:
            txt += " ~0"
        return txt


def measure(
    agent: str,
    challenge: str,
    pairs: list[tuple[float | None, float | None, str | None]],
    *,
    alpha: float = 0.05,
    n_boot: int = 5_000,
    seed: int = 0,
) -> SkillCell:
    """Build a cell from ``(control, challenged, failure)`` per competition.

    Uses the same paired sign-flip test as :mod:`mlea.compare`, so a skill
    finding is held to the standard this repository applies to an agent
    comparison. A pair where either side could not be graded contributes to the
    break count and not to the deltas -- averaging a missing score as zero would
    manufacture an effect.
    """
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    broken = 0
    first_failure: str | None = None
    for control, challenged, failure in pairs:
        if failure is not None or control is None or challenged is None:
            broken += 1
            first_failure = first_failure or failure or "no gradeable submission"
            continue
        deltas.append(challenged - control)

    if not deltas:
        return SkillCell(agent, challenge, (), broken, first_failure, alpha=alpha)

    arr = np.asarray(deltas, dtype=float)
    lo, hi = cluster_bootstrap_ci(arr, n_boot=n_boot, alpha=alpha, rng=rng)
    p = float(
        permutation_pvalues(arr[None, :], sign_matrix(arr.size, 9_999, rng))[0]
    )
    return SkillCell(
        agent, challenge, tuple(float(d) for d in arr), broken, first_failure,
        lo, hi, p, alpha,
    )


def design_floor(n_pairs: int) -> float:
    """Smallest p-value this many paired competitions can ever produce."""
    return smallest_detectable_pvalue(n_pairs, 9_999)


@dataclass
class SkillProfile:
    cells: list[SkillCell] = field(default_factory=list)

    @property
    def agents(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.cells:
            seen.setdefault(c.agent, None)
        return list(seen)

    @property
    def challenges(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.cells:
            seen.setdefault(c.challenge, None)
        return list(seen)

    def cell(self, agent: str, challenge: str) -> SkillCell | None:
        for c in self.cells:
            if c.agent == agent and c.challenge == challenge:
                return c
        return None

    def robustness(self, agent: str) -> float:
        """Mean cost across challenges, with a break counted as a total loss.

        A single headline number *for the profile*, reported only alongside it.
        On its own it would reproduce the problem the profile exists to solve.
        """
        rows = [c for c in self.cells if c.agent == agent]
        if not rows:
            return 0.0
        costs = []
        for c in rows:
            if c.broke:
                costs.append(-1.0)
            elif c.delta is not None:
                # Only effects distinguishable from zero count against an agent.
                costs.append(min(c.delta, 0.0) if c.meaningful else 0.0)
        return sum(costs) / len(costs) if costs else 0.0

    def hardest_for(self, agent: str) -> str | None:
        """The challenge that costs this agent the most -- its weakest skill."""
        rows = [c for c in self.cells if c.agent == agent]
        if not rows:
            return None
        def cost(c: SkillCell) -> float:
            if c.broke:
                return -1.0
            if c.delta is None or not c.meaningful:
                return 0.0
            return c.delta

        worst = min(rows, key=cost)
        return worst.challenge if cost(worst) < 0 else None

    def dominant_agent(self) -> str | None:
        """An agent that is not *measurably* beaten on any challenge, if one exists.

        Deliberately strict about what counts as being beaten. An earlier version
        compared raw means and declared "no agent dominates" off differences that
        turned out, once measured over enough paired competitions, to be
        indistinguishable from zero. A lead only counts when the two cells'
        confidence intervals do not overlap.
        """
        for candidate in self.agents:
            beaten = False
            for ch in self.challenges:
                mine = self.cell(candidate, ch)
                if mine is None or mine.broke:
                    beaten = True
                    break
                for other in self.agents:
                    if other == candidate:
                        continue
                    theirs = self.cell(other, ch)
                    if theirs is None or theirs.broke or theirs.delta is None:
                        continue
                    if mine.delta is None:
                        beaten = True
                        break
                    # A real lead: their interval sits entirely above mine.
                    if (
                        theirs.ci_low is not None
                        and mine.ci_high is not None
                        and theirs.ci_low > mine.ci_high
                        and theirs.delta - mine.delta >= MIN_MEANINGFUL
                    ):
                        beaten = True
                        break
                if beaten:
                    break
            if not beaten:
                return candidate
        return None

    def no_agent_dominates(self) -> bool:
        """True when no single agent is best-or-tied on every challenge."""
        return self.dominant_agent() is None

    def to_dict(self) -> dict:
        return {"cells": [c.__dict__ for c in self.cells]}


def load(path: str | Path) -> SkillProfile:
    blob = json.loads(Path(path).read_text())
    return SkillProfile([SkillCell(**c) for c in blob["cells"]])


__all__ = [
    "MIN_MEANINGFUL",
    "SkillCell",
    "SkillProfile",
    "design_floor",
    "load",
    "measure",
]
