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


@dataclass(frozen=True)
class SkillCell:
    """One agent against one challenge, versus its matched control."""

    agent: str
    challenge: str
    control_percentile: float | None
    challenged_percentile: float | None
    #: Set when the agent failed to produce a gradeable submission at all.
    failure: str | None = None

    @property
    def delta(self) -> float | None:
        """Challenged minus control, in percentile points.

        Negative means the pathology cost the agent something -- that is the
        competence it lacks. Zero means it handled it.
        """
        if self.control_percentile is None or self.challenged_percentile is None:
            return None
        return self.challenged_percentile - self.control_percentile

    @property
    def broke(self) -> bool:
        return self.failure is not None


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
                costs.append(min(c.delta, 0.0))
        return sum(costs) / len(costs) if costs else 0.0

    def hardest_for(self, agent: str) -> str | None:
        """The challenge that costs this agent the most -- its weakest skill."""
        rows = [c for c in self.cells if c.agent == agent]
        if not rows:
            return None
        return min(
            rows, key=lambda c: (-1.0 if c.broke else (c.delta if c.delta is not None else 0.0))
        ).challenge

    def no_agent_dominates(self) -> bool:
        """True when different agents are best at different challenges.

        The argument for a profile over a score: if one agent were best
        everywhere, a single number would suffice.
        """
        winners = set()
        for ch in self.challenges:
            rows = [
                c for c in self.cells
                if c.challenge == ch and not c.broke and c.delta is not None
            ]
            if rows:
                winners.add(max(rows, key=lambda c: c.delta).agent)
        return len(winners) > 1

    def to_dict(self) -> dict:
        return {"cells": [c.__dict__ for c in self.cells]}


def load(path: str | Path) -> SkillProfile:
    blob = json.loads(Path(path).read_text())
    return SkillProfile([SkillCell(**c) for c in blob["cells"]])


__all__ = ["SkillCell", "SkillProfile", "load"]
