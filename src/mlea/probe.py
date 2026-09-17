"""A contamination probe with a positive control.

The finding this exists to address: MLE-bench's 2024 contamination check
returned a null (8.5% vs 8.4%), and it was never established that the check
*could* have returned anything else. It rewrote competition **descriptions**;
it did not alter the data. A model recognising a dataset by its contents would
sail straight through it.

That question cannot be settled on real competitions, because you cannot match
two Kaggle competitions on difficulty and any systematic difficulty difference
is indistinguishable from memorisation. On generated competitions it is settled
by construction: a clone *is* the same problem, and the residual difficulty
difference is measured rather than assumed.

So this runs a two-by-two: a simulated memoriser and an honest solver, each on
an original and on a clone. The honest solver is the negative control -- its gap
must be ~0, or the clone is not really the same problem. The memoriser is the
positive control -- its gap must be large, or the probe cannot see memorisation
at all and a null result from it means nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProbeCell:
    """One agent on one competition."""

    agent: str
    competition_id: str
    is_clone: bool
    transform: str | None
    score: float | None
    #: The score as a leaderboard percentile. Raw scores are not comparable
    #: across competitions -- an AUC gap and an RMSE gap are different units
    #: pointing opposite ways -- so every aggregate below uses this.
    percentile: float | None
    #: Did it earn any medal? The unit MLE-bench actually reports.
    any_medal: bool
    valid: bool
    recalled: bool


@dataclass
class ProbeResult:
    """The two-by-two, and what it licenses you to conclude."""

    cells: list[ProbeCell]

    def gap(self, agent: str, transform: str) -> float | None:
        """Original minus clone, in leaderboard percentile points.

        Positive means the agent did better on what it had seen -- the signature
        of recall rather than capability. Percentile units are used so the
        average is meaningful across competitions with different metrics.
        """
        orig = [
            c
            for c in self.cells
            if c.agent == agent and not c.is_clone and c.valid and c.percentile is not None
        ]
        clone = [
            c
            for c in self.cells
            if c.agent == agent
            and c.is_clone
            and c.transform == transform
            and c.valid
            and c.percentile is not None
        ]
        if not orig or not clone:
            return None
        by_source = {c.competition_id: c.percentile for c in orig}
        diffs = [
            by_source[c.competition_id.rsplit("__", 1)[0]] - c.percentile
            for c in clone
            if c.competition_id.rsplit("__", 1)[0] in by_source
        ]
        return sum(diffs) / len(diffs) if diffs else None

    def medal_gap(self, agent: str, transform: str) -> float | None:
        """Medal-rate difference, original minus clone.

        Reported alongside the percentile gap because percentile **compresses
        at the top**: an agent that scores perfectly and one that scores merely
        well both sit in the leaderboard's top few percent, so a large real
        advantage shrinks to a few percentile points. Medal rate is the unit
        MLE-bench reports, and it does not compress the same way.
        """
        orig = [c for c in self.cells if c.agent == agent and not c.is_clone]
        clone = [
            c for c in self.cells
            if c.agent == agent and c.is_clone and c.transform == transform
        ]
        if not orig or not clone:
            return None
        by_source = {c.competition_id: c.any_medal for c in orig}
        diffs = [
            float(by_source[c.competition_id.rsplit("__", 1)[0]]) - float(c.any_medal)
            for c in clone
            if c.competition_id.rsplit("__", 1)[0] in by_source
        ]
        return sum(diffs) / len(diffs) if diffs else None

    def recall_rate(self, agent: str, transform: str | None) -> float:
        rows = [
            c
            for c in self.cells
            if c.agent == agent
            and (c.transform == transform if c.is_clone else transform is None)
        ]
        return sum(1 for c in rows if c.recalled) / len(rows) if rows else 0.0

    def to_dict(self) -> dict:
        return {"cells": [c.__dict__ for c in self.cells]}


def load(path: str | Path) -> ProbeResult:
    blob = json.loads(Path(path).read_text())
    return ProbeResult([ProbeCell(**c) for c in blob["cells"]])


__all__ = ["ProbeCell", "ProbeResult", "load"]
