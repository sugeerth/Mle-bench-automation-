"""Run records and the comparability guard.

The guard exists because the public MLE-bench literature routinely compares
numbers that are not comparable: 65.3% (full 75), 68.2% (lite 22) and 85.71%
(a selected lite subset) all get quoted as "SOTA on MLE-bench". A split
identifier is part of a result's identity, not metadata, so this module makes
an incomparable comparison raise rather than silently return a number.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class IncomparableError(ValueError):
    """Raised when two run sets must not be compared."""


@dataclass(frozen=True)
class Fingerprint:
    """Identity of the conditions a sweep ran under.

    Two run sets may only be compared when their fingerprints agree. Reduced
    hardware, a different competition split, or a different harness version all
    produce numbers that look comparable and are not.
    """

    split_id: str
    container_config: str = "default"
    harness_version: str = "unknown"

    def conflicts_with(self, other: "Fingerprint") -> list[str]:
        reasons = []
        if self.split_id != other.split_id:
            reasons.append(f"split_id: {self.split_id!r} vs {other.split_id!r}")
        if self.container_config != other.container_config:
            reasons.append(
                f"container_config: {self.container_config!r} vs {other.container_config!r}"
            )
        if self.harness_version != other.harness_version:
            reasons.append(
                f"harness_version: {self.harness_version!r} vs {other.harness_version!r}"
            )
        return reasons


@dataclass(frozen=True)
class RunRecord:
    """One agent run on one competition with one seed."""

    competition_id: str
    seed: int
    medal: bool
    #: Set for runs that never produced a gradeable result. These are excluded
    #: from capability metrics -- an infra failure is our fault, and counting it
    #: as an agent failure understates the agent.
    infra_failure: bool = False

    def __post_init__(self) -> None:
        if self.infra_failure and self.medal:
            raise ValueError(
                f"{self.competition_id} seed {self.seed}: a run cannot both be an "
                "infra failure and have earned a medal"
            )


@dataclass
class RunSet:
    """All runs for one condition (an agent, a model, a cutoff era)."""

    label: str
    fingerprint: Fingerprint
    runs: list[RunRecord] = field(default_factory=list)

    @property
    def gradeable(self) -> list[RunRecord]:
        return [r for r in self.runs if not r.infra_failure]

    @property
    def n_infra_failures(self) -> int:
        return sum(1 for r in self.runs if r.infra_failure)

    def competitions(self) -> set[str]:
        return {r.competition_id for r in self.gradeable}

    def medal_rate_by_competition(self) -> dict[str, float]:
        """Mean medal rate per competition, over every run recorded for it.

        Deliberately averages *all* matching runs rather than taking the latest.
        Re-running a competition until it medals is the most natural and most
        corrupting thing a person can do with a benchmark harness; averaging
        makes it structurally ineffective instead of relying on discipline.
        """
        buckets: dict[str, list[bool]] = defaultdict(list)
        for r in self.gradeable:
            buckets[r.competition_id].append(r.medal)
        return {c: sum(v) / len(v) for c, v in buckets.items()}

    def seeds_by_competition(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for r in self.gradeable:
            counts[r.competition_id] += 1
        return dict(counts)

    def any_medal_rate(self) -> float:
        """Headline metric: mean over competitions of the per-competition rate.

        Averaged per competition, not per run, so a competition that happened to
        get extra seeds does not gain extra weight in the headline number.
        """
        per_comp = self.medal_rate_by_competition()
        if not per_comp:
            raise ValueError(f"run set {self.label!r} has no gradeable runs")
        return sum(per_comp.values()) / len(per_comp)

    @classmethod
    def from_records(
        cls,
        label: str,
        fingerprint: Fingerprint,
        records: Iterable[Mapping[str, object]],
    ) -> "RunSet":
        runs = [
            RunRecord(
                competition_id=str(rec["competition_id"]),
                seed=int(rec.get("seed", 0)),  # type: ignore[arg-type]
                medal=bool(rec.get("any_medal", rec.get("medal", False))),
                infra_failure=bool(rec.get("infra_failure", False)),
            )
            for rec in records
        ]
        return cls(label=label, fingerprint=fingerprint, runs=runs)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunSet":
        """Load a run set from JSON.

        Expected shape::

            {"label": "...",
             "fingerprint": {"split_id": "low", ...},
             "runs": [{"competition_id": "...", "seed": 0, "any_medal": true}, ...]}
        """
        blob = json.loads(Path(path).read_text())
        fp = blob.get("fingerprint", {})
        if "split_id" not in fp:
            raise IncomparableError(
                f"{path}: fingerprint.split_id is required. A result without a split "
                "identifier cannot be compared to anything."
            )
        return cls.from_records(
            label=str(blob.get("label", Path(path).stem)),
            fingerprint=Fingerprint(
                split_id=str(fp["split_id"]),
                container_config=str(fp.get("container_config", "default")),
                harness_version=str(fp.get("harness_version", "unknown")),
            ),
            records=blob["runs"],
        )


def assert_comparable(
    a: RunSet,
    b: RunSet,
    *,
    allow_fingerprint_mismatch: bool = False,
) -> None:
    """Raise unless ``a`` and ``b`` may legitimately be compared.

    ``allow_fingerprint_mismatch`` is deliberately awkward to reach: it exists
    for the matched pre/post-cutoff design, where the two arms are *supposed* to
    be different competition sets, and nowhere else.
    """
    reasons = a.fingerprint.conflicts_with(b.fingerprint)
    if reasons and not allow_fingerprint_mismatch:
        raise IncomparableError(
            f"refusing to compare {a.label!r} against {b.label!r}: "
            + "; ".join(reasons)
            + ". These results were produced under different conditions and any "
            "difference between them confounds the change with the conditions. "
            "Pass allow_fingerprint_mismatch=True only if you know the design "
            "intends it (e.g. a matched pre/post-cutoff comparison)."
        )


def common_competitions(a: RunSet, b: RunSet) -> list[str]:
    shared = sorted(a.competitions() & b.competitions())
    if not shared:
        raise IncomparableError(
            f"{a.label!r} and {b.label!r} share no competitions with gradeable runs"
        )
    return shared


def load_pairs(path: str | Path) -> list[tuple[str, str]]:
    """Load matched competition pairs for the pre/post-cutoff design.

    JSON: ``[["pre-comp-id", "post-comp-id"], ...]``
    """
    blob = json.loads(Path(path).read_text())
    pairs = [(str(x), str(y)) for x, y in blob]
    if not pairs:
        raise ValueError(f"{path}: no pairs")
    return pairs


__all__ = [
    "Fingerprint",
    "IncomparableError",
    "RunRecord",
    "RunSet",
    "assert_comparable",
    "common_competitions",
    "load_pairs",
]
