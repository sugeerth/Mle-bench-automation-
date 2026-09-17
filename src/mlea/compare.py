"""Paired comparison of two MLE-bench run sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .records import RunSet, assert_comparable, common_competitions
from .stats import (
    cluster_bootstrap_ci,
    permutation_pvalues,
    sign_matrix,
    smallest_detectable_pvalue,
)


@dataclass(frozen=True)
class Comparison:
    """Result of comparing two run sets over shared (or matched) units."""

    label_a: str
    label_b: str
    n_units: int
    rate_a: float
    rate_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_value_floor: float
    alpha: float
    min_seeds: int
    unit_diffs: tuple[float, ...]
    matched: bool
    warnings: tuple[str, ...] = ()

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def underpowered_by_construction(self) -> bool:
        """True when the design cannot reach significance at any effect size."""
        return self.p_value_floor >= self.alpha

    def summary(self) -> str:
        kind = "matched pairs" if self.matched else "shared competitions"
        lines = [
            f"{self.label_b} vs {self.label_a}",
            f"  units          : {self.n_units} {kind}, min {self.min_seeds} seed(s) each",
            f"  {self.label_a:<14.14}: {self.rate_a:6.1%}",
            f"  {self.label_b:<14.14}: {self.rate_b:6.1%}",
            f"  difference     : {self.difference:+6.1%} "
            f"(95% CI {self.ci_low:+.1%} to {self.ci_high:+.1%})",
            f"  p (paired perm): {self.p_value:.4f}"
            + ("  [SIGNIFICANT]" if self.significant else "  [not significant]"),
        ]
        if self.underpowered_by_construction:
            lines.append(
                f"  !! this design cannot produce p < {self.alpha} at any effect size "
                f"(floor p = {self.p_value_floor:.4f}). Add units before drawing "
                f"conclusions."
            )
        for w in self.warnings:
            lines.append(f"  !! {w}")
        return "\n".join(lines)


def _rates(runset: RunSet, units: Sequence[str]) -> np.ndarray:
    per_comp = runset.medal_rate_by_competition()
    return np.array([per_comp[u] for u in units], dtype=float)


def compare(
    a: RunSet,
    b: RunSet,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    n_perm: int = 9_999,
    seed: int = 0,
) -> Comparison:
    """Compare run set ``b`` against baseline ``a``.

    By default the two arms must share a fingerprint and are paired on
    competition id -- competition difficulty is the dominant variance term on
    this benchmark, and pairing removes it.

    ``pairs`` switches to the matched design used for the pre/post-cutoff
    contamination experiment, where the two arms are *different* competitions
    deliberately matched on difficulty proxies. That design permits a
    fingerprint mismatch (the split ids differ by construction) but is
    correspondingly weaker: matching quality, not competition identity, is what
    removes variance.
    """
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    if pairs is None:
        assert_comparable(a, b)
        units = common_competitions(a, b)
        rates_a = _rates(a, units)
        rates_b = _rates(b, units)
        only_a = a.competitions() - set(units)
        only_b = b.competitions() - set(units)
        if only_a or only_b:
            warnings.append(
                f"{len(only_a)} competition(s) only in {a.label!r} and "
                f"{len(only_b)} only in {b.label!r} were dropped; the comparison "
                f"covers {len(units)} of {len(a.competitions() | b.competitions())}"
            )
        seed_counts = list(a.seeds_by_competition().values()) + list(
            b.seeds_by_competition().values()
        )
        matched = False
    else:
        assert_comparable(a, b, allow_fingerprint_mismatch=True)
        rates_a_map = a.medal_rate_by_competition()
        rates_b_map = b.medal_rate_by_competition()
        missing = [
            p for p in pairs if p[0] not in rates_a_map or p[1] not in rates_b_map
        ]
        if missing:
            raise ValueError(
                f"{len(missing)} matched pair(s) have no gradeable runs on one side, "
                f"first: {missing[0]}"
            )
        units = [f"{x}|{y}" for x, y in pairs]
        rates_a = np.array([rates_a_map[x] for x, _ in pairs], dtype=float)
        rates_b = np.array([rates_b_map[y] for _, y in pairs], dtype=float)
        seed_counts = [a.seeds_by_competition()[x] for x, _ in pairs] + [
            b.seeds_by_competition()[y] for _, y in pairs
        ]
        matched = True
        warnings.append(
            "matched-pair design: the difference confounds the intended effect "
            "with any residual difficulty mismatch between paired competitions. "
            "Report the matching covariates alongside this number."
        )

    if a.n_infra_failures or b.n_infra_failures:
        warnings.append(
            f"excluded {a.n_infra_failures + b.n_infra_failures} infra failure(s) "
            "from both arms (harness faults are not agent capability)"
        )

    diffs = rates_b - rates_a
    n_units = diffs.size

    signs = sign_matrix(n_units, n_perm, rng)
    p_value = float(permutation_pvalues(diffs[None, :], signs)[0])
    ci_low, ci_high = cluster_bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, rng=rng)

    min_seeds = min(seed_counts) if seed_counts else 0
    if min_seeds < 3:
        warnings.append(
            f"minimum {min_seeds} seed(s) per competition; MLE-bench convention is "
            "at least 3, and fewer inflates within-competition noise"
        )

    return Comparison(
        label_a=a.label,
        label_b=b.label,
        n_units=n_units,
        rate_a=float(rates_a.mean()),
        rate_b=float(rates_b.mean()),
        difference=float(diffs.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        p_value_floor=smallest_detectable_pvalue(n_units, n_perm),
        alpha=alpha,
        min_seeds=min_seeds,
        unit_diffs=tuple(float(d) for d in diffs),
        matched=matched,
        warnings=tuple(warnings),
    )


__all__ = ["Comparison", "compare"]
