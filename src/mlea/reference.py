"""Observed per-competition medal rates from published MLE-bench runs.

Power calculations used to assume a Beta distribution over per-competition medal
probabilities, with a concentration picked by judgement. That assumption was
wrong by a factor of about four, and wrong in the optimistic direction.

Upstream ``openai/mle-bench`` ships raw per-seed grading reports under ``runs/``.
They are git-LFS tracked, so they are invisible to ordinary raw fetches and easy
to miss. Aggregated per competition they give the real distribution, and it is
strongly U-shaped rather than Beta-ish: for o1-preview with AIDE, 42 of 75
competitions were never medalled in 21 seeds and 2 were medalled every time.

Resampling these observed rates removes the parametric assumption from the power
model entirely. The Beta path is kept for hypothetical designs at rates nobody
has published.

Provenance: aggregated from ``runs/*/`` grading reports in openai/mle-bench.
Verified by reproducing the paper's headline 16.9% for ``models-o1-preview-aide``
(this file gives 0.1698).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

DATA_PATH = Path(__file__).parent / "data" / "mlebench_per_competition_medals.csv"
#: Upstream's complexity tiers (``experiments/splits/{low,medium,high}.txt``),
#: defined by how long an experienced ML engineer would need: under 2 hours,
#: 2-10 hours, over 10 hours. The low tier *is* MLE-bench Lite.
COMPLEXITY_PATH = Path(__file__).parent / "data" / "mlebench_complexity.csv"

#: The 22 low-complexity competitions (upstream ``experiments/splits/low.txt``).
LITE_COMPETITIONS: frozenset[str] = frozenset({
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
})

#: Reference experiments worth using as a baseline arm, with why.
#: Seed counts matter: a rate measured over 3 seeds is mostly binomial noise.
REFERENCE_EXPERIMENTS = {
    "models-o1-preview-aide": "AIDE + o1-preview, ~21 seeds. The paper's headline run.",
    "scaffolding-gpt4o-aide": "AIDE + GPT-4o, ~39 seeds. The most seeds available.",
    "aira-dojo": "AIRA, ~20 seeds. A stronger mid-era agent.",
    "pievolve": "~6 seeds, 80% mean rate. Closest to current SOTA territory.",
    "famou-agent": "~9 seeds, 56% mean rate.",
}


class UnknownExperiment(KeyError):
    """Raised for a reference experiment that is not in the shipped data."""


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, tuple[int, int]]]:
    table: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    with DATA_PATH.open() as fh:
        for row in csv.DictReader(fh):
            table[row["experiment"]][row["competition_id"]] = (
                int(row["medals"]),
                int(row["seeds"]),
            )
    return dict(table)


def experiments() -> list[str]:
    return sorted(_load())


def reference_rates(
    experiment: str, *, competitions: frozenset[str] | None = None
) -> tuple[float, ...]:
    """Observed per-competition medal rates, as a pool to resample from.

    ``competitions`` restricts to a split -- pass :data:`LITE_COMPETITIONS` for
    the 22-competition lite set.
    """
    table = _load()
    if experiment not in table:
        raise UnknownExperiment(
            f"{experiment!r} is not in the shipped data. Available: "
            f"{', '.join(experiments())}"
        )
    rows = table[experiment]
    if competitions is not None:
        rows = {c: v for c, v in rows.items() if c in competitions}
    if not rows:
        raise UnknownExperiment(
            f"{experiment!r} has no competitions in the requested split"
        )
    return tuple(k / s for k, s in rows.values())



@lru_cache(maxsize=1)
def complexity() -> dict[str, str]:
    """Competition id -> upstream complexity tier."""
    with COMPLEXITY_PATH.open() as fh:
        return {r["competition_id"]: r["complexity"] for r in csv.DictReader(fh)}


def complete_experiments(n_competitions: int = 75) -> list[str]:
    """Experiments that reported every competition.

    An experiment missing competitions cannot go in a shared matrix without
    imputing the gaps, and imputing a medal rate is inventing a result. Several
    published runs cover only a subset (``pievolve`` reports 53 of 75), so the
    matrix is built from the complete ones and the rest are named, not dropped
    silently.
    """
    return sorted(e for e, d in _load().items() if len(d) == n_competitions)


def matrix(
    *,
    competitions: frozenset[str] | None = None,
    experiments_: Sequence[str] | None = None,
):
    """The published results as a subject-by-item matrix.

    Returns ``(rates, seeds, experiment_names, competition_ids)`` where
    ``rates[i][j]`` is experiment ``i``'s medal rate on competition ``j`` and
    ``seeds[i][j]`` is how many runs that rate was measured over. This is the
    input :mod:`mlea.instrument` needs to treat MLE-bench as an instrument and
    ask what it can and cannot resolve.
    """
    table = _load()
    names = list(experiments_) if experiments_ is not None else complete_experiments()
    missing = [e for e in names if e not in table]
    if missing:
        raise UnknownExperiment(f"not in the shipped data: {', '.join(sorted(missing))}")
    if not names:
        raise UnknownExperiment("no experiments selected")
    common = set.intersection(*(set(table[e]) for e in names))
    if competitions is not None:
        common &= set(competitions)
    comps = sorted(common)
    if not comps:
        raise UnknownExperiment("the selected experiments share no competitions")
    rates = [[table[e][c][0] / table[e][c][1] for c in comps] for e in names]
    seeds = [[table[e][c][1] for c in comps] for e in names]
    return rates, seeds, names, comps


def summarise(experiment: str, *, competitions: frozenset[str] | None = None) -> dict:
    """Shape of a reference pool: mean, and how much sits at the extremes."""
    rates = reference_rates(experiment, competitions=competitions)
    n = len(rates)
    return {
        "n_competitions": n,
        "mean": sum(rates) / n,
        "never": sum(1 for r in rates if r == 0.0),
        "always": sum(1 for r in rates if r == 1.0),
    }


__all__ = [
    "COMPLEXITY_PATH",
    "DATA_PATH",
    "LITE_COMPETITIONS",
    "REFERENCE_EXPERIMENTS",
    "UnknownExperiment",
    "complete_experiments",
    "complexity",
    "experiments",
    "matrix",
    "reference_rates",
    "summarise",
]
