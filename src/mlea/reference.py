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

DATA_PATH = Path(__file__).parent / "data" / "mlebench_per_competition_medals.csv"

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
    "DATA_PATH",
    "LITE_COMPETITIONS",
    "REFERENCE_EXPERIMENTS",
    "UnknownExperiment",
    "experiments",
    "reference_rates",
    "summarise",
]
