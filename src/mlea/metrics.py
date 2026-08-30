"""Competition metrics, in numpy only.

Deliberately dependency-free beyond numpy: this has to run on a free notebook
where installing scikit-learn fights the preinstalled stack, and on a grading
box that should not need an ML environment at all.

Semantics follow upstream MLE-bench's grading contract: a metric either returns
a float or raises :class:`InvalidSubmission`, and the caller distinguishes
"there was no submission" from "there was one and it was not gradeable" from
"it was graded and scored X".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class InvalidSubmission(ValueError):
    """The submission exists but cannot be scored. Mirrors upstream's error."""


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    #: True when a larger score is a better score. Everything downstream --
    #: leaderboard sort order, medal thresholds, "did it improve" -- depends on
    #: this, and getting it backwards silently inverts a whole competition.
    greater_is_better: bool

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape[0] != y_pred.shape[0]:
            raise InvalidSubmission(
                f"expected {y_true.shape[0]} rows, got {y_pred.shape[0]}"
            )
        if y_pred.size == 0:
            raise InvalidSubmission("submission has no rows")
        if not np.all(np.isfinite(y_pred)):
            n = int((~np.isfinite(y_pred)).sum())
            raise InvalidSubmission(f"{n} non-finite prediction(s)")
        return float(self.fn(y_true, y_pred))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC with correct tie handling.

    Ties matter more than they look: a constant-prediction submission is the
    single most common degenerate agent output, and a naive implementation
    scores it 0.0 or 1.0 rather than the correct 0.5.
    """
    y_true = np.asarray(y_true).astype(float)
    pos, neg = y_true == 1, y_true == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise InvalidSubmission("AUC needs both classes present in the answers")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    # Average ranks within tied groups, else ties leak ordering information.
    sorted_scores = np.asarray(y_score)[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.round(np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if np.any(yp < -1) or np.any(yt < -1):
        raise InvalidSubmission("RMSLE is undefined for values below -1")
    return float(np.sqrt(np.mean((np.log1p(yp) - np.log1p(yt)) ** 2)))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


METRICS: dict[str, Metric] = {
    "roc_auc": Metric("roc_auc", roc_auc, True),
    "accuracy": Metric("accuracy", accuracy, True),
    "rmse": Metric("rmse", rmse, False),
    "mae": Metric("mae", mae, False),
    "rmsle": Metric("rmsle", rmsle, False),
    "log_loss": Metric("log_loss", log_loss, False),
}


def get_metric(name: str) -> Metric:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(
            f"unknown metric {name!r}; available: {', '.join(sorted(METRICS))}"
        ) from None


__all__ = [
    "InvalidSubmission",
    "METRICS",
    "Metric",
    "accuracy",
    "get_metric",
    "log_loss",
    "mae",
    "rmse",
    "rmsle",
    "roc_auc",
]
