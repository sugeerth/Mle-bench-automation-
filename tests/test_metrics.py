import numpy as np
import pytest

from mlea.metrics import (
    InvalidSubmission,
    METRICS,
    accuracy,
    get_metric,
    log_loss,
    mae,
    rmse,
    rmsle,
    roc_auc,
)


def test_perfect_and_inverted_auc():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_constant_predictions_score_exactly_half():
    """The most common degenerate agent output. A naive AUC scores it 0 or 1."""
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, [0.5] * 4) == 0.5
    assert roc_auc(y, [0.0] * 4) == 0.5


def test_partial_ties_are_averaged():
    y = np.array([0, 1, 0, 1])
    assert roc_auc(y, [0.1, 0.5, 0.5, 0.9]) == pytest.approx(0.875)


def test_auc_is_invariant_to_monotone_rescaling():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200)
    assert roc_auc(y, s) == pytest.approx(roc_auc(y, 3 * s + 10))
    assert roc_auc(y, s) == pytest.approx(roc_auc(y, np.exp(s)))


def test_auc_needs_both_classes():
    with pytest.raises(InvalidSubmission, match="both classes"):
        roc_auc(np.zeros(5), np.arange(5))


def test_row_count_mismatch_is_invalid():
    with pytest.raises(InvalidSubmission, match="expected 4"):
        get_metric("rmse")(np.zeros(4), np.zeros(3))


def test_non_finite_predictions_are_invalid():
    with pytest.raises(InvalidSubmission, match="non-finite"):
        get_metric("rmse")(np.zeros(3), np.array([1.0, np.nan, np.inf]))


def test_empty_submission_is_invalid():
    with pytest.raises(InvalidSubmission):
        get_metric("rmse")(np.array([]), np.array([]))


def test_regression_metrics():
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0
    assert rmse([0, 0], [1, 1]) == 1.0
    assert mae([0, 0], [1, -3]) == 2.0
    assert log_loss([1, 0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_log_loss_is_clipped_not_infinite():
    """An overconfident wrong prediction must score badly, not NaN."""
    v = log_loss([1, 0], [0.0, 1.0])
    assert np.isfinite(v) and v > 30


def test_rmsle_rejects_impossible_values():
    with pytest.raises(InvalidSubmission, match="below -1"):
        rmsle([1, 2], [-5, 1])


def test_accuracy_rounds():
    assert accuracy([1, 0, 1], [0.9, 0.4, 0.51]) == 1.0


def test_direction_is_declared_for_every_metric():
    """Everything downstream -- leaderboard sort, medals, 'did it improve' --
    depends on this, and inverting it silently inverts a competition."""
    assert METRICS["roc_auc"].greater_is_better
    assert METRICS["accuracy"].greater_is_better
    for name in ("rmse", "mae", "rmsle", "log_loss"):
        assert not METRICS[name].greater_is_better


def test_unknown_metric_lists_options():
    with pytest.raises(KeyError, match="available"):
        get_metric("nope")
