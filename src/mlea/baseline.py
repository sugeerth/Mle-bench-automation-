"""Reference agents: real models, and every failure mode worth testing.

Run as ``python -m mlea.baseline --strategy <name>``. Reads ``DATA_DIR`` and
``SUBMISSION_PATH`` from the environment, matching the harness contract, so
these are wired in exactly like a real agent would be.

The modelling strategies genuinely fit models and genuinely differ in quality,
so a sweep over them produces real medal variation rather than a fixed answer.
The failure strategies exist so the self-test can prove that triage classifies
each failure mode correctly against a real run rather than a synthetic fixture.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

MODELLING = ("constant", "linear", "tuned")
FAILING = ("broken", "silent", "crash", "hungry")
STRATEGIES = MODELLING + FAILING


def _read_csv(path: Path) -> tuple[list[str], np.ndarray, np.ndarray | None]:
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    ids = [r[0] for r in body]
    has_target = header[-1] == "target"
    end = -1 if has_target else len(header)
    X = np.array([[float(v) for v in r[1:end]] for r in body], dtype=float)
    y = np.array([float(r[-1]) for r in body]) if has_target else None
    return ids, X, y


def _design(X: np.ndarray, capacity: int) -> np.ndarray:
    cols = [np.ones((X.shape[0], 1)), X]
    if capacity >= 2 and X.shape[1] > 1:
        cols.append((X[:, 0] * X[:, 1])[:, None])
    if capacity >= 3 and X.shape[1] > 2:
        cols.append(np.tanh(2.0 * X[:, 2])[:, None])
    if capacity >= 4:
        cols.append(X**2)
    return np.hstack(cols)


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def _write(path: Path, ids: list[str], preds: np.ndarray) -> None:
    """Write atomically.

    The harness snapshots this file on a timer from another thread. A plain
    truncate-and-write would let a snapshot catch a half-written file and score
    it as malformed -- an instrumentation artefact indistinguishable from a real
    agent bug.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "target"])
        w.writerows([[i, f"{p:.6f}"] for i, p in zip(ids, preds)])
    tmp.replace(path)


def _holdout_score(X: np.ndarray, y: np.ndarray, capacity: int, alpha: float,
                   rng: np.random.Generator) -> float:
    n = X.shape[0]
    idx = rng.permutation(n)
    cut = int(n * 0.75)
    tr, va = idx[:cut], idx[cut:]
    beta = _ridge(_design(X[tr], capacity), y[tr], alpha)
    pred = _design(X[va], capacity) @ beta
    return float(-np.mean((y[va] - pred) ** 2))  # higher is better


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mlea-baseline")
    p.add_argument("--strategy", required=True, choices=STRATEGIES)
    p.add_argument("--think-seconds", type=float, default=0.0,
                   help="pause between search steps, so checkpoint marks land "
                        "between improvements and a real curve is visible")
    args = p.parse_args(argv)

    data_dir = Path(os.environ["DATA_DIR"])
    sub_path = Path(os.environ["SUBMISSION_PATH"])
    strategy = args.strategy

    if strategy == "silent":
        print("thought about it, wrote nothing", flush=True)
        return 0
    if strategy == "crash":
        print("Traceback (most recent call last):\nValueError: bad shape",
              file=sys.stderr, flush=True)
        return 1
    if strategy == "hungry":
        print("RuntimeError: CUDA out of memory. Tried to allocate 20.00 GiB",
              file=sys.stderr, flush=True)
        return 1

    train_ids, X_tr, y_tr = _read_csv(data_dir / "train.csv")
    test_ids, X_te, _ = _read_csv(data_dir / "test.csv")
    assert y_tr is not None

    if strategy == "broken":
        # A plausible real bug: right rows, wrong column name.
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        with sub_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "prediction"])
            w.writerows([[i, "0.5"] for i in test_ids])
        print("wrote submission", flush=True)
        return 0

    if strategy == "constant":
        _write(sub_path, test_ids, np.full(len(test_ids), float(np.mean(y_tr))))
        print("wrote constant baseline", flush=True)
        return 0

    if strategy == "linear":
        beta = _ridge(_design(X_tr, 1), y_tr, 1.0)
        _write(sub_path, test_ids, _design(X_te, 1) @ beta)
        print("fit ridge on raw features", flush=True)
        return 0

    # tuned: a real search, writing a new submission whenever it improves.
    rng = np.random.default_rng(0)
    best = -np.inf
    best_beta = None
    best_capacity = 1
    for capacity in (1, 2, 3, 4):
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            score = _holdout_score(X_tr, y_tr, capacity, alpha, rng)
            if score > best:
                best, best_capacity = score, capacity
                best_beta = _ridge(_design(X_tr, capacity), y_tr, alpha)
                _write(sub_path, test_ids, _design(X_te, best_capacity) @ best_beta)
                print(f"improved: capacity={capacity} alpha={alpha} "
                      f"holdout={score:.5f}", flush=True)
            if args.think_seconds:
                time.sleep(args.think_seconds)
    print(f"done, best holdout {best:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
