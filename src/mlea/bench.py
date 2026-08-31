"""Generate real, gradeable competitions without Kaggle.

The gap this closes: every piece of tooling in this repository had only ever run
against stub agents, because exercising it for real needs Kaggle credentials
*and* a rules-acceptance click that cannot be automated (upstream literally does
``webbrowser.open(...)`` then ``input()``). So the pipeline was untested against
anything that could actually be scored.

These competitions are synthetic but not fake. Each has a latent function, real
predictive signal, a held-out test split whose answers the agent cannot see, a
real metric, and a leaderboard built by **fitting actual models** -- ridge
regressions of varying capacity on bootstrap resamples of the training data --
and scoring them on the same test split the agent is graded on. Medal thresholds
come from that distribution using upstream's rank tiers.

Two properties this buys that the real benchmark does not have:

* **Zero threshold-transfer bias.** The leaderboard is computed on the same
  split as the agent's score. Upstream compares a score on its own re-split
  against a raw value from Kaggle's differently-split leaderboard, which is
  biased upward (see :mod:`mlea.grade`).
* **A known ground truth.** The latent function is known, so an oracle score is
  computable, which makes it possible to check that a grader, a harness or a
  metric is behaving -- the correctness oracle whose absence makes a rolling
  Kaggle split infeasible.

They are not a substitute for MLE-bench. They are tabular, generated, and much
smaller. Their job is to prove the plumbing works before anyone spends money.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grade import thresholds_from_leaderboard
from .metrics import get_metric

#: Task shapes, each with the metric it is scored by.
TASKS = {
    "binary": "roc_auc",
    "regression": "rmse",
}


@dataclass(frozen=True)
class CompetitionSpec:
    id: str
    task: str = "binary"
    n_train: int = 2_000
    n_test: int = 1_000
    n_features: int = 12
    #: 0 = trivially separable, 1 = almost pure noise. Controls how much of the
    #: target the latent function explains.
    difficulty: float = 0.5
    n_teams: int = 300
    seed: int = 0

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {sorted(TASKS)}")
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        if self.n_teams < 1:
            raise ValueError("n_teams must be >= 1")

    @property
    def metric(self) -> str:
        return TASKS[self.task]


def _latent(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A non-trivial signal: linear terms, an interaction, and a nonlinearity.

    Purely linear targets make ridge an oracle and collapse the leaderboard, so
    there has to be structure a linear model cannot fully capture -- otherwise
    every team ties and medal thresholds become meaningless.

    The interacting and nonlinear columns are chosen **at random per
    competition**. Fixing them at columns 0, 1 and 2 made column position carry
    universal information about where the signal lives, which any model written
    against this generator could exploit -- an advantage no real competition
    offers, and one that shows up as a spurious effect the moment columns are
    permuted.
    """
    n_features = X.shape[1]
    w = rng.normal(0, 1, size=n_features)
    linear = X @ w
    i, j, k = rng.choice(n_features, size=3, replace=n_features < 3)
    interaction = 1.5 * X[:, i] * X[:, j]
    nonlinear = 1.2 * np.tanh(2.0 * X[:, k])
    return linear + interaction + nonlinear


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge. Solved rather than inverted, for conditioning."""
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ y)


def _design_invariant(X: np.ndarray) -> np.ndarray:
    """A feature map that does not depend on column *position*.

    :func:`_design` hand-picks ``X[:,0]*X[:,1]`` and ``tanh(X[:,2])``, so a
    column permutation moves the signal out from under it. Measuring a clone's
    difficulty with such a model reports a difference that is an artefact of the
    measuring model, not of the problem -- which is exactly the confound a
    contamination probe must not have. Linear and elementwise-square terms are
    permutation-equivariant, so a ridge fit on them is invariant to column order.
    """
    return np.hstack([np.ones((X.shape[0], 1)), X, X**2])  # see _design


def _design(X: np.ndarray, capacity: int) -> np.ndarray:
    """Feature map at a given modelling capacity -- a team's skill level.

    Every level is **permutation-equivariant**: the set of columns produced does
    not depend on the input column order. A design that hand-picks indices makes
    a model's score depend on where the generator happened to put the signal,
    which is both unrealistic and fatal to a clone-based probe.
    """
    cols = [np.ones((X.shape[0], 1)), X]
    if capacity >= 2:
        cols.append(X**2)
    if capacity >= 3 and X.shape[1] > 1:
        cols.append(np.tanh(2.0 * X))
    if capacity >= 4 and X.shape[1] > 1:
        iu = np.triu_indices(X.shape[1], k=1)
        cols.append(X[:, iu[0]] * X[:, iu[1]])
    return np.hstack(cols)


def _simulate_leaderboard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric_name: str,
    n_teams: int,
    rng: np.random.Generator,
) -> list[float]:
    """Score a field of real fitted models on the real test split.

    Each team gets a capacity, a regularisation strength and a bootstrap
    resample of the training data, so the spread is genuine model-quality
    variation rather than noise added to an oracle.
    """
    metric = get_metric(metric_name)
    n = X_train.shape[0]
    scores: list[float] = []
    for _ in range(n_teams):
        capacity = int(rng.choice([1, 1, 2, 2, 3, 3, 4], size=1)[0])
        alpha = float(10 ** rng.uniform(-2, 2.5))
        frac = float(rng.uniform(0.25, 1.0))
        idx = rng.integers(0, n, size=max(int(n * frac), 20))
        A = _design(X_train[idx], capacity)
        beta = _ridge(A, y_train[idx], alpha)
        pred = _design(X_test, capacity) @ beta
        try:
            scores.append(metric(y_test, pred))
        except Exception:
            continue  # a degenerate fit is a team that scored nothing
    if not scores:
        raise RuntimeError("leaderboard simulation produced no valid scores")
    return scores


def _write_csv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


DESCRIPTION = """# {id}

## Task

{task_line}

## Data

`train.csv` has {n_train} rows: an `{id_column}` column, {n_features} numeric
feature columns `f0`..`f{last_feature}`, and the target column `{target_column}`.

`test.csv` has {n_test} rows with the same feature columns and no target.

## Evaluation

Submissions are scored by **{metric}** ({direction} is better).

## Submission format

A CSV with exactly the columns `{id_column},{target_column}`, one row per row of
`test.csv`, matching `sample_submission.csv`. Every test id must appear exactly
once. {target_hint}
"""


def make_competition(spec: CompetitionSpec, root: str | Path) -> Path:
    """Create one competition on disk in the mle-bench prepared layout.

    Deterministic in ``spec.seed``: the same spec always produces byte-identical
    data and the same medal thresholds, so a run is reproducible and a
    regression in the tooling cannot hide behind fresh random data.
    """
    rng = np.random.default_rng(spec.seed)
    comp_dir = Path(root) / spec.id
    public = comp_dir / "prepared" / "public"
    private = comp_dir / "prepared" / "private"

    n_total = spec.n_train + spec.n_test
    X = rng.normal(0, 1, size=(n_total, spec.n_features))
    signal = _latent(X, rng)
    signal = (signal - signal.mean()) / (signal.std() + 1e-12)
    # difficulty 0 -> all signal; difficulty 1 -> all noise.
    noise_sd = np.tan(np.clip(spec.difficulty, 0.0, 0.98) * np.pi / 2)
    noisy = signal + rng.normal(0, noise_sd, size=n_total)

    if spec.task == "binary":
        y = (noisy > np.median(noisy)).astype(float)
        # Best achievable prediction given X: the noise-free latent signal.
        # (Using `noisy` would leak the label and score a meaningless 1.0.)
        clean = signal
        target_hint = "Predictions may be any real number; only their ranking matters."
        task_line = (
            "Predict the probability that each row belongs to the positive class."
        )
    else:
        y = 10.0 + 3.0 * noisy
        clean = 10.0 + 3.0 * signal
        target_hint = "Predictions are real-valued."
        task_line = "Predict the continuous target for each row."

    X_train, X_test = X[: spec.n_train], X[spec.n_train :]
    y_train, y_test = y[: spec.n_train], y[spec.n_train :]

    id_column, target_column = "id", "target"
    feats = [f"f{i}" for i in range(spec.n_features)]
    train_ids = [f"train_{i}" for i in range(spec.n_train)]
    test_ids = [f"test_{i}" for i in range(spec.n_test)]

    _write_csv(
        public / "train.csv",
        [id_column, *feats, target_column],
        ([i, *[f"{v:.6f}" for v in row], f"{t:.6f}"]
         for i, row, t in zip(train_ids, X_train, y_train)),
    )
    _write_csv(
        public / "test.csv",
        [id_column, *feats],
        ([i, *[f"{v:.6f}" for v in row]] for i, row in zip(test_ids, X_test)),
    )
    baseline = 0.5 if spec.task == "binary" else float(np.mean(y_train))
    _write_csv(
        public / "sample_submission.csv",
        [id_column, target_column],
        ([i, f"{baseline:.6f}"] for i in test_ids),
    )
    (public / "description.md").write_text(
        DESCRIPTION.format(
            id=spec.id, task_line=task_line, n_train=spec.n_train,
            n_test=spec.n_test, n_features=spec.n_features,
            last_feature=spec.n_features - 1, id_column=id_column,
            target_column=target_column, metric=spec.metric,
            direction="higher" if get_metric(spec.metric).greater_is_better else "lower",
            target_hint=target_hint,
        )
    )

    private.mkdir(parents=True, exist_ok=True)
    (private / "answers.json").write_text(
        json.dumps({i: float(v) for i, v in zip(test_ids, y_test)})
    )

    leaderboard = _simulate_leaderboard(
        X_train, y_train, X_test, y_test, spec.metric, spec.n_teams, rng
    )
    thresholds = thresholds_from_leaderboard(
        leaderboard, get_metric(spec.metric).greater_is_better
    )
    # The oracle: the score of the best prediction obtainable from X, i.e. the
    # noise-free target. For a metric where lower is better this is the
    # irreducible error, not zero -- scoring the answers against themselves
    # would report a perfect 0.0000 and mean nothing.
    oracle = get_metric(spec.metric)(y_test, clean[spec.n_train :])

    (comp_dir / "competition.json").write_text(
        json.dumps(
            {
                "id": spec.id,
                "task": spec.task,
                "metric": spec.metric,
                "id_column": id_column,
                "target_column": target_column,
                "difficulty": spec.difficulty,
                "seed": spec.seed,
                "thresholds": {
                    "gold": thresholds.gold, "silver": thresholds.silver,
                    "bronze": thresholds.bronze, "median": thresholds.median,
                    "n_teams": thresholds.n_teams,
                },
                "oracle_score": oracle,
                "generated_by": "mlea.bench",
            },
            indent=2,
        )
    )
    (comp_dir / "leaderboard.json").write_text(json.dumps(sorted(leaderboard)))
    return comp_dir


# --- matched clones: the contamination probe the real benchmark cannot run ----


CLONE_TRANSFORMS = ("relabel", "rescale")


def clone_competition(
    source: str | Path,
    clone_id: str,
    root: str | Path,
    *,
    transform: str = "rescale",
    seed: int = 0,
) -> Path:
    """Create a surface-different competition with the same underlying problem.

    This is the thing a rolling Kaggle split cannot give you. Matching two real
    competitions on difficulty is guesswork, and a systematic difficulty
    difference is indistinguishable from contamination -- the bias that made
    :mod:`docs/PROPOSAL-mle-bench-live` unworkable. Here the clone *is* the same
    problem, so the difficulty difference is zero (``relabel``) or measurable
    and tiny (``rescale``).

    ``relabel`` permutes and renames the columns, shuffles the rows and reissues
    the ids. Values are untouched, so difficulty is **exactly** preserved. It
    defeats recall keyed on names, ids or ordering, and does not defeat recall
    keyed on the values themselves.

    ``rescale`` additionally applies a positive per-column affine map. That
    defeats value-keyed recall too, at the cost of perturbing the latent
    nonlinearity, so the difficulty match becomes empirical rather than exact --
    :func:`clone_difficulty_delta` measures it.
    """
    if transform not in CLONE_TRANSFORMS:
        raise ValueError(f"transform must be one of {CLONE_TRANSFORMS}")
    src = Path(source)
    spec_json = json.loads((src / "competition.json").read_text())
    rng = np.random.default_rng(seed)

    def read(path: Path) -> tuple[list[str], list[list[str]]]:
        with path.open(newline="") as fh:
            rows = list(csv.reader(fh))
        return rows[0], rows[1:]

    train_head, train_rows = read(src / "prepared" / "public" / "train.csv")
    test_head, test_rows = read(src / "prepared" / "public" / "test.csv")
    answers = json.loads((src / "prepared" / "private" / "answers.json").read_text())

    n_features = len(test_head) - 1
    perm = rng.permutation(n_features)
    # Opaque names, so nothing about the original survives in the header.
    new_names = [f"v{rng.integers(1000, 9999)}_{i}" for i in range(n_features)]
    if transform == "rescale":
        scale = rng.uniform(0.5, 2.0, size=n_features)
        shift = rng.normal(0, 0.5, size=n_features)
    else:
        scale, shift = np.ones(n_features), np.zeros(n_features)

    def remap(row: list[str], has_target: bool) -> list[str]:
        feats = np.array([float(v) for v in row[1 : 1 + n_features]])
        moved = feats[perm] * scale[perm] + shift[perm]
        out = [f"{v:.6f}" for v in moved]
        return out + ([row[-1]] if has_target else [])

    train_order = rng.permutation(len(train_rows))
    test_order = rng.permutation(len(test_rows))
    new_train_ids = [f"r{i}" for i in range(len(train_rows))]
    new_test_ids = [f"q{i}" for i in range(len(test_rows))]

    comp_dir = Path(root) / clone_id
    public, private = comp_dir / "prepared" / "public", comp_dir / "prepared" / "private"
    _write_csv(
        public / "train.csv",
        ["id", *new_names, "target"],
        ([nid, *remap(train_rows[o], True)]
         for nid, o in zip(new_train_ids, train_order)),
    )
    _write_csv(
        public / "test.csv",
        ["id", *new_names],
        ([nid, *remap(test_rows[o], False)]
         for nid, o in zip(new_test_ids, test_order)),
    )
    old_test_ids = [r[0] for r in test_rows]
    new_answers = {
        nid: answers[old_test_ids[o]] for nid, o in zip(new_test_ids, test_order)
    }
    baseline = 0.5 if spec_json["task"] == "binary" else float(
        np.mean([float(r[-1]) for r in train_rows])
    )
    _write_csv(
        public / "sample_submission.csv",
        ["id", "target"],
        ([i, f"{baseline:.6f}"] for i in new_test_ids),
    )
    (public / "description.md").write_text(
        (src / "prepared" / "public" / "description.md")
        .read_text()
        .replace(spec_json["id"], clone_id)
        .replace("`f0`..`f" + str(n_features - 1) + "`", "opaque feature columns")
    )
    private.mkdir(parents=True, exist_ok=True)
    (private / "answers.json").write_text(json.dumps(new_answers))

    clone_spec = dict(spec_json)
    clone_spec.update(
        {
            "id": clone_id,
            "cloned_from": spec_json["id"],
            "clone_transform": transform,
            "generated_by": "mlea.bench.clone_competition",
        }
    )
    (comp_dir / "competition.json").write_text(json.dumps(clone_spec, indent=2))
    (comp_dir / "leaderboard.json").write_text(
        (src / "leaderboard.json").read_text()
    )
    return comp_dir


def clone_difficulty_delta(original: str | Path, clone: str | Path) -> float:
    """Measured difficulty gap between a competition and its clone.

    Fits a **permutation-invariant** reference model on each and returns
    ``clone - original`` in metric units, signed so that positive always means
    *the clone is easier*. A probe's whole validity rests on this being ~0: any
    real difficulty difference is indistinguishable from a memorisation effect.

    Permutation invariance is not a detail. Measuring with a model that
    hand-picks column indices reports a difference that is an artefact of the
    measuring model rather than of the problem.
    """
    def fit_score(comp: Path) -> float:
        spec = json.loads((comp / "competition.json").read_text())
        metric = get_metric(spec["metric"])
        pub = comp / "prepared" / "public"

        def load(path: Path, has_target: bool):
            with path.open(newline="") as fh:
                rows = list(csv.reader(fh))
            body = rows[1:]
            end = -1 if has_target else len(rows[0])
            X = np.array([[float(v) for v in r[1:end]] for r in body])
            y = np.array([float(r[-1]) for r in body]) if has_target else None
            return [r[0] for r in body], X, y

        _, X_tr, y_tr = load(pub / "train.csv", True)
        ids, X_te, _ = load(pub / "test.csv", False)
        beta = _ridge(_design_invariant(X_tr), y_tr, 1.0)
        pred = _design_invariant(X_te) @ beta
        answers = json.loads((comp / "prepared/private/answers.json").read_text())
        y_true = np.array([answers[i] for i in ids])
        return metric(y_true, pred), metric.greater_is_better

    a, gib = fit_score(Path(original))
    b, _ = fit_score(Path(clone))
    return (b - a) if gib else (a - b)


#: A default suite spanning task type and difficulty, so a self-test exercises
#: an easy competition, a hard one, and both metric directions.
SUITE: tuple[CompetitionSpec, ...] = (
    CompetitionSpec("synth-easy-binary", "binary", difficulty=0.25, seed=1),
    CompetitionSpec("synth-hard-binary", "binary", difficulty=0.75, seed=2),
    CompetitionSpec("synth-easy-regression", "regression", difficulty=0.3, seed=3),
    CompetitionSpec("synth-hard-regression", "regression", difficulty=0.8, seed=4),
    CompetitionSpec("synth-wide-binary", "binary", n_features=30, difficulty=0.5, seed=5),
    CompetitionSpec("synth-small-binary", "binary", n_train=400, n_test=250,
                    difficulty=0.4, seed=6),
)


def make_suite(root: str | Path, specs: tuple[CompetitionSpec, ...] = SUITE) -> list[Path]:
    return [make_competition(s, root) for s in specs]


__all__ = [
    "CLONE_TRANSFORMS",
    "CompetitionSpec",
    "clone_competition",
    "clone_difficulty_delta",
    "SUITE",
    "TASKS",
    "make_competition",
    "make_suite",
]
