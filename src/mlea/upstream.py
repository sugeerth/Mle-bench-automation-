"""Adapter to the real MLE-bench: its competition definitions and its graders.

Everything else in this package is self-contained, which is what makes it
runnable for $0. This module is the bridge to the real thing: when upstream
``mlebench`` is importable, a competition can be generated in a **real
competition's schema** and graded by that competition's **real grader**, rather
than by this package's reimplementation.

What that buys, precisely: the pipeline stops being verified only against its
own idea of what a competition looks like. The one thing still missing is the
Kaggle bytes -- which need an account and a rules-acceptance click that cannot
be automated -- and everything above them is exercised for real.

Optional by construction. Nothing here is imported at package load, and every
entry point degrades to a clear message naming what to install.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class UpstreamUnavailable(RuntimeError):
    """Raised when ``mlebench`` cannot be imported."""


INSTALL_HINT = (
    "upstream mlebench is not importable. Clone https://github.com/openai/mle-bench "
    "and put it on PYTHONPATH, then `pip install tqdm pandas scikit-learn scipy "
    "pyyaml diskcache appdirs tenacity py7zr`. A full `pip install -e .` also "
    "pulls the kaggle client, which is only needed for `prepare` and often fails "
    "to build; grading does not need it."
)


def available() -> bool:
    try:
        import mlebench  # noqa: F401
        import mlebench.grade_helpers  # noqa: F401
    except Exception:
        return False
    return True


def _require():
    if not available():
        raise UpstreamUnavailable(INSTALL_HINT)


@lru_cache(maxsize=1)
def competitions_dir() -> Path:
    _require()
    import mlebench

    return Path(mlebench.__file__).parent / "competitions"


@dataclass(frozen=True)
class UpstreamSchema:
    """What a real competition's submission has to look like.

    ``id_column`` and ``target_column`` are read out of the competition's own
    ``grade.py``, which is the only place they are stated -- upstream has no
    machine-readable schema, and the prepared ``sample_submission.csv`` that
    would show it needs the Kaggle download.
    """

    competition_id: str
    metric: str
    id_column: str
    target_column: str
    #: The shared helper the grader routes through, when it uses one.
    helper: str | None = None
    #: Inclusive bounds the grader enforces on the target, when it states any.
    #: Real competitions constrain their target domain -- petfinder rejects any
    #: Pawpularity outside [1, 100] -- so copying only the column names produces
    #: submissions the real grader refuses.
    target_range: tuple[float, float] | None = None

    @property
    def kind(self) -> str | None:
        """The task shape this schema can be synthesised as, if any."""
        if self.metric == "auc-roc":
            return "binary"
        if self.metric in ("root-mean-squared-error", "rmse"):
            return "regression"
        return None


_ID_RE = re.compile(r'id_col\s*=\s*["\']([^"\']+)')
_TGT_RE = re.compile(r'target_col\s*=\s*["\']([^"\']+)')
_HELPER_RE = re.compile(r"prepare_for_(\w+)_metric")
_RANGE_RE = re.compile(r"\.between\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")


def read_schema(competition_id: str) -> UpstreamSchema | None:
    """Extract a competition's submission schema, or ``None`` if it is not stated.

    Two thirds of upstream's competitions do not name their columns in a form
    that can be read without the prepared data -- image and text tasks mostly
    construct them inline. Those are simply skipped rather than guessed at.
    """
    import yaml

    d = competitions_dir() / competition_id
    cfg_path, grade_path = d / "config.yaml", d / "grade.py"
    if not cfg_path.exists() or not grade_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    metric = (cfg.get("grader") or {}).get("name")
    src = grade_path.read_text()
    idc, tgt = _ID_RE.search(src), _TGT_RE.search(src)
    if not (metric and idc and tgt):
        return None
    helper = _HELPER_RE.search(src)
    rng = _RANGE_RE.search(src)
    return UpstreamSchema(
        competition_id, metric, idc.group(1), tgt.group(1),
        helper.group(1) if helper else None,
        (float(rng.group(1)), float(rng.group(2))) if rng else None,
    )


def all_schemas() -> list[UpstreamSchema]:
    return [
        s for c in sorted(p.name for p in competitions_dir().iterdir() if p.is_dir())
        if (s := read_schema(c)) is not None
    ]


def synthesisable_schemas() -> list[UpstreamSchema]:
    """Real competitions this package can generate data for and grade for real."""
    return [s for s in all_schemas() if s.kind is not None]


@lru_cache(maxsize=128)
def load_grader(competition_id: str):
    """The competition's own grader object, loaded from its config."""
    _require()
    import yaml
    from mlebench.grade_helpers import Grader

    cfg = yaml.safe_load(
        (competitions_dir() / competition_id / "config.yaml").read_text()
    )
    return Grader.from_dict(cfg["grader"])


def grade_with_upstream(
    submission_csv: str | Path, answers_csv: str | Path, competition_id: str
) -> tuple[float | None, str | None]:
    """Grade with the real grader. Returns ``(score, error)``.

    Upstream returns ``None`` for a submission its grader rejects and logs the
    reason rather than raising, so an invalid submission arrives here as
    ``(None, message)`` -- the same distinction this package's triage draws
    between an ungradeable submission and a low score.
    """
    _require()
    import pandas as pd

    grader = load_grader(competition_id)
    try:
        submission = pd.read_csv(submission_csv)
        answers = pd.read_csv(answers_csv)
    except Exception as exc:  # pragma: no cover - pandas raises many types
        return None, f"could not read csv: {exc}"
    score = grader(submission, answers)
    if score is None:
        return None, "upstream grader rejected the submission"
    return float(score), None


__all__ = [
    "INSTALL_HINT",
    "UpstreamSchema",
    "UpstreamUnavailable",
    "all_schemas",
    "available",
    "competitions_dir",
    "grade_with_upstream",
    "load_grader",
    "read_schema",
    "synthesisable_schemas",
]
