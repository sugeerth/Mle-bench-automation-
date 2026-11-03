"""Conformance against the real MLE-bench.

These skip unless upstream ``mlebench`` is importable, so the package still
installs and tests with no heavyweight dependencies. When it *is* available they
are the tests that matter most: they check this package's grader against the
real competitions' own graders.
"""

import json
import os
import subprocess
import sys

import pytest

from mlea import upstream

pytestmark = pytest.mark.skipif(
    not upstream.available(), reason="upstream mlebench not importable"
)
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX only")


def test_upstream_exposes_the_full_competition_set():
    assert len(list(upstream.competitions_dir().iterdir())) > 70


def test_schemas_are_read_from_the_real_grade_files():
    schemas = {s.competition_id: s for s in upstream.all_schemas()}
    pizza = schemas["random-acts-of-pizza"]
    assert pizza.id_column == "request_id"
    assert pizza.target_column == "requester_received_pizza"
    assert pizza.metric == "auc-roc"


def test_unreadable_schemas_are_skipped_not_guessed():
    """Two thirds of competitions build their columns inline; those return None
    rather than a plausible-looking guess."""
    assert upstream.read_schema("no-such-competition") is None
    assert len(upstream.all_schemas()) < len(
        list(upstream.competitions_dir().iterdir())
    )


def test_target_domains_are_extracted():
    """Copying only column names produces submissions the real grader refuses."""
    petfinder = upstream.read_schema("petfinder-pawpularity-score")
    assert petfinder.target_range == (1.0, 100.0)
    assert upstream.read_schema("random-acts-of-pizza").target_range is None


def test_synthesisable_set_is_a_real_subset():
    syn = upstream.synthesisable_schemas()
    assert 8 <= len(syn) <= len(upstream.all_schemas())
    assert all(s.kind in ("binary", "regression") for s in syn)


def test_real_grader_loads_and_scores():
    import numpy as np
    import pandas as pd

    grader = upstream.load_grader("random-acts-of-pizza")
    n = 200
    rng = np.random.default_rng(0)
    ids = [f"r{i}" for i in range(n)]
    y = rng.integers(0, 2, n)
    answers = pd.DataFrame({"request_id": ids, "requester_received_pizza": y})
    perfect = pd.DataFrame(
        {"request_id": ids, "requester_received_pizza": y.astype(float)})
    assert grader(perfect, answers) == 1.0
    const = pd.DataFrame(
        {"request_id": ids, "requester_received_pizza": np.full(n, 0.5)})
    assert grader(const, answers) == 0.5


def test_real_grader_rejects_out_of_range_probabilities():
    import numpy as np
    import pandas as pd

    grader = upstream.load_grader("random-acts-of-pizza")
    ids = [f"r{i}" for i in range(50)]
    answers = pd.DataFrame(
        {"request_id": ids, "requester_received_pizza": np.resize([0, 1], 50)})
    bad = pd.DataFrame({"request_id": ids, "requester_received_pizza": np.arange(50.0)})
    assert grader(bad, answers) is None, "raw scores are not probabilities"


def test_we_match_upstreams_five_decimal_rounding():
    """Upstream's Grader.__call__ returns round(score, 5). Matching it is what
    makes a score here comparable to a published MLE-bench number."""
    from mlea.grade import UPSTREAM_SCORE_DECIMALS

    assert UPSTREAM_SCORE_DECIMALS == 5
    import inspect

    from mlebench.grade_helpers import Grader

    assert "round(score, 5)" in inspect.getsource(Grader.__call__)


@posix_only
@pytest.mark.parametrize(
    "competition_id",
    ["random-acts-of-pizza", "aerial-cactus-identification",
     "petfinder-pawpularity-score"],
)
def test_our_grader_agrees_with_the_real_one(tmp_path, competition_id):
    """The conformance check, on one competition of each shape."""
    from mlea.bench import from_upstream, make_competition
    from mlea.cli import _write_answers_csv
    from mlea.grade import grade_submission

    spec = from_upstream(competition_id, n_train=400, n_test=250, n_teams=40,
                         difficulty=0.4, seed=5)
    comp = make_competition(spec, tmp_path)
    answers_csv = comp / "prepared" / "private" / "answers.csv"
    _write_answers_csv(comp, spec, answers_csv)

    for strategy in ("constant", "linear", "broken"):
        sub = comp / f"{strategy}.csv"
        env = dict(os.environ, DATA_DIR=str(comp / "prepared" / "public"),
                   SUBMISSION_PATH=str(sub))
        subprocess.run(
            [sys.executable, "-m", "mlea.baseline", "--strategy", strategy],
            env=env, capture_output=True, text=True, timeout=300)
        ours = grade_submission(sub, comp)
        theirs, _ = upstream.grade_with_upstream(sub, answers_csv, competition_id)
        if ours.valid_submission:
            assert theirs is not None, f"{strategy}: we accepted, upstream rejected"
            assert ours.score == theirs, f"{strategy}: {ours.score} != {theirs}"
        else:
            assert theirs is None, f"{strategy}: we rejected, upstream accepted"


def test_generated_competition_uses_the_real_schema(tmp_path):
    from mlea.bench import from_upstream, make_competition

    spec = from_upstream("random-acts-of-pizza", n_train=200, n_test=100,
                         n_teams=30, seed=1)
    comp = make_competition(spec, tmp_path)
    header = (comp / "prepared/public/sample_submission.csv").read_text().split("\n")[0]
    assert header == "request_id,requester_received_pizza"
    meta = json.loads((comp / "competition.json").read_text())
    assert meta["upstream_id"] == "random-acts-of-pizza"


def test_unsynthesisable_competition_is_refused():
    from mlea.bench import from_upstream

    with pytest.raises(ValueError, match="cannot synthesise"):
        from_upstream("billion-word-imputation")
    with pytest.raises(ValueError, match="does not state"):
        from_upstream("no-such-competition")
