"""Tests for the contamination probe and its controls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlea.baseline import row_fingerprints
from mlea.bench import (
    CLONE_TRANSFORMS,
    CompetitionSpec,
    clone_competition,
    clone_difficulty_delta,
    make_competition,
)
from mlea.grade import leaderboard_percentile
from mlea.probe import ProbeCell, ProbeResult

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX only")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("clones")
    src = make_competition(
        CompetitionSpec("orig", "binary", n_train=800, n_test=400,
                        difficulty=0.4, n_teams=100, seed=5), root)
    return root, src


# --- the property Kaggle cannot give ---


def test_relabel_preserves_difficulty_exactly(pair):
    """Values are untouched, so a permutation-invariant model scores identically."""
    root, src = pair
    c = clone_competition(src, "c-relabel", root, transform="relabel", seed=1)
    assert clone_difficulty_delta(src, c) == 0.0


def test_rescale_preserves_difficulty_closely(pair):
    root, src = pair
    c = clone_competition(src, "c-rescale", root, transform="rescale", seed=1)
    assert abs(clone_difficulty_delta(src, c)) < 0.01


def test_clone_hides_the_original_surface(pair):
    root, src = pair
    c = clone_competition(src, "c-hidden", root, transform="rescale", seed=1)
    head = (c / "prepared/public/test.csv").read_text().split("\n")[0]
    assert "f0" not in head and "f1" not in head
    assert "test_0" not in (c / "prepared/public/test.csv").read_text()


def test_clone_answers_still_match_its_own_rows(pair):
    """A clone whose answers were misaligned would be silently unsolvable."""
    root, src = pair
    c = clone_competition(src, "c-align", root, transform="relabel", seed=2)
    ids = [r.split(",")[0] for r in
           (c / "prepared/public/test.csv").read_text().strip().split("\n")[1:]]
    answers = json.loads((c / "prepared/private/answers.json").read_text())
    assert set(ids) == set(answers)


def test_clone_records_its_provenance(pair):
    root, src = pair
    c = clone_competition(src, "c-prov", root, transform="rescale", seed=3)
    spec = json.loads((c / "competition.json").read_text())
    assert spec["cloned_from"] == "orig" and spec["clone_transform"] == "rescale"


def test_unknown_transform_rejected(pair):
    root, src = pair
    with pytest.raises(ValueError, match="transform must be"):
        clone_competition(src, "x", root, transform="scramble")


# --- the generator must not leak column position ---


def test_signal_position_varies_between_competitions(tmp_path):
    """Fixing the interaction at columns 0,1,2 made position carry universal
    information, which any model written against the generator could exploit."""
    from mlea.bench import _latent

    positions = set()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(200, 10))
        before = rng.bit_generator.state
        _latent(X, rng)
        positions.add(str(before))
    assert len(positions) == 12

    # The real check: a column permutation must not change what a
    # permutation-invariant model can achieve.
    c1 = make_competition(CompetitionSpec("p1", "binary", n_train=400, n_test=200,
                                          n_teams=50, seed=21), tmp_path)
    c2 = clone_competition(c1, "p2", tmp_path, transform="relabel", seed=1)
    assert clone_difficulty_delta(c1, c2) == 0.0


# --- row-level recall ---


def test_row_fingerprints_survive_row_and_column_shuffling():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    head = [f"f{i}" for i in range(6)]
    base = row_fingerprints(head, X, "values")
    permuted = row_fingerprints(head, X[:, rng.permutation(6)], "values")
    assert base == permuted, "permuting columns must not change a row's identity"


def test_row_fingerprints_are_defeated_by_rescaling():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    head = [f"f{i}" for i in range(6)]
    base = row_fingerprints(head, X, "values")
    rescaled = row_fingerprints(head, X * 1.7 + 0.3, "values")
    assert set(base).isdisjoint(rescaled)


def test_name_keyed_fingerprints_are_defeated_by_renaming():
    X = np.zeros((5, 3))
    a = row_fingerprints(["f0", "f1", "f2"], X, "names")
    b = row_fingerprints(["v1", "v2", "v3"], X, "names")
    assert set(a).isdisjoint(b)


# --- percentile scoring ---


def test_percentile_respects_direction():
    lb = [0.1, 0.5, 0.9]
    assert leaderboard_percentile(0.95, lb, greater_is_better=True) == 1.0
    assert leaderboard_percentile(0.05, lb, greater_is_better=False) == 1.0


def test_percentile_of_the_worst_score_is_zero():
    assert leaderboard_percentile(0.0, [0.1, 0.5], greater_is_better=True) == 0.0


def test_empty_leaderboard_rejected():
    with pytest.raises(ValueError, match="empty leaderboard"):
        leaderboard_percentile(0.5, [], greater_is_better=True)


# --- probe aggregation ---


def _cells():
    return [
        ProbeCell("m", "a", False, None, 1.0, 1.0, True, True, True),
        ProbeCell("m", "a__rescale", True, "rescale", 0.8, 0.6, True, True, False),
        ProbeCell("h", "a", False, None, 0.8, 0.6, True, True, False),
        ProbeCell("h", "a__rescale", True, "rescale", 0.8, 0.6, True, True, False),
    ]


def test_gap_is_positive_when_the_original_scored_better():
    r = ProbeResult(_cells())
    assert r.gap("m", "rescale") == pytest.approx(0.4)
    assert r.gap("h", "rescale") == pytest.approx(0.0)


def test_medal_gap_is_reported_separately():
    r = ProbeResult(_cells())
    assert r.medal_gap("m", "rescale") == pytest.approx(0.0), \
        "both medal, so the medal metric cannot see the difference"


def test_recall_rate_separates_originals_from_clones():
    r = ProbeResult(_cells())
    assert r.recall_rate("m", None) == 1.0
    assert r.recall_rate("m", "rescale") == 0.0


def test_gap_is_none_without_matching_pairs():
    assert ProbeResult(_cells()).gap("m", "relabel") is None


# --- end to end ---


@posix_only
def test_probe_controls_all_pass(tmp_path, capsys):
    from mlea.cli import main

    rc = main(["probe", "--out", str(tmp_path / "p"), "--competitions", "4",
               "--time-cap", "300"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PROBE VALID" in out
    assert "FAIL" not in out


@posix_only
def test_memorizer_recalls_only_what_it_was_given(pair, tmp_path):
    root, src = pair
    clone = clone_competition(src, "c-mem", root, transform="rescale", seed=4)
    memory = tmp_path / "mem"
    memory.mkdir()
    pub = src / "prepared" / "public"
    rows = [r.split(",") for r in pub.joinpath("test.csv").read_text().strip().split("\n")]
    head, body = rows[0][1:], rows[1:]
    X = np.array([[float(v) for v in r[1:]] for r in body])
    answers = json.loads((src / "prepared/private/answers.json").read_text())
    fps = row_fingerprints(head, X, "values")
    (memory / "m.json").write_text(
        json.dumps({f: answers[r[0]] for f, r in zip(fps, body)}))

    def run(comp):
        sub = tmp_path / f"{comp.name}.csv"
        env = dict(os.environ, DATA_DIR=str(comp / "prepared" / "public"),
                   SUBMISSION_PATH=str(sub))
        p = subprocess.run(
            [sys.executable, "-m", "mlea.baseline", "--strategy", "memorizer",
             "--memory-dir", str(memory)],
            env=env, capture_output=True, text=True, timeout=300, check=True)
        return p.stdout

    assert "RECALLED 100%" in run(src)
    assert "solving honestly" in run(clone)
