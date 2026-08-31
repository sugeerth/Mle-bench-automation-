import json

import pytest

from mlea.dashboard import AgentRuns, load_session, render_dashboard, write_dashboard


def mkrun(root, agent, comp, seed, *, meta=None, submission=None, agent_log=""):
    d = root / "runs" / agent / f"{comp}__seed{seed}"
    (d / "logs").mkdir(parents=True)
    base = {"competition_id": comp, "seed": seed, "exit_code": 0,
            "wall_clock_seconds": 10, "time_cap_seconds": 60}
    base.update(meta or {})
    (d / "metadata.json").write_text(json.dumps(base))
    if submission is not None:
        (d / "submission").mkdir()
        (d / "submission" / "submission.csv").write_text(submission)
    if agent_log:
        (d / "logs" / "agent.log").write_text(agent_log)


def mkgrades(root, agent, rows):
    d = root / "grades" / agent
    d.mkdir(parents=True)
    (d / "grading_report.json").write_text(json.dumps({"reports": rows}))


def mkcomp(root, comp, metric="roc_auc", gold=0.9, leaderboard=None):
    d = root / "data" / comp
    d.mkdir(parents=True)
    (d / "competition.json").write_text(json.dumps({
        "id": comp, "metric": metric, "task": "binary",
        "thresholds": {"gold": gold, "silver": 0.85, "bronze": 0.8,
                       "median": 0.7, "n_teams": 10}}))
    (d / "leaderboard.json").write_text(
        json.dumps(leaderboard or [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]))


@pytest.fixture
def session_dir(tmp_path):
    for agent, medal, score in (("good", True, 0.95), ("weak", False, 0.55)):
        rows = []
        for comp in ("c1", "c2"):
            for seed in (0, 1):
                mkrun(tmp_path, agent, comp, seed, submission="id,target\n1,0\n")
                rows.append({"competition_id": comp, "seed": seed, "score": score,
                             "any_medal": medal, "gold_medal": medal,
                             "silver_medal": medal, "bronze_medal": medal,
                             "above_median": True, "valid_submission": True,
                             "submission_exists": True, "error": None})
        mkgrades(tmp_path, agent, rows)
    mkrun(tmp_path, "brokenagent", "c1", 0, meta={"exit_code": 1},
          agent_log="Traceback\nValueError")
    for c in ("c1", "c2"):
        mkcomp(tmp_path, c)
    return tmp_path


def test_loads_every_agent(session_dir):
    s = load_session(session_dir)
    assert {a.label for a in s.agents} == {"good", "weak", "brokenagent"}
    assert s.competitions == ["c1", "c2"]
    assert s.seeds == [0, 1]


def test_ranks_by_medal_rate(session_dir):
    assert load_session(session_dir).ranked()[0].label == "good"


def test_percentile_is_computed_from_the_leaderboard(session_dir):
    s = load_session(session_dir)
    good = next(a for a in s.agents if a.label == "good")
    weak = next(a for a in s.agents if a.label == "weak")
    assert s.mean_percentile(good) > s.mean_percentile(weak)


def test_percentile_breaks_a_medal_rate_tie(tmp_path):
    """The reason percentile is shown: medal rate saturates."""
    for agent, score in (("a", 0.95), ("b", 0.88)):
        rows = []
        for seed in (0, 1):
            mkrun(tmp_path, agent, "c1", seed, submission="id,target\n1,0\n")
            rows.append({"competition_id": "c1", "seed": seed, "score": score,
                         "any_medal": True, "gold_medal": True, "silver_medal": True,
                         "bronze_medal": True, "above_median": True,
                         "valid_submission": True, "submission_exists": True,
                         "error": None})
        mkgrades(tmp_path, agent, rows)
    mkcomp(tmp_path, "c1")
    s = load_session(tmp_path)
    assert s.agents[0].medal_rate == s.agents[1].medal_rate == 1.0
    assert s.ranked()[0].label == "a", "the tie must break on percentile"
    assert s.medal_rate_is_saturated
    assert "Medal rate has saturated" in render_dashboard(s)


def test_no_saturation_warning_when_a_single_agent_leads(session_dir):
    assert not load_session(session_dir).medal_rate_is_saturated
    assert "saturated" not in render_dashboard(load_session(session_dir))


def test_missing_grades_degrade_to_zero_not_a_crash(tmp_path):
    mkrun(tmp_path, "a", "c1", 0, submission="id,target\n1,0\n")
    s = load_session(tmp_path)
    assert s.agents[0].medal_rate == 0.0
    assert "<!doctype html>" in render_dashboard(s)


def test_corrupt_grades_are_ignored(tmp_path):
    mkrun(tmp_path, "a", "c1", 0, submission="id,target\n1,0\n")
    d = tmp_path / "grades" / "a"
    d.mkdir(parents=True)
    (d / "grading_report.json").write_text("{not json")
    assert load_session(tmp_path).agents[0].grades == {}


def test_missing_runs_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="runs"):
        load_session(tmp_path)


def test_render_is_self_contained_and_theme_aware(session_dir):
    doc = render_dashboard(load_session(session_dir))
    assert "<script src=" not in doc and "http://" not in doc
    assert "prefers-color-scheme: dark" in doc
    assert ':root[data-theme="dark"]' in doc


def test_matrix_covers_every_agent_and_competition(session_dir):
    doc = render_dashboard(load_session(session_dir))
    for label in ("good", "weak", "brokenagent"):
        assert label in doc
    assert "<polygon" in doc, "the failing agent must render as an agent-bug mark"


def test_mechanical_failure_warning(session_dir):
    doc = render_dashboard(load_session(session_dir))
    assert "failed mechanically" in doc


def test_comparison_note_is_escaped_and_shown(session_dir):
    doc = render_dashboard(load_session(session_dir), "p = 0.04 <significant>")
    assert "&lt;significant&gt;" in doc
    assert "<significant>" not in doc


def test_write_dashboard_creates_the_file(session_dir, tmp_path):
    out = tmp_path / "nested" / "d.html"
    assert write_dashboard(session_dir, out).exists()
    assert out.read_text().startswith("<!doctype")
