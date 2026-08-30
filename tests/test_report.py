import json

import pytest

from mlea.report import (
    OUTCOME_ORDER,
    TIER_OF,
    ReportData,
    RunRow,
    collect,
    render_html,
    write_report,
)
from mlea.triage import Outcome


def mkrun(root, comp, seed, *, meta=None, submission=None, agent_log="",
          harness_log=""):
    d = root / f"{comp}__seed{seed}"
    (d / "logs").mkdir(parents=True)
    base = {"competition_id": comp, "seed": seed, "exit_code": 0,
            "wall_clock_seconds": 600, "time_cap_seconds": 3600}
    base.update(meta or {})
    (d / "metadata.json").write_text(json.dumps(base))
    if submission is not None:
        (d / "submission").mkdir()
        (d / "submission" / "submission.csv").write_text(submission)
    if agent_log:
        (d / "logs" / "agent.log").write_text(agent_log)
    if harness_log:
        (d / "logs" / "harness.log").write_text(harness_log)
    return d


@pytest.fixture
def group(tmp_path):
    g = tmp_path / "group"
    g.mkdir()
    mkrun(g, "c1", 0, submission="id,y\n1,0\n")
    mkrun(g, "c1", 1, submission="id,y\n1,0\n",
          meta={"checkpoints": [{"elapsed_seconds": 60, "sha256": "a",
                                 "path": "checkpoints/t=60/submission.csv"}]})
    mkrun(g, "c2", 0, meta={"exit_code": 1},
          agent_log="RuntimeError: CUDA out of memory")
    mkrun(g, "c2", 1, meta={"exit_code": 1},
          harness_log="Spot ITN received. Instance will be interrupted")
    return g


# --- every outcome has a tier, or the chart silently drops runs ---


def test_every_outcome_is_tiered():
    for outcome in Outcome:
        assert outcome in TIER_OF
        assert TIER_OF[outcome] in {"signal", "agent", "excluded"}


def test_only_valid_is_capability_signal():
    signal = [o for o, t in TIER_OF.items() if t == "signal"]
    assert signal == [Outcome.VALID]


def test_breakdown_order_covers_every_outcome():
    assert set(OUTCOME_ORDER) == set(Outcome)


# --- collection ---


def test_collect_reads_every_run(group):
    data = collect(group)
    assert len(data.rows) == 4
    assert data.competitions == ["c1", "c2"]
    assert data.seeds == [0, 1]


def test_collect_assigns_the_right_tiers(group):
    tiers = collect(group).tier_counts()
    assert tiers == {"signal": 2, "agent": 1, "excluded": 1}


def test_infra_is_excluded_from_the_denominator(group):
    assert collect(group).effective_denominator == 3


def test_collect_reads_checkpoints(group):
    rows = {(r.competition_id, r.seed): r for r in collect(group).rows}
    assert rows[("c1", 1)].checkpoints == (60.0,)
    assert rows[("c1", 0)].checkpoints == ()


def test_collect_survives_corrupt_metadata(tmp_path):
    g = tmp_path / "g"
    d = g / "c1__seed0"
    (d / "logs").mkdir(parents=True)
    (d / "metadata.json").write_text("{not json")
    assert len(collect(g).rows) == 1


def test_collect_ignores_loose_files(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    mkrun(g, "c1", 0, submission="a\n1\n")
    (g / "submissions.jsonl").write_text("{}\n")
    assert len(collect(g).rows) == 1


# --- rendering ---


def test_render_is_self_contained(group):
    doc = render_html(collect(group))
    assert doc.startswith("<!doctype html>")
    assert "<script src=" not in doc and "<link rel=\"stylesheet\"" not in doc
    assert "http://" not in doc


def test_render_declares_dark_mode_under_both_scopes(group):
    doc = render_html(collect(group))
    assert "prefers-color-scheme: dark" in doc
    assert ':root[data-theme="dark"]' in doc
    assert ':root:not([data-theme="light"])' in doc


def test_render_has_one_mark_per_run(group):
    doc = render_html(collect(group))
    assert doc.count('class="dot" tabindex="0"') >= 4


def test_render_includes_a_legend_and_a_table(group):
    doc = render_html(collect(group))
    assert "Capability signal" in doc and "Agent bug" in doc
    assert "<table>" in doc


def test_marks_differ_by_shape_not_only_colour(group):
    """Colour must never be the only channel carrying the tier."""
    doc = render_html(collect(group))
    assert "<polygon" in doc        # agent bug: diamond
    assert 'fill="none"' in doc     # excluded: hollow ring
    assert "<circle" in doc         # signal: filled circle


def test_plumbing_warning_appears_when_agent_bugs_dominate(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    for i in range(4):
        mkrun(g, f"c{i}", 0, meta={"exit_code": 1})
    assert "failed mechanically" in render_html(collect(g))


def test_no_plumbing_warning_when_runs_are_clean(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    for i in range(10):
        mkrun(g, f"c{i}", 0, submission="a\n1\n")
    assert "failed mechanically" not in render_html(collect(g))


def test_timeline_only_appears_when_checkpoints_exist(tmp_path, group):
    assert "last change its submission" in render_html(collect(group))
    g = tmp_path / "plain"
    g.mkdir()
    mkrun(g, "c1", 0, submission="a\n1\n")
    assert "last change its submission" not in render_html(collect(g))


def test_competition_names_are_escaped(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    mkrun(g, "evil<script>alert(1)</script>", 0, submission="a\n1\n")
    doc = render_html(collect(g))
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_evidence_from_agent_logs_is_escaped(tmp_path):
    """Evidence text is agent-controlled and lands in HTML attributes."""
    g = tmp_path / "g"
    g.mkdir()
    mkrun(g, "c1", 0, meta={"exit_code": 1},
          agent_log='CUDA out of memory" onmouseover="alert(1)')
    doc = render_html(collect(g))
    assert 'onmouseover="alert(1)"' not in doc


def test_empty_group_renders(tmp_path):
    g = tmp_path / "empty"
    g.mkdir()
    doc = render_html(collect(g))
    assert "No runs." in doc


def test_missing_cells_render_as_absent(tmp_path):
    """A competition run at seed 0 only must not fake a seed-1 result."""
    g = tmp_path / "g"
    g.mkdir()
    mkrun(g, "c1", 0, submission="a\n1\n")
    mkrun(g, "c2", 0, submission="a\n1\n")
    mkrun(g, "c2", 1, submission="a\n1\n")
    data = collect(g)
    assert data.by_key().get(("c1", 1)) is None
    assert "opacity=\"0.4\"" in render_html(data)


def test_write_report_creates_the_file(tmp_path, group):
    out = tmp_path / "nested" / "r.html"
    assert write_report(group, out).exists()
    assert out.read_text().startswith("<!doctype")


# --- scale and axis quality ---


def test_timeline_caps_its_row_count(tmp_path):
    """225 runs must not render 225 rows."""
    from mlea.report import TIMELINE_MAX_ROWS

    g = tmp_path / "big"
    g.mkdir()
    for i in range(TIMELINE_MAX_ROWS + 12):
        mkrun(g, f"c{i:03d}", 0, submission="a\n1\n",
              meta={"checkpoints": [{"elapsed_seconds": i + 1, "sha256": "x",
                                     "path": "p"}]})
    doc = render_html(collect(g))
    assert "12 more are in the table below" in doc


def test_timeline_shows_the_latest_changing_runs_first(tmp_path):
    """A run still improving at the cap is the finding; it must not be cut."""
    from mlea.report import TIMELINE_MAX_ROWS

    g = tmp_path / "big"
    g.mkdir()
    for i in range(TIMELINE_MAX_ROWS + 5):
        mkrun(g, f"c{i:03d}", 0, submission="a\n1\n",
              meta={"checkpoints": [{"elapsed_seconds": i + 1, "sha256": "x",
                                     "path": "p"}]})
    doc = render_html(collect(g))
    latest = f"c{TIMELINE_MAX_ROWS + 4:03d}"
    assert latest in doc
    assert "c000 · s0" not in doc


@pytest.mark.parametrize("tmax", [6.15, 60, 3600, 86400, 0.5])
def test_axis_ticks_are_round_numbers(tmax):
    from mlea.report import _nice_ticks

    ticks = _nice_ticks(tmax)
    assert ticks[0] == 0
    assert len(ticks) >= 2
    steps = {round(b - a, 6) for a, b in zip(ticks, ticks[1:])}
    assert len(steps) == 1, "tick spacing must be uniform"


def test_nice_ticks_handles_zero():
    from mlea.report import _nice_ticks

    assert _nice_ticks(0) == [0.0]


def test_timeline_axis_spans_the_data_not_the_cap(tmp_path):
    """A 24h cap with 3-second runs squashes every mark against the origin."""
    g = tmp_path / "g"
    g.mkdir()
    mkrun(g, "c1", 0, submission="a\n1\n",
          meta={"wall_clock_seconds": 3, "time_cap_seconds": 86400,
                "checkpoints": [{"elapsed_seconds": 1, "sha256": "x", "path": "p"}]})
    data = collect(g)
    assert data.max_time < 10
    assert data.time_cap == 86400
    assert data.cap_headroom < 0.01
    assert "used 0% of it" in render_html(data)


def test_no_cap_note_when_runs_use_most_of_the_budget(tmp_path):
    g = tmp_path / "g"
    g.mkdir()
    mkrun(g, "c1", 0, submission="a\n1\n",
          meta={"wall_clock_seconds": 90, "time_cap_seconds": 100,
                "checkpoints": [{"elapsed_seconds": 50, "sha256": "x", "path": "p"}]})
    assert "The axis spans the longest run" not in render_html(collect(g))
