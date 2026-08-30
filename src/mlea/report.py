"""Render a run group as a self-contained HTML report.

The headline view is the **eval dots** grid: one dot per run, competitions down,
seeds across. It answers the question a table of percentages hides -- *is this
score made of ML results or of plumbing failures?* -- at a glance.

Colour encodes the three tiers this repository has argued for throughout: a
gradeable result (capability signal), an agent bug, and our own infrastructure
fault (excluded from the denominator). These are **kinds of thing, not
good/bad states**, so they take categorical identity colours rather than the
status palette.

The palette was validated rather than eyeballed, and the first attempt failed:
status-good green against status-serious measured CVD Delta E 5.6 (protan), and
green against status-critical red measured 4.1 (deutan) -- so the chart's single
most important distinction would have been invisible to a red-green colourblind
reader. The shipped blue/red/neutral clears every gate in both modes. Every dot
additionally carries a distinct shape and a text label in its tooltip, and a
table view is one click away, so colour never carries meaning alone.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .triage import Outcome, TriageReport, classify, from_run_dir

#: Validated in both modes with the dataviz palette validator.
#: Light/dark pairs; the neutral is deliberately achromatic -- "not a result"
#: should read as an absence of colour, which is the one check it fails.
PALETTE = {
    "signal": ("#2a78d6", "#3987e5"),
    "agent": ("#e34948", "#e34948"),
    "excluded": ("#898781", "#898781"),
}

TIER_LABEL = {
    "signal": "Capability signal",
    "agent": "Agent bug",
    "excluded": "Our fault (excluded)",
}

TIER_OF: dict[Outcome, str] = {
    Outcome.VALID: "signal",
    Outcome.INFRA: "excluded",
    Outcome.TIMEOUT: "agent",
    Outcome.OOM: "agent",
    Outcome.CRASH: "agent",
    Outcome.NO_SUBMISSION: "agent",
    Outcome.INVALID_SUBMISSION: "agent",
}

#: Order for the breakdown chart: signal first, then agent bugs, then excluded.
OUTCOME_ORDER = (
    Outcome.VALID,
    Outcome.TIMEOUT,
    Outcome.OOM,
    Outcome.CRASH,
    Outcome.NO_SUBMISSION,
    Outcome.INVALID_SUBMISSION,
    Outcome.INFRA,
)


@dataclass
class RunRow:
    competition_id: str
    seed: int
    outcome: Outcome
    wall_clock_seconds: float = 0.0
    time_cap_seconds: float | None = None
    checkpoints: tuple[float, ...] = ()
    evidence: str = ""
    truncated: bool = False

    @property
    def tier(self) -> str:
        return TIER_OF[self.outcome]


@dataclass
class ReportData:
    title: str
    rows: list[RunRow] = field(default_factory=list)

    @property
    def competitions(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.rows:
            seen.setdefault(r.competition_id, None)
        return sorted(seen)

    @property
    def seeds(self) -> list[int]:
        return sorted({r.seed for r in self.rows})

    def by_key(self) -> dict[tuple[str, int], RunRow]:
        return {(r.competition_id, r.seed): r for r in self.rows}

    def tier_counts(self) -> dict[str, int]:
        out = {"signal": 0, "agent": 0, "excluded": 0}
        for r in self.rows:
            out[r.tier] += 1
        return out

    def outcome_counts(self) -> dict[Outcome, int]:
        out = {o: 0 for o in OUTCOME_ORDER}
        for r in self.rows:
            out[r.outcome] += 1
        return out

    @property
    def effective_denominator(self) -> int:
        return len(self.rows) - self.tier_counts()["excluded"]

    @property
    def max_time(self) -> float:
        caps = [r.time_cap_seconds for r in self.rows if r.time_cap_seconds]
        walls = [r.wall_clock_seconds for r in self.rows]
        return max(caps + walls + [1.0])


def collect(run_group: str | Path, title: str | None = None) -> ReportData:
    """Walk a run group, classify every run, and gather what the charts need."""
    root = Path(run_group)
    data = ReportData(title=title or root.name)
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        artifacts = from_run_dir(child)
        result = classify(artifacts)
        meta_path = child / "metadata.json"
        checkpoints: tuple[float, ...] = ()
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                checkpoints = tuple(
                    float(c["elapsed_seconds"]) for c in meta.get("checkpoints", [])
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                checkpoints = ()
        data.rows.append(
            RunRow(
                competition_id=result.competition_id,
                seed=result.seed,
                outcome=result.outcome,
                wall_clock_seconds=artifacts.wall_clock_seconds,
                time_cap_seconds=artifacts.time_cap_seconds,
                checkpoints=checkpoints,
                evidence=str(result.evidence[0]) if result.evidence else "",
                truncated=result.truncated,
            )
        )
    return data


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


# --- marks -------------------------------------------------------------------
# Shape is the secondary encoding: a filled circle, a filled diamond and a
# hollow ring stay distinguishable with colour removed entirely.

def _mark(tier: str, cx: float, cy: float, r: float = 7.0) -> str:
    fill = f"var(--tier-{tier})"
    if tier == "signal":
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'
    if tier == "agent":
        d = r * 1.15
        pts = f"{cx},{cy - d} {cx + d},{cy} {cx},{cy + d} {cx - d},{cy}"
        return f'<polygon points="{pts}" fill="{fill}"/>'
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r - 1}" fill="none" '
        f'stroke="{fill}" stroke-width="2"/>'
    )


def _eval_dots(data: ReportData) -> str:
    comps, seeds = data.competitions, data.seeds
    if not comps:
        return "<p class='empty'>No runs.</p>"
    lookup = data.by_key()
    label_w, cell, top = 300, 30, 30
    width = label_w + len(seeds) * cell + 16
    height = top + len(comps) * cell + 8

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="One dot per run, competitions down, seeds across">'
    ]
    for i, s in enumerate(seeds):
        x = label_w + i * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{top - 12}" class="tick mid" '
            f'text-anchor="middle">{_e(s)}</text>'
        )
    parts.append(
        f'<text x="{label_w - 10}" y="{top - 12}" class="tick mid" '
        f'text-anchor="end">seed</text>'
    )

    for j, comp in enumerate(comps):
        y = top + j * cell + cell / 2
        parts.append(
            f'<text x="{label_w - 14}" y="{y + 4}" class="tick" '
            f'text-anchor="end">{_e(comp)}</text>'
        )
        for i, s in enumerate(seeds):
            x = label_w + i * cell + cell / 2
            row = lookup.get((comp, s))
            if row is None:
                parts.append(
                    f'<circle cx="{x}" cy="{y}" r="2" fill="var(--muted)" '
                    f'opacity="0.4"/>'
                )
                continue
            tip = f"{comp} · seed {s}\\n{row.outcome.value}"
            if row.truncated:
                tip += " (truncated)"
            tip += f"\\n{_fmt_duration(row.wall_clock_seconds)}"
            if row.evidence:
                tip += f"\\n{row.evidence}"
            parts.append(
                f'<g class="dot" tabindex="0" data-tip="{_e(tip)}">'
                f'<rect x="{x - cell / 2}" y="{y - cell / 2}" width="{cell}" '
                f'height="{cell}" fill="transparent"/>'
                f"{_mark(row.tier, x, y)}</g>"
            )
    parts.append("</svg>")
    return "".join(parts)


def _breakdown(data: ReportData) -> str:
    counts = data.outcome_counts()
    total = max(len(data.rows), 1)
    present = [(o, c) for o, c in counts.items() if c]
    if not present:
        return ""
    rowh, barw, labelw = 28, 320, 190
    height = len(present) * rowh + 8
    width = labelw + barw + 56
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Run count by outcome">'
    ]
    for i, (outcome, count) in enumerate(present):
        y = i * rowh + 6
        w = max(count / total * barw, 2)
        tier = TIER_OF[outcome]
        parts.append(
            f'<text x="{labelw - 12}" y="{y + 14}" class="tick" '
            f'text-anchor="end">{_e(outcome.value)}</text>'
            f'<g class="dot" tabindex="0" '
            f'data-tip="{_e(outcome.value)}: {count} of {total} run(s) — '
            f'{_e(TIER_LABEL[tier])}">'
            f'<rect x="{labelw}" y="{y + 3}" width="{w}" height="14" rx="4" '
            f'fill="var(--tier-{tier})"/></g>'
            f'<text x="{labelw + w + 10}" y="{y + 14}" class="tick num">'
            f'{count}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _nice_ticks(tmax: float, count: int = 5) -> list[float]:
    """Round tick values, so an axis never reads "0s 2s 3s 5s 6s"."""
    if tmax <= 0:
        return [0.0]
    raw = tmax / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), 10 * mag)
    ticks, t = [], 0.0
    while t <= tmax + step * 0.001:
        ticks.append(t)
        t += step
    return ticks


#: Rows shown in the timeline before it becomes an unreadable wall. A full
#: 75-competition, 3-seed sweep is 225 runs; the interesting ones are those that
#: kept improving, so the list is sorted by last change and truncated.
TIMELINE_MAX_ROWS = 30


def _timeline(data: ReportData) -> tuple[str, int]:
    """When did each run last change its submission?

    Available without any grading: the checkpoint marks record when the file on
    disk changed. A run whose dots all cluster early plateaued; one still
    changing at the cap wanted more budget -- which is the distinction the
    anytime proposal exists to surface.
    """
    candidates = [r for r in data.rows if r.checkpoints]
    if not candidates:
        return "", 0
    # Latest last-change first: a run still improving at the cap is the finding.
    ordered = sorted(
        candidates,
        key=lambda r: (-max(r.checkpoints), r.competition_id, r.seed),
    )
    rows = ordered[:TIMELINE_MAX_ROWS]
    omitted = len(ordered) - len(rows)
    labelw, plotw, rowh = 300, 420, 24
    height = len(rows) * rowh + 40
    width = labelw + plotw + 60
    tmax = data.max_time

    def x_of(t: float) -> float:
        return labelw + (t / tmax) * plotw

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Submission change times per run">'
    ]
    for t in _nice_ticks(tmax):
        x = labelw + (t / tmax) * plotw
        parts.append(
            f'<line x1="{x}" y1="20" x2="{x}" y2="{height - 24}" '
            f'class="grid"/>'
            f'<text x="{x}" y="{height - 8}" class="tick mid" '
            f'text-anchor="middle">{_e(_fmt_duration(t))}</text>'
        )
    for i, r in enumerate(rows):
        y = 28 + i * rowh
        parts.append(
            f'<text x="{labelw - 14}" y="{y + 4}" class="tick" text-anchor="end">'
            f'{_e(r.competition_id)} · s{r.seed}</text>'
            f'<line x1="{labelw}" y1="{y}" x2="{x_of(r.wall_clock_seconds)}" '
            f'y2="{y}" class="rule"/>'
        )
        for t in r.checkpoints:
            parts.append(
                f'<g class="dot" tabindex="0" data-tip="submission changed by '
                f'{_e(_fmt_duration(t))}">'
                f'<circle cx="{x_of(t)}" cy="{y}" r="5" '
                f'fill="var(--tier-{r.tier})" stroke="var(--surface-1)" '
                f'stroke-width="2"/></g>'
            )
    parts.append("</svg>")
    return "".join(parts), omitted


def _table(data: ReportData) -> str:
    head = (
        "<tr><th>competition</th><th>seed</th><th>outcome</th><th>tier</th>"
        "<th>wall clock</th><th>evidence</th></tr>"
    )
    body = "".join(
        f"<tr><td>{_e(r.competition_id)}</td><td class='num'>{r.seed}</td>"
        f"<td>{_e(r.outcome.value)}</td><td>{_e(TIER_LABEL[r.tier])}</td>"
        f"<td class='num'>{_e(_fmt_duration(r.wall_clock_seconds))}</td>"
        f"<td>{_e(r.evidence)}</td></tr>"
        for r in sorted(data.rows, key=lambda r: (r.competition_id, r.seed))
    )
    return f"<table>{head}{body}</table>"


_CSS = """
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --rule:#c3c2b7; --border:rgba(11,11,11,0.10);
  --tier-signal:#2a78d6; --tier-agent:#e34948; --tier-excluded:#898781;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --rule:#383835; --border:rgba(255,255,255,0.10);
    --tier-signal:#3987e5; --tier-agent:#e34948; --tier-excluded:#898781;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --rule:#383835; --border:rgba(255,255,255,0.10);
  --tier-signal:#3987e5; --tier-agent:#e34948; --tier-excluded:#898781;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 72px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-0.01em}
.sub{color:var(--ink-2);margin:0 0 28px}
section{background:var(--surface-1);border:1px solid var(--border);
  border-radius:10px;padding:22px 24px;margin-bottom:20px}
h2{font-size:14px;margin:0 0 4px;letter-spacing:0.01em}
.note{color:var(--ink-2);margin:0 0 18px;max-width:62ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.tile{background:var(--surface-1);border:1px solid var(--border);
  border-radius:10px;padding:16px 18px}
.tile .v{font-size:30px;line-height:1.1;letter-spacing:-0.02em}
.tile .k{color:var(--ink-2);font-size:12px;margin-top:5px}
.scroll{overflow-x:auto;padding-bottom:4px}
.tick{fill:var(--ink-2);font-size:11px}
.tick.mid{fill:var(--muted)}
.num{font-variant-numeric:tabular-nums}
text.num{fill:var(--ink-2);font-size:11px;font-variant-numeric:tabular-nums}
.grid{stroke:var(--grid);stroke-width:1}
.rule{stroke:var(--rule);stroke-width:2;stroke-linecap:round}
.dot{cursor:default;outline:none}
.dot:hover,.dot:focus-visible{opacity:0.82}
.dot:focus-visible{outline:2px solid var(--ink);outline-offset:1px;border-radius:4px}
.legend{display:flex;flex-wrap:wrap;gap:20px;margin-top:18px;
  padding-top:16px;border-top:1px solid var(--border)}
.legend span{display:flex;align-items:center;gap:8px;color:var(--ink-2);font-size:12px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--border)}
th{color:var(--ink-2);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
details summary{cursor:pointer;color:var(--ink-2);font-size:13px}
.warn{margin-top:18px;padding:12px 14px;border-radius:8px;
  border:1px solid var(--border);background:var(--plane);color:var(--ink-2)}
.empty{color:var(--muted)}
#tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--ink);color:var(--surface-1);padding:7px 10px;border-radius:6px;
  font-size:12px;line-height:1.45;white-space:pre;z-index:9;max-width:340px}
"""

_JS = """
(function(){
  var tip=document.getElementById('tip');
  function show(e,t){tip.textContent=t;tip.style.opacity='1';move(e);}
  function move(e){
    var x=(e.clientX||0)+14, y=(e.clientY||0)+16, r=tip.getBoundingClientRect();
    if(x+r.width>innerWidth-8) x=innerWidth-r.width-8;
    if(y+r.height>innerHeight-8) y=(e.clientY||0)-r.height-12;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }
  function hide(){tip.style.opacity='0';}
  document.querySelectorAll('[data-tip]').forEach(function(el){
    var t=el.getAttribute('data-tip');
    el.addEventListener('mousemove',function(e){show(e,t);});
    el.addEventListener('mouseleave',hide);
    el.addEventListener('focus',function(){
      var b=el.getBoundingClientRect();
      show({clientX:b.left+b.width/2,clientY:b.top},t);
    });
    el.addEventListener('blur',hide);
  });
})();
"""


def render_html(data: ReportData) -> str:
    tiers = data.tier_counts()
    total = len(data.rows) or 1
    mech = tiers["agent"] / total

    legend = "".join(
        f'<span><svg width="18" height="18" aria-hidden="true">'
        f"{_mark(t, 9, 9, 6)}</svg>{_e(TIER_LABEL[t])} · {tiers[t]}</span>"
        for t in ("signal", "agent", "excluded")
    )

    warn = ""
    if mech > 0.15:
        warn = (
            f'<div class="warn"><strong>{mech:.0%} of runs failed mechanically '
            f"rather than on ML.</strong> The score is measuring plumbing, not "
            f"capability — fix these before reading anything into it.</div>"
        )

    timeline, omitted = _timeline(data)
    more = (
        f" Showing the {TIMELINE_MAX_ROWS} latest-changing runs; "
        f"{omitted} more are in the table below."
        if omitted
        else ""
    )
    timeline_section = (
        f"<section><h2>When did each run last change its submission?</h2>"
        f'<p class="note">Checkpoint marks, available without any grading. Dots '
        f"clustered early mean the run plateaued; a dot near the cap means it "
        f"still wanted budget.{_e(more)}</p>"
        f'<div class="scroll">{timeline}</div></section>'
        if timeline
        else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(data.title)} · mlea</title>
<style>{_CSS}</style></head>
<body><div id="tip" role="status" aria-live="polite"></div>
<div class="wrap">
<h1>{_e(data.title)}</h1>
<p class="sub">{len(data.rows)} run(s) · {len(data.competitions)} competition(s)
 · {len(data.seeds)} seed(s)</p>

<div class="tiles">
  <div class="tile"><div class="v">{tiers['signal']}</div>
    <div class="k">gradeable results (capability signal)</div></div>
  <div class="tile"><div class="v">{tiers['agent']}</div>
    <div class="k">agent bugs — real failures, not ML</div></div>
  <div class="tile"><div class="v">{tiers['excluded']}</div>
    <div class="k">our faults, excluded from the denominator</div></div>
  <div class="tile"><div class="v">{data.effective_denominator}
    <span style="font-size:16px;color:var(--ink-2)"> / {len(data.rows)}</span></div>
    <div class="k">effective denominator</div></div>
</div>

<section style="margin-top:20px">
<h2>Eval dots — one dot per run</h2>
<p class="note">Every run in the group. Shape and colour both carry the tier, so
the grid stays readable with colour removed; hover or focus a dot for the exact
outcome.</p>
<div class="scroll">{_eval_dots(data)}</div>
<div class="legend">{legend}</div>
{warn}
</section>

<section>
<h2>What the runs actually ended as</h2>
<p class="note">Only <code>valid</code> says anything about how good the agent is
at machine learning. Everything above it is a bug in something.</p>
<div class="scroll">{_breakdown(data)}</div>
</section>

{timeline_section}

<section>
<h2>Every run</h2>
<details><summary>Show table</summary>
<div class="scroll" style="margin-top:14px">{_table(data)}</div>
</details>
</section>
</div>
<script>{_JS}</script>
</body></html>
"""


def write_report(run_group: str | Path, out: str | Path, title: str | None = None) -> Path:
    data = collect(run_group, title)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(data), encoding="utf-8")
    return path


__all__ = [
    "OUTCOME_ORDER",
    "TIMELINE_MAX_ROWS",
    "PALETTE",
    "ReportData",
    "RunRow",
    "TIER_OF",
    "collect",
    "render_html",
    "write_report",
]
