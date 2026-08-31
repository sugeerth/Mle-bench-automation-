"""Multi-agent comparative dashboard.

:mod:`mlea.report` renders one run group. That answers "what happened in this
sweep" and cannot answer the question anyone actually has, which is "which agent
is better, by how much, and could this sweep have told me?"

Reads the layout ``mlea selftest`` and ``mlea probe`` produce::

    <session>/runs/<agent>/<competition>__seed<n>/
    <session>/grades/<agent>/grading_report.json    (optional)
    <session>/data/<competition>/competition.json   (optional -- oracle scores)

Form choices, per the data's job rather than per habit:

* Agent standing is *compare magnitude* -> horizontal bars, one hue, with the
  leader emphasised. Agents are nominal, so a value ramp would double-encode bar
  length as colour and burn the only free channel.
* Per-run outcome is *identity of a kind* -> the three-tier eval-dot marks,
  extended to an agent x competition matrix.
* Achieved-against-achievable is *a ratio against a limit* -> a meter per
  competition, with the medal threshold marked on the track.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .grade import leaderboard_percentile
from .metrics import get_metric
from .report import TIER_OF, _e, _fmt_duration, _mark, _CSS, _JS
from .skills import SkillProfile
from .skills import load as load_skills
from .triage import Outcome, TriageReport, classify, from_run_dir


@dataclass
class AgentRuns:
    label: str
    report: TriageReport
    #: (competition_id, seed) -> grading row
    grades: dict[tuple[str, int], dict] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.report.total

    @property
    def medal_rate(self) -> float:
        comps: dict[str, list[bool]] = {}
        for (cid, _), g in self.grades.items():
            comps.setdefault(cid, []).append(bool(g.get("any_medal")))
        if not comps:
            return 0.0
        return sum(sum(v) / len(v) for v in comps.values()) / len(comps)

    @property
    def valid_rate(self) -> float:
        return len(self.report.gradeable) / self.n if self.n else 0.0

    @property
    def mechanical_rate(self) -> float:
        return len(self.report.mechanical) / self.n if self.n else 0.0

    def outcome_at(self, cid: str, seed: int) -> Outcome | None:
        for r in self.report.results:
            if r.competition_id == cid and r.seed == seed:
                return r.outcome
        return None


@dataclass
class Session:
    title: str
    agents: list[AgentRuns] = field(default_factory=list)
    #: competition_id -> competition.json
    specs: dict[str, dict] = field(default_factory=dict)
    leaderboards: dict[str, list[float]] = field(default_factory=dict)
    skills: SkillProfile | None = None

    @property
    def competitions(self) -> list[str]:
        seen: dict[str, None] = {}
        for a in self.agents:
            for r in a.report.results:
                seen.setdefault(r.competition_id, None)
        return sorted(seen)

    @property
    def seeds(self) -> list[int]:
        return sorted({r.seed for a in self.agents for r in a.report.results})

    def mean_percentile(self, agent: AgentRuns) -> float | None:
        """Mean leaderboard percentile across competitions.

        Medal rate saturates: once an agent is good enough to medal, a better
        agent gets the same number. Percentile keeps discriminating, which is why
        it is the tiebreak and why it is shown next to every bar.
        """
        vals = [
            p for c in self.competitions
            if (p := self.percentile(agent, c)) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def ranked(self) -> list[AgentRuns]:
        return sorted(
            self.agents,
            key=lambda a: (-a.medal_rate, -(self.mean_percentile(a) or 0.0),
                           -a.valid_rate, a.label),
        )

    @property
    def medal_rate_is_saturated(self) -> bool:
        """True when two or more agents tie on medal rate but differ on percentile."""
        top = [a for a in self.agents if a.medal_rate == max(
            x.medal_rate for x in self.agents) and a.medal_rate > 0]
        if len(top) < 2:
            return False
        pcts = [self.mean_percentile(a) for a in top]
        pcts = [p for p in pcts if p is not None]
        return len(pcts) >= 2 and (max(pcts) - min(pcts)) > 0.01

    def percentile(self, agent: AgentRuns, cid: str) -> float | None:
        """Mean leaderboard percentile for one agent on one competition."""
        spec, lb = self.specs.get(cid), self.leaderboards.get(cid)
        if not spec or not lb:
            return None
        gib = get_metric(spec["metric"]).greater_is_better
        vals = [
            leaderboard_percentile(g["score"], lb, gib)
            for (c, _), g in agent.grades.items()
            if c == cid and g.get("score") is not None
        ]
        return sum(vals) / len(vals) if vals else None


def load_session(root: str | Path, title: str | None = None) -> Session:
    root = Path(root)
    session = Session(title=title or root.name)
    runs_root = root / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(f"{runs_root} does not exist")

    for agent_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        results = [
            classify(from_run_dir(child))
            for child in sorted(agent_dir.iterdir())
            if child.is_dir()
        ]
        if not results:
            continue
        grades: dict[tuple[str, int], dict] = {}
        gpath = root / "grades" / agent_dir.name / "grading_report.json"
        if gpath.exists():
            try:
                for row in json.loads(gpath.read_text()).get("reports", []):
                    grades[(row["competition_id"], int(row.get("seed", 0)))] = row
            except (OSError, json.JSONDecodeError, KeyError):
                grades = {}
        session.agents.append(AgentRuns(agent_dir.name, TriageReport(results), grades))

    skills_path = root / "skills.json"
    if skills_path.exists():
        try:
            session.skills = load_skills(skills_path)
        except (OSError, json.JSONDecodeError, TypeError):
            session.skills = None

    data_root = root / "data"
    if data_root.is_dir():
        for comp in sorted(p for p in data_root.iterdir() if p.is_dir()):
            spec_path, lb_path = comp / "competition.json", comp / "leaderboard.json"
            if spec_path.exists():
                try:
                    session.specs[comp.name] = json.loads(spec_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            if lb_path.exists():
                try:
                    session.leaderboards[comp.name] = json.loads(lb_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
    return session


# --- marks --------------------------------------------------------------------


def _standing(session: Session) -> str:
    """Horizontal bars, one hue, leader emphasised."""
    agents = session.ranked()
    if not agents:
        return ""
    rowh, barw, labelw = 30, 300, 118
    height = len(agents) * rowh + 26
    width = labelw + barw + 112
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Medal rate and mean percentile by agent">',
        f'<text x="{labelw + barw + 10}" y="12" class="tick mid">medal</text>'
        f'<text x="{labelw + barw + 56}" y="12" class="tick mid">pctile</text>',
    ]
    for i, a in enumerate(agents):
        y = i * rowh + 20
        w = max(a.medal_rate * barw, 2)
        lead = i == 0 and a.medal_rate > 0
        fill = "var(--tier-signal)" if lead else "var(--dim)"
        pct = session.mean_percentile(a)
        pct_txt = "" if pct is None else f"{pct:.0%}"
        parts.append(
            f'<text x="{labelw - 12}" y="{y + 14}" class="tick" '
            f'text-anchor="end">{_e(a.label)}</text>'
            f'<rect x="{labelw}" y="{y + 2}" width="{barw}" height="15" rx="4" '
            f'fill="var(--track)"/>'
            f'<g class="dot" tabindex="0" data-tip="{_e(a.label)}: '
            f'{a.medal_rate:.0%} medal rate over {len(session.competitions)} '
            f'competition(s)&#10;mean leaderboard percentile {pct_txt or "n/a"}'
            f'&#10;{a.valid_rate:.0%} of runs gradeable">'
            f'<rect x="{labelw}" y="{y + 2}" width="{w}" height="15" rx="4" '
            f'fill="{fill}"/></g>'
            f'<text x="{labelw + barw + 10}" y="{y + 14}" class="tick num">'
            f'{a.medal_rate:.0%}</text>'
            f'<text x="{labelw + barw + 56}" y="{y + 14}" class="tick mid num">'
            f'{pct_txt}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _matrix(session: Session) -> str:
    """Agents down, competition x seed across. The comparative eval dots."""
    comps, seeds, agents = session.competitions, session.seeds, session.ranked()
    if not comps or not agents:
        return "<p class='empty'>No runs.</p>"
    cell, labelw, gap, top = 22, 118, 12, 46
    block = len(seeds) * cell + gap
    width = labelw + len(comps) * block + 20
    height = top + len(agents) * cell + 10
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Outcome per agent, competition and seed">'
    ]
    for ci, comp in enumerate(comps):
        x0 = labelw + ci * block
        short = comp.replace("synth-", "")
        parts.append(
            f'<text x="{x0 + (len(seeds) * cell) / 2}" y="{top - 26}" '
            f'class="tick mid" text-anchor="middle" '
            f'transform="rotate(-32 {x0 + (len(seeds) * cell) / 2} {top - 26})">'
            f"{_e(short)}</text>"
        )
    for ai, a in enumerate(agents):
        y = top + ai * cell + cell / 2
        parts.append(
            f'<text x="{labelw - 12}" y="{y + 4}" class="tick" '
            f'text-anchor="end">{_e(a.label)}</text>'
        )
        for ci, comp in enumerate(comps):
            for si, seed in enumerate(seeds):
                x = labelw + ci * block + si * cell + cell / 2
                outcome = a.outcome_at(comp, seed)
                if outcome is None:
                    parts.append(
                        f'<circle cx="{x}" cy="{y}" r="1.5" fill="var(--muted)" '
                        f'opacity="0.35"/>'
                    )
                    continue
                g = a.grades.get((comp, seed))
                tip = f"{a.label} · {comp} · seed {seed}\\n{outcome.value}"
                if g and g.get("score") is not None:
                    tip += f"\\nscore {g['score']:.4f}"
                    if g.get("any_medal"):
                        medal = ("gold" if g.get("gold_medal") else
                                 "silver" if g.get("silver_medal") else "bronze")
                        tip += f" · {medal}"
                parts.append(
                    f'<g class="dot" tabindex="0" data-tip="{_e(tip)}">'
                    f'<rect x="{x - cell / 2}" y="{y - cell / 2}" width="{cell}" '
                    f'height="{cell}" fill="transparent"/>'
                    f"{_mark(TIER_OF[outcome], x, y, 5.5)}</g>"
                )
    parts.append("</svg>")
    return "".join(parts)


def _headroom(session: Session) -> str:
    """One meter per competition: how much of the achievable was achieved."""
    agents = session.ranked()
    comps = [c for c in session.competitions if c in session.specs]
    if not agents or not comps:
        return ""
    best = agents[0]
    rowh, barw, labelw = 28, 300, 190
    rows = [(c, session.percentile(best, c)) for c in comps]
    rows = [(c, p) for c, p in rows if p is not None]
    if not rows:
        return ""
    height = len(rows) * rowh + 10
    width = labelw + barw + 96
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Leaderboard percentile reached per competition">'
    ]
    for i, (comp, pct) in enumerate(rows):
        y = i * rowh + 7
        w = max(pct * barw, 2)
        spec = session.specs[comp]
        lb = session.leaderboards.get(comp, [])
        gold_pct = None
        if lb:
            gib = get_metric(spec["metric"]).greater_is_better
            gold_pct = leaderboard_percentile(spec["thresholds"]["gold"], lb, gib)
        parts.append(
            f'<text x="{labelw - 12}" y="{y + 14}" class="tick" text-anchor="end">'
            f'{_e(comp.replace("synth-", ""))}</text>'
            f'<rect x="{labelw}" y="{y + 3}" width="{barw}" height="14" rx="4" '
            f'fill="var(--track)"/>'
            f'<g class="dot" tabindex="0" data-tip="{_e(best.label)} on '
            f'{_e(comp)}&#10;beat {pct:.0%} of the {len(lb)}-team leaderboard">'
            f'<rect x="{labelw}" y="{y + 3}" width="{w}" height="14" rx="4" '
            f'fill="var(--tier-signal)"/></g>'
        )
        if gold_pct is not None:
            gx = labelw + gold_pct * barw
            parts.append(
                f'<line x1="{gx}" y1="{y}" x2="{gx}" y2="{y + 20}" '
                f'class="threshold"/>'
            )
        parts.append(
            f'<text x="{labelw + barw + 10}" y="{y + 14}" class="tick num">'
            f'{pct:.0%}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _div_fill(delta: float | None, broke: bool) -> tuple[str, str]:
    """Diverging scale: two hues that read as opposite, neutral at zero.

    Returns ``(fill, text-class)``. The strongest step is a *light* colour on the
    dark surface and a saturated one on the light surface, so theme ink drops
    below the 4.5:1 text floor on it in dark mode. Those cells therefore carry a
    fixed dark ink in both themes; every other step uses theme ink, which is
    already correct against its own surface.
    """
    if broke:
        return "var(--div-neg-3)", "cellv strong"
    if delta is None:
        return "var(--track)", "cellv"
    mag = abs(delta)
    if mag < 0.02:
        return "var(--div-zero)", "cellv"
    step = 1 if mag < 0.05 else 2 if mag < 0.20 else 3
    arm = "pos" if delta > 0 else "neg"
    return f"var(--div-{arm}-{step})", ("cellv strong" if step == 3 else "cellv")


def _skill_grid(profile: SkillProfile) -> str:
    """Agents down, pathologies across, cell = cost versus the matched control.

    Every cell carries its number as well as its colour: the value is the point,
    and colour alone would make the grid unreadable to a CVD reader and useless
    in print.
    """
    agents, challenges = profile.agents, profile.challenges
    if not agents or not challenges:
        return ""
    cw, ch_, labelw, top = 96, 34, 96, 30
    width = labelw + len(challenges) * cw + 10
    height = top + len(agents) * ch_ + 8
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Cost of each pathology per agent">'
    ]
    for ci, ch in enumerate(challenges):
        parts.append(
            f'<text x="{labelw + ci * cw + cw / 2}" y="{top - 11}" class="tick mid" '
            f'text-anchor="middle">{_e(ch)}</text>'
        )
    for ai, agent in enumerate(agents):
        y = top + ai * ch_
        parts.append(
            f'<text x="{labelw - 12}" y="{y + 21}" class="tick" '
            f'text-anchor="end">{_e(agent)}</text>'
        )
        for ci, chal in enumerate(challenges):
            cell = profile.cell(agent, chal)
            x = labelw + ci * cw
            delta = None if cell is None else cell.delta
            broke = bool(cell and cell.broke)
            label = ("BROKE" if broke else "—" if delta is None
                     else f"{delta:+.0%}")
            tip = f"{agent} · {chal}\n"
            if broke:
                tip += f"no gradeable submission: {cell.failure}"
            elif cell is not None and delta is not None:
                tip += (f"control {cell.control_percentile:.0%} → "
                        f"challenged {cell.challenged_percentile:.0%}\n"
                        f"cost {delta:+.0%} percentile points")
            fill, text_class = _div_fill(delta, broke)
            parts.append(
                f'<g class="dot" tabindex="0" data-tip="{_e(tip)}">'
                f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{ch_ - 4}" '
                f'rx="5" fill="{fill}"/>'
                f'<text x="{x + cw / 2}" y="{y + 21}" class="{text_class}" '
                f'text-anchor="middle">{_e(label)}</text></g>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _table(session: Session) -> str:
    head = ("<tr><th>agent</th><th class='num'>medal rate</th>"
            "<th class='num'>mean pctile</th>"
            "<th class='num'>gradeable</th><th class='num'>agent bugs</th>"
            "<th class='num'>our faults</th><th class='num'>runs</th></tr>")

    def pct(a):
        v = session.mean_percentile(a)
        return "—" if v is None else f"{v:.1%}"

    body = "".join(
        f"<tr><td>{_e(a.label)}</td>"
        f"<td class='num'>{a.medal_rate:.0%}</td>"
        f"<td class='num'>{pct(a)}</td>"
        f"<td class='num'>{len(a.report.gradeable)}</td>"
        f"<td class='num'>{len(a.report.mechanical)}</td>"
        f"<td class='num'>{len(a.report.infra)}</td>"
        f"<td class='num'>{a.n}</td></tr>"
        for a in session.ranked()
    )
    return f"<table>{head}{body}</table>"


EXTRA_CSS = """
.dim{color:var(--ink-2)}
.threshold{stroke:var(--ink);stroke-width:1.5;opacity:.55}
.hero{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin:0 0 6px}
.hero .v{font-size:44px;line-height:1;letter-spacing:-.03em;font-weight:600}
.hero .who{font-size:15px;color:var(--ink-2)}
.split{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}
.keyline{display:flex;gap:8px;align-items:center;font-size:12px;color:var(--ink-2);
  margin-top:12px}
.keyline i{display:inline-block;width:2px;height:13px;background:var(--ink);
  opacity:.55}
text.cellv{fill:var(--ink);font-size:11.5px;font-variant-numeric:tabular-nums;
  font-weight:500}
/* The strongest diverging step is light on dark and saturated on light, so it
   needs a fixed dark ink to clear 4.5:1 in both themes. */
text.cellv.strong{fill:#101318}
"""


def render_dashboard(session: Session, comparison_note: str = "") -> str:
    agents = session.ranked()
    if not agents:
        return "<!doctype html><title>empty</title><p>No runs.</p>"
    best = agents[0]
    total_runs = sum(a.n for a in agents)
    worst_mech = max(agents, key=lambda a: a.mechanical_rate)

    legend = "".join(
        f'<span class="lg"><svg width="16" height="16" aria-hidden="true">'
        f"{_mark(t, 8, 8, 5.5)}</svg>{label}</span>"
        for t, label in (
            ("signal", "gradeable result"),
            ("agent", "agent bug"),
            ("excluded", "our fault"),
        )
    )
    saturation_note = ""
    if session.medal_rate_is_saturated:
        tied = [a for a in agents if a.medal_rate == agents[0].medal_rate]
        spread = [session.mean_percentile(a) for a in tied]
        spread = [p for p in spread if p is not None]
        saturation_note = (
            f'<div class="warn"><strong>Medal rate has saturated here.</strong> '
            f'{len(tied)} agents tie at {agents[0].medal_rate:.0%}, but their mean '
            f'leaderboard percentile spans {min(spread):.0%}–{max(spread):.0%}. '
            f'Any-medal is binary: once an agent is good enough to medal, a better '
            f'one gets the same number. Rank on percentile when the field is '
            f'strong.</div>'
        )

    skills_section = ""
    if session.skills is not None and session.skills.cells:
        p = session.skills
        rows = "".join(
            f"<tr><td>{_e(a)}</td><td class='num'>{p.robustness(a):+.0%}</td>"
            f"<td>{_e(p.hardest_for(a) or '—')}</td></tr>"
            for a in sorted(p.agents, key=lambda a: -p.robustness(a))
        )
        verdict = (
            "<p class='note'><strong>No agent dominates.</strong> Different "
            "pathologies reward different competences, so a single headline score "
            "cannot say what an agent is missing — which is the argument for "
            "profiling rather than ranking.</p>"
            if p.no_agent_dominates() else
            "<p class='note'>One agent leads on every pathology, so for this field "
            "a single score would have sufficed.</p>"
        )
        skills_section = f"""<section>
<h2>Which competence is missing?</h2>
<p class="note">Each pathology is generated alongside an otherwise identical clean
control — same seed, same latent function — so the difference isolates one skill
rather than reporting an aggregate that hides which one is absent. Blue is better
than the control, red is worse, and <strong>BROKE</strong> means no gradeable
submission at all.</p>
<div class="scroll">{_skill_grid(p)}</div>
{verdict}
<div class="scroll" style="margin-top:16px"><table>
<tr><th>agent</th><th class="num">robustness</th><th>weakest skill</th></tr>
{rows}</table></div>
</section>"""

    headroom = _headroom(session)
    headroom_section = (
        f"<section><h2>How much of the achievable did {_e(best.label)} reach?</h2>"
        f'<p class="note">Leaderboard percentile per competition — the fraction of '
        f"the simulated field it beat. The vertical rule marks the gold threshold. "
        f"Percentile rather than raw score, because an AUC gap and an RMSE gap are "
        f"different units pointing opposite ways.</p>"
        f'<div class="scroll">{headroom}</div>'
        f'<div class="keyline"><i></i> gold threshold</div></section>'
        if headroom else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(session.title)} · mlea</title>
<style>{_CSS}{EXTRA_CSS}</style></head>
<body><div id="tip" role="status" aria-live="polite"></div>
<div class="wrap">
<h1>{_e(session.title)}</h1>
<p class="sub">{len(agents)} agent(s) · {len(session.competitions)} competition(s)
 · {len(session.seeds)} seed(s) · {total_runs} runs</p>

<section>
  <div class="hero"><span class="v">{best.medal_rate:.0%}</span>
    <span class="who">medal rate — <strong>{_e(best.label)}</strong>, the leader
    of {len(agents)}</span></div>
  <p class="note">Ranked by medal rate, then by mean leaderboard percentile. A bar
  is only meaningful to the extent the agent produced results rather than failures
  — the matrix below is the check on that.</p>
  <div class="scroll">{_standing(session)}</div>
  {saturation_note}
</section>

<section>
<h2>Every run, every agent</h2>
<p class="note">Agents down, competitions across, one mark per seed. Shape and
colour both carry the tier, so the grid survives having colour removed. Hover any
mark for its outcome and score.</p>
<div class="scroll">{_matrix(session)}</div>
<div class="legend">{legend}</div>
{f'<div class="warn"><strong>{worst_mech.mechanical_rate:.0%} of {_e(worst_mech.label)} runs failed mechanically rather than on ML.</strong> For that agent the score is measuring plumbing, not capability.</div>' if worst_mech.mechanical_rate > 0.15 else ''}
</section>

{skills_section}

{headroom_section}

{f'<section><h2>Is the difference real?</h2><pre class="cmp">{_e(comparison_note)}</pre></section>' if comparison_note else ''}

<section>
<h2>Standings</h2>
<div class="scroll">{_table(session)}</div>
</section>
</div>
<script>{_JS}</script>
</body></html>
"""


def write_dashboard(
    root: str | Path, out: str | Path, title: str | None = None,
    comparison_note: str = "",
) -> Path:
    session = load_session(root, title)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(session, comparison_note), encoding="utf-8")
    return path


__all__ = [
    "AgentRuns",
    "Session",
    "load_session",
    "render_dashboard",
    "write_dashboard",
]
