"""Command line interface: ``mlea compare`` and ``mlea power``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from dataclasses import replace
from pathlib import Path

from .compare import compare
from .power import (
    DESIGNS,
    Design,
    minimum_detectable_effect,
    power_for_effect,
    seeds_needed,
)
from .records import IncomparableError, RunSet, load_pairs
from .harness import (
    DEFAULT_CHECKPOINT_MARKS,
    CommandAgent,
    RunConfig,
    Task,
    resolve_competition_data_dir,
    run_one,
    run_sweep,
)
from .bench import SUITE, CompetitionSpec, make_suite
from .grade import grade_submission, leaderboard_percentile
from .metrics import get_metric
from .dashboard import write_dashboard
from .report import write_report
from .triage import triage_run_group

COST_PER_RUN_USD = 150.0
"""Rough cost of one 24 h reference-hardware run. See PLAN.md section 5 -- an
order-of-magnitude estimate, used only to put a design's price next to its power."""


def _cmd_compare(args: argparse.Namespace) -> int:
    a = RunSet.from_json(args.baseline)
    b = RunSet.from_json(args.candidate)
    pairs = load_pairs(args.pairs) if args.pairs else None
    try:
        result = compare(a, b, pairs=pairs, alpha=args.alpha, seed=args.seed)
    except IncomparableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(result.summary())
    if args.fail_on_regression and result.significant and result.difference < 0:
        print("\nregression gate: FAILED", file=sys.stderr)
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    competitions = args.competition
    if args.competition_set:
        # More lenient than upstream, which does a bare splitlines() and turns a
        # blank line or a comment into a competition id. A file that works here
        # may still break `mlebench prepare --list`; emit bare ids for that.
        competitions = [
            line.strip()
            for line in Path(args.competition_set).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not competitions:
        print("error: give --competition or --competition-set", file=sys.stderr)
        return 2

    tasks = [
        Task(c, resolve_competition_data_dir(data_root, c), seed=s)
        for c in competitions
        for s in range(args.seeds)
    ]
    marks = (
        tuple(float(m) for m in args.checkpoint_marks.split(","))
        if args.checkpoint_marks
        else DEFAULT_CHECKPOINT_MARKS
    )
    config = RunConfig(
        output_root=Path(args.out),
        time_cap_seconds=args.time_cap,
        isolation=args.isolation,
        checkpoint_marks=marks,
        force=args.force,
        submission_glob=args.submission_glob,
    )
    if config.isolation == "none":
        print(
            "WARNING: isolation=none. The agent runs with this process's "
            "privileges and full network access, so it can retrieve public "
            "solutions for the competition being graded. Fine for pipeline "
            "work; do not report these as benchmark numbers.\n",
            file=sys.stderr,
        )

    agent = CommandAgent(args.agent_name, args.agent_cmd)
    print(f"{len(tasks)} run(s): {len(competitions)} competition(s) x {args.seeds} seed(s)")

    def report(result) -> None:
        state = (
            "harness-error" if result.harness_error
            else "timeout" if result.timed_out
            else f"exit {result.exit_code}"
        )
        sub = "submission" if result.has_submission else "NO submission"
        ckpt = f", {len(result.checkpoints)} checkpoint(s)" if result.checkpoints else ""
        print(
            f"  {result.task.slug:<50} {state:<14} "
            f"{result.wall_clock_seconds:7.1f}s  {sub}{ckpt}"
        )

    results = run_sweep(agent, tasks, config, on_result=report)

    jsonl = config.output_root / "submissions.jsonl"
    n_sub = sum(1 for r in results if r.submission_path.exists())
    print()
    print(f"{n_sub}/{len(results)} run(s) produced a submission")
    # --output-dir is required by upstream, not optional; without it the
    # command exits on argparse. It also does not create parent directories.
    print(
        f"grade with : mlebench grade --submission {jsonl} "
        f"--output-dir {config.output_root / 'grades'}"
    )
    print(f"triage with: mlea triage {config.output_root}")
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    root = Path(args.session)
    if not (root / "runs").is_dir():
        print(f"error: {root / 'runs'} does not exist. A session directory holds "
              f"runs/<agent>/ and optionally grades/<agent>/ and data/.",
              file=sys.stderr)
        return 2
    note = Path(args.note).read_text() if args.note else ""
    path = write_dashboard(root, args.out, args.title, note)
    print(f"wrote {path}")
    return 0


def _cmd_conform(args: argparse.Namespace) -> int:
    """Grade the same submissions with our grader and the real one, and compare.

    This is the check that makes the package an MLE-bench harness rather than a
    harness for its own idea of a competition. For every real competition whose
    submission schema can be read, it generates data in that schema, has real
    agents produce submissions, and grades each one twice: once with
    ``mlea.grade`` and once with that competition's **own grader** loaded from
    upstream. A disagreement is a bug in this package.
    """
    from .bench import from_upstream, make_competition
    from .grade import grade_submission
    from . import upstream

    if not upstream.available():
        print(f"error: {upstream.INSTALL_HINT}", file=sys.stderr)
        return 2

    schemas = upstream.synthesisable_schemas()
    if args.competition:
        wanted = set(args.competition)
        schemas = [s for s in schemas if s.competition_id in wanted]
        missing = wanted - {s.competition_id for s in schemas}
        if missing:
            print(f"error: not synthesisable: {sorted(missing)}", file=sys.stderr)
            return 2
    schemas = schemas[: args.limit] if args.limit else schemas

    data = Path(args.out) / "data"
    print(f"checking {len(schemas)} real competition(s) against their own graders\n")
    print(f"   {'competition':44} {'strategy':9} {'mlea':>9} {'upstream':>9}  agree")

    disagreements: list[str] = []
    checked = 0
    for schema in schemas:
        spec = from_upstream(
            schema.competition_id, n_train=args.n_train, n_test=args.n_test,
            n_teams=60, difficulty=0.4, seed=args.seed,
        )
        comp = make_competition(spec, data)
        answers_csv = comp / "prepared" / "private" / "answers.csv"
        _write_answers_csv(comp, spec, answers_csv)

        for strategy in ("constant", "linear", "tuned", "broken"):
            sub = comp / f"sub-{strategy}.csv"
            env = dict(
                os.environ,
                DATA_DIR=str(comp / "prepared" / "public"),
                SUBMISSION_PATH=str(sub),
            )
            subprocess.run(
                [sys.executable, "-m", "mlea.baseline", "--strategy", strategy],
                env=env, capture_output=True, text=True, timeout=args.time_cap,
            )
            ours = grade_submission(sub, comp)
            theirs, err = (
                upstream.grade_with_upstream(sub, answers_csv, schema.competition_id)
                if sub.exists() else (None, "no submission")
            )
            checked += 1
            agree = (
                (ours.score is None and theirs is None)
                or (ours.score is not None and theirs is not None
                    and abs(ours.score - theirs) <= args.tolerance)
            )
            fmt = lambda v: "reject" if v is None else f"{v:.5f}"
            print(f"   {schema.competition_id:44} {strategy:9} "
                  f"{fmt(ours.score):>9} {fmt(theirs):>9}  "
                  f"{'yes' if agree else 'NO'}")
            if not agree:
                disagreements.append(
                    f"{schema.competition_id}/{strategy}: "
                    f"mlea={fmt(ours.score)} upstream={fmt(theirs)}"
                )

    print(f"\n{checked - len(disagreements)}/{checked} agreed "
          f"within {args.tolerance}")
    if disagreements:
        print("\nDISAGREEMENTS", file=sys.stderr)
        for d in disagreements:
            print(f"  {d}", file=sys.stderr)
        return 1
    print("\nCONFORMANT — this package's grader matches every real grader tested,")
    print("on both valid submissions and the ones a grader should reject.")
    return 0


def _write_answers_csv(comp: Path, spec, out: Path) -> None:
    """Upstream graders take an answers DataFrame; ours takes answers.json."""
    import csv as _csv

    answers = json.loads((comp / "prepared/private/answers.json").read_text())
    with out.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([spec.id_column, spec.target_column])
        for k, v in answers.items():
            w.writerow([k, int(v) if spec.task == "binary" else v])


def _cmd_skills(args: argparse.Namespace) -> int:
    """Profile which ML competences each agent has, with intervals."""
    from .bench import CHALLENGES, CompetitionSpec, make_competition
    from .skills import SkillProfile, design_floor, measure

    out = Path(args.out)
    data = out / "data"
    challenges = args.challenge or sorted(CHALLENGES)
    agents = args.agent or ["naive", "careful", "expert"]
    seeds = list(range(args.seed, args.seed + args.competitions))

    def spec_for(ch: str | None, seed: int) -> CompetitionSpec:
        name = f"skill-{ch or 'control'}-s{seed}"
        return CompetitionSpec(
            name, args.task, n_train=args.n_train, n_test=args.n_test,
            difficulty=args.difficulty, n_teams=args.n_teams, seed=seed,
            challenges=frozenset({ch}) if ch else frozenset(),
        )

    # One control per seed, shared across challenges: with the same seed and no
    # pathology they are the identical competition, so generating and running it
    # per challenge would only buy noise.
    for seed in seeds:
        make_competition(spec_for(None, seed), data)
        for ch in challenges:
            make_competition(spec_for(ch, seed), data)
    print(f"1. generated {len(seeds)} competition(s) x {len(challenges)} "
          f"pathology, each with a shared matched control")

    def run_and_score(agent: str, comp_id: str) -> tuple[float | None, str | None]:
        comp = data / comp_id
        task = Task(comp_id, resolve_competition_data_dir(data, comp_id), seed=0)
        cfg = RunConfig(output_root=out / "runs" / agent, force=True,
                        time_cap_seconds=args.time_cap, checkpoint_marks=())
        res = run_one(
            CommandAgent(agent, f"{sys.executable} -m mlea.baseline "
                                f"--strategy {agent}"), task, cfg)
        rep = grade_submission(res.submission_path, comp)
        if not rep.valid_submission:
            return None, rep.error or "no submission"
        lb = json.loads((comp / "leaderboard.json").read_text())
        spec = json.loads((comp / "competition.json").read_text())
        gib = get_metric(spec["metric"]).greater_is_better
        return leaderboard_percentile(rep.score, lb, gib), None

    profile = SkillProfile()
    for agent in agents:
        controls = {s: run_and_score(agent, spec_for(None, s).id) for s in seeds}
        for ch in challenges:
            pairs = []
            for seed in seeds:
                ctrl_pct, ctrl_err = controls[seed]
                chal_pct, chal_err = run_and_score(agent, spec_for(ch, seed).id)
                pairs.append((ctrl_pct, chal_pct, chal_err or ctrl_err))
            profile.cells.append(measure(agent, ch, pairs, alpha=args.alpha))
        worst = profile.hardest_for(agent)
        print(f"2. {agent:9} robustness {profile.robustness(agent):+7.1%}  "
              f"weakest: {worst or 'nothing measurable'}")

    (out / "skills.json").write_text(json.dumps(profile.to_dict(), indent=2))

    floor = design_floor(len(seeds))
    print(f"\n3. cost of each pathology, in leaderboard percentile points")
    print(f"   {len(seeds)} paired competition(s) per cell; 95% CI in brackets; "
          f"'ns' = not distinguishable from zero")
    if floor >= args.alpha:
        print(f"   !! {len(seeds)} pairs cannot reach p<{args.alpha} at any effect "
              f"size (floor {floor:.3f}). Raise --competitions.")
    for agent in agents:
        print(f"   {agent}")
        for ch in challenges:
            cell = profile.cell(agent, ch)
            if cell is None:
                continue
            if cell.broke:
                print(f"      {ch:10} BROKE on all {cell.n_broken}: {cell.failure}")
                continue
            extra = (f"  ({cell.n_broken} run(s) ungradeable)"
                     if cell.partially_broke else "")
            mark = "" if cell.significant else "   ns"
            print(f"      {ch:10} {cell.delta:+7.1%}  "
                  f"[{cell.ci_low:+.1%}, {cell.ci_high:+.1%}]  "
                  f"p={cell.p_value:.4f}{mark}{extra}")

    print()
    dominant = profile.dominant_agent()
    if dominant is None:
        best = {
            ch: max(
                (c for c in profile.cells
                 if c.challenge == ch and not c.broke and c.delta is not None),
                key=lambda c: c.delta, default=None)
            for ch in challenges
        }
        wins = ", ".join(
            f"{ch}: {c.agent}" for ch, c in best.items() if c is not None)
        print(f"   NO AGENT DOMINATES — {wins}")
        print("   Different pathologies reward different competences, so a single")
        print("   headline score cannot say what an agent is missing.")
    else:
        print(f"   {dominant.upper()} DOMINATES — it is not measurably beaten on any")
        print("   pathology, so for this field a single score would have sufficed.")
        print("   The profile still says WHERE the others lose, which a score cannot.")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    """Run the contamination probe, with both controls."""
    from .bench import (
        CLONE_TRANSFORMS,
        SUITE,
        clone_competition,
        clone_difficulty_delta,
        make_suite,
    )
    from .probe import ProbeCell, ProbeResult

    out = Path(args.out)
    data = out / "data"
    specs = SUITE[: args.competitions]
    make_suite(data, specs)

    print(f"1. generated {len(specs)} competition(s) and their clones")
    deltas: dict[str, list[float]] = {t: [] for t in CLONE_TRANSFORMS}
    for spec in specs:
        for t in CLONE_TRANSFORMS:
            c = clone_competition(data / spec.id, f"{spec.id}__{t}", data,
                                  transform=t, seed=hash(spec.id) % 10_000)
            deltas[t].append(clone_difficulty_delta(data / spec.id, c))
    for t in CLONE_TRANSFORMS:
        worst = max(abs(d) for d in deltas[t])
        print(f"   {t:8} difficulty delta: worst |{worst:.6f}| across "
              f"{len(deltas[t])} clone(s)")
        if worst > args.max_delta:
            print(f"   the {t} clones are not the same problem; probe invalid",
                  file=sys.stderr)
            return 1

    # Give the memoriser the originals' answers -- and only the originals'.
    # Absolute: the agent runs with its cwd set to its own code directory, so a
    # relative memory path silently resolves to nothing and every lookup misses.
    memory = (out / "memory").resolve()
    memory.mkdir(parents=True, exist_ok=True)
    from .baseline import row_fingerprints
    import csv as _csv

    for spec in specs:
        pub = data / spec.id / "prepared" / "public"
        with (pub / "test.csv").open(newline="") as fh:
            rows = list(_csv.reader(fh))
        header, body = rows[0][1:], rows[1:]
        X = np.array([[float(v) for v in r[1:]] for r in body])
        answers = json.loads(
            (data / spec.id / "prepared/private/answers.json").read_text())
        fps = row_fingerprints(header, X, args.memory_key)
        (memory / f"{spec.id}.json").write_text(
            json.dumps({fp: answers[r[0]] for fp, r in zip(fps, body)}))
    print(f"2. memorised {len(specs)} original(s), keyed on {args.memory_key}")

    cells: list[ProbeCell] = []
    agents = {"memorizer": "memorizer", "honest": "tuned"}
    targets = [(s.id, False, None) for s in specs] + [
        (f"{s.id}__{t}", True, t) for s in specs for t in CLONE_TRANSFORMS
    ]
    for label, strategy in agents.items():
        tasks = [
            Task(cid, resolve_competition_data_dir(data, cid), seed=0)
            for cid, _, _ in targets
        ]
        cfg = RunConfig(output_root=out / "runs" / label,
                        time_cap_seconds=args.time_cap, checkpoint_marks=(), force=True)
        agent = CommandAgent(
            label,
            f"{sys.executable} -m mlea.baseline --strategy {strategy} "
            f"--memory-dir {memory} --memory-key {args.memory_key}",
        )
        results = run_sweep(agent, tasks, cfg)
        for (cid, is_clone, transform), res in zip(targets, results):
            rep = grade_submission(res.submission_path, data / cid)
            log = (res.run_dir / "logs" / "agent.log")
            recalled = "RECALLED" in (log.read_text() if log.exists() else "")
            pct = None
            if rep.valid_submission:
                lb = json.loads((data / cid / "leaderboard.json").read_text())
                spec = json.loads((data / cid / "competition.json").read_text())
                pct = leaderboard_percentile(
                    rep.score, lb, get_metric(spec["metric"]).greater_is_better)
            cells.append(ProbeCell(label, cid, is_clone, transform, rep.score,
                                   pct, rep.any_medal, rep.valid_submission,
                                   recalled))
        print(f"3. ran {label} on {len(targets)} competition(s)")

    result = ProbeResult(cells)
    (out / "probe.json").write_text(json.dumps(result.to_dict(), indent=2))

    print("\n4. gap = original minus clone, in leaderboard percentile points "
          "(positive means it did better on what it had seen)")
    print(f"   {'agent':10} {'seen the original':>18}")
    for label in agents:
        print(f"   {label:10} {result.recall_rate(label, None):17.0%}")
    print(f"\n   {'agent':10} {'transform':10} {'still recalled':>15} "
          f"{'pctile gap':>12} {'medal gap':>11}")
    verdicts = []
    for label in agents:
        for t in CLONE_TRANSFORMS:
            gap = result.gap(label, t)
            mgap = result.medal_gap(label, t)
            rr = result.recall_rate(label, t)
            print(f"   {label:10} {t:10} {rr:14.0%} "
                  f"{'  n/a' if gap is None else f'{gap:+11.1%}'} "
                  f"{'  n/a' if mgap is None else f'{mgap:+10.0%}'}")
            verdicts.append((label, t, gap, rr))

    print("\n5. what the controls say")
    problems = []

    def check(ok: bool, msg: str) -> None:
        print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
        if not ok:
            problems.append(msg)

    honest_gaps = [abs(g) for l, t, g, _ in verdicts if l == "honest" and g is not None]
    check(
        bool(honest_gaps) and max(honest_gaps) < 0.10,
        f"negative control: an honest solver shows no gap "
        f"(worst |{max(honest_gaps):.1%}|)" if honest_gaps else
        "negative control: an honest solver shows no gap",
    )
    mem_rescale = result.gap("memorizer", "rescale")
    honest_rescale = result.gap("honest", "rescale")
    separated = (
        mem_rescale is not None
        and honest_rescale is not None
        and mem_rescale - honest_rescale > 0.05
    )
    check(
        separated,
        f"positive control: rescale separates the memoriser from an honest "
        f"solver ({mem_rescale:+.1%} vs {honest_rescale:+.1%})"
        if mem_rescale is not None and honest_rescale is not None
        else "positive control: rescale separates the memoriser",
    )
    mem_medal = result.medal_gap("memorizer", "rescale")
    if mem_medal is not None and mem_rescale is not None:
        if abs(mem_medal) < 1e-9:
            print(
                f"\n   FINDING: the memoriser is worth {mem_rescale:+.1%} in "
                f"percentile but {mem_medal:+.0%} in medal rate.\n"
                f"   Any-medal is binary and saturates: an agent good enough to "
                f"medal without recall\n   and one that recalls perfectly are the "
                f"same number. So MLE-bench's own headline\n   metric cannot see a "
                f"contamination effect that does not flip a medal -- and the "
                f"stronger\n   agents get, the more effects fall into that blind "
                f"spot. A probe must score on\n   percentile or raw score, never on "
                f"medal rate."
            )
        else:
            print(
                f"\n   note: percentile gap {mem_rescale:+.1%} vs medal gap "
                f"{mem_medal:+.0%}. Both compress near the top of a\n   crowded "
                f"leaderboard, so a contamination effect is hardest to see exactly "
                f"where\n   agents are strongest."
            )
    check(
        result.recall_rate("memorizer", None) > 0.9,
        f"the memoriser does recall what it was shown "
        f"({result.recall_rate('memorizer', None):.0%} of originals)",
    )
    relabel_recall = result.recall_rate("memorizer", "relabel")
    check(
        relabel_recall > 0.5,
        f"renaming and reordering alone does NOT defeat value-keyed recall "
        f"({relabel_recall:.0%} of relabel clones still recalled)",
    )
    check(
        result.recall_rate("memorizer", "rescale") < 0.1,
        f"altering the values does defeat it "
        f"({result.recall_rate('memorizer', 'rescale'):.0%} of rescale clones recalled)",
    )
    if problems:
        print(f"\nPROBE FAILED: {len(problems)} control(s)", file=sys.stderr)
        return 1
    print("\nPROBE VALID — it detects memorisation when memorisation is present")
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    """Exercise the whole pipeline against real, gradeable competitions.

    Every other test in this repository runs against stub agents or synthetic
    fixtures. This one generates competitions, fits real models, grades real
    scores, classifies real failures and renders the result -- and fails loudly
    if any stage disagrees with the others.
    """
    from .bench import SUITE, make_suite
    from .compare import compare
    from .records import RunSet
    from .triage import Outcome, triage_run_group

    out = Path(args.out)
    data = out / "data"
    specs = SUITE[: args.competitions]
    make_suite(data, specs)
    print(f"1. generated {len(specs)} competition(s)")

    strategies = ["tuned", "linear", "constant", "broken", "silent", "crash", "hungry"]
    marks = (1.0, 2.0, 4.0)
    results: dict[str, dict] = {}
    for strategy in strategies:
        tasks = [
            Task(spec.id, resolve_competition_data_dir(data, spec.id), seed=seed)
            for spec in specs
            for seed in range(args.seeds)
        ]
        cfg = RunConfig(
            output_root=out / "runs" / strategy,
            time_cap_seconds=args.time_cap,
            checkpoint_marks=marks,
            force=True,
        )
        think = "--think-seconds 0.15 " if strategy == "tuned" else ""
        agent = CommandAgent(
            strategy,
            f"{sys.executable} -m mlea.baseline {think}--strategy {strategy}",
        )
        run_sweep(agent, tasks, cfg)

        grade_dir = out / "grades" / strategy
        rc = _cmd_grade(
            argparse.Namespace(
                submission=cfg.output_root / "submissions.jsonl",
                data_root=data,
                output_dir=grade_dir,
                quiet=True,
                func=None,
            )
        )
        if rc != 0:
            return rc
        # Triage AFTER grading, so the grader's rejections are visible.
        report = triage_run_group(cfg.output_root)
        raw = json.loads((grade_dir / "medals.json").read_text())
        medals = {
            (k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in raw.items()
        }
        runset_path = out / f"{strategy}.json"
        runset_path.write_text(
            json.dumps(
                {
                    "label": strategy,
                    "fingerprint": {"split_id": "synth", "container_config": "none"},
                    "runs": report.to_runset_records(medals),
                },
                indent=2,
            )
        )
        rs = RunSet.from_json(runset_path)
        results[strategy] = {
            "report": report,
            "runset": rs,
            "medal_rate": rs.any_medal_rate() if rs.competitions() else 0.0,
        }
        print(
            f"2. {strategy:9} medal rate {results[strategy]['medal_rate']:6.1%}  "
            f"gradeable {len(report.gradeable)}/{report.total}"
        )

    print("\n3. checking the pipeline agrees with itself")
    problems: list[str] = []

    def check(ok: bool, msg: str) -> None:
        print(f"   {'PASS' if ok else 'FAIL'}  {msg}")
        if not ok:
            problems.append(msg)

    check(
        results["tuned"]["medal_rate"] > results["constant"]["medal_rate"],
        "a real model out-medals a constant baseline",
    )
    check(
        results["constant"]["medal_rate"] == 0.0,
        "a constant baseline earns no medals",
    )
    for strategy, expected in (
        ("broken", Outcome.INVALID_SUBMISSION),
        ("silent", Outcome.NO_SUBMISSION),
        ("crash", Outcome.CRASH),
        ("hungry", Outcome.OOM),
    ):
        outcomes = {r.outcome for r in results[strategy]["report"].results}
        check(outcomes == {expected}, f"{strategy} classifies as {expected.value}")
    check(
        all(
            r.outcome is Outcome.VALID
            for r in results["tuned"]["report"].results
        ),
        "every tuned run is a gradeable result",
    )
    curves = sum(
        1
        for d in (out / "runs" / "tuned").iterdir()
        if d.is_dir() and (d / "checkpoints").is_dir()
    )
    check(curves > 0, f"checkpoint curves captured for {curves} tuned run(s)")

    cmp = compare(results["constant"]["runset"], results["tuned"]["runset"])
    print(f"\n4. {cmp.summary()}")

    note_path = out / "comparison.txt"
    note_path.write_text(cmp.summary())
    report_path = write_report(
        out / "runs" / "tuned", out / "report.html", title="mlea selftest · tuned"
    )
    dash_path = write_dashboard(
        out, out / "dashboard.html",
        title=f"mlea selftest · {len(strategies)} agents", comparison_note=cmp.summary(),
    )
    print(f"\n5. wrote {report_path}\n   wrote {dash_path}")

    if problems:
        print(f"\nSELFTEST FAILED: {len(problems)} check(s)", file=sys.stderr)
        return 1
    print("\nSELFTEST PASSED")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    specs = SUITE[: args.count] if args.count else SUITE
    paths = make_suite(args.out, specs)
    print(f"generated {len(paths)} competition(s) in {args.out}")
    for spec, path in zip(specs, paths):
        meta = json.loads((path / "competition.json").read_text())
        t = meta["thresholds"]
        print(
            f"  {spec.id:26} {spec.task:10} {spec.metric:8} "
            f"difficulty={spec.difficulty:.2f}  oracle={meta['oracle_score']:.4f}  "
            f"gold={t['gold']:.4f} median={t['median']:.4f} ({t['n_teams']} teams)"
        )
    return 0


def _cmd_grade(args: argparse.Namespace) -> int:
    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports, medals = [], {}
    for line in Path(args.submission).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        row = json.loads(line)
        rep = grade_submission(row["submission_path"], root / row["competition_id"])
        reports.append({**rep.to_dict(), "seed": row.get("seed", 0)})
        medals[f"{row['competition_id']}|{row.get('seed', 0)}"] = rep.any_medal
        # Feed the grader's verdict back into the run directory. Triage runs
        # before grading and cannot otherwise tell a well-formed submission from
        # one the grader refuses -- a file with the right rows and a wrong column
        # name looks valid on disk and is not.
        run_dir = row.get("run_dir")
        if run_dir and rep.submission_exists and not rep.valid_submission:
            meta_path = Path(run_dir) / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    meta["validation_error"] = rep.error
                    meta_path.write_text(json.dumps(meta, indent=2))
                except (OSError, json.JSONDecodeError):
                    pass

    graded = [r for r in reports if r["valid_submission"]]
    n_medal = sum(1 for r in reports if r["any_medal"])
    summary = {
        "n_submissions": len(reports),
        "n_valid": len(graded),
        "n_any_medal": n_medal,
        "any_medal_rate": n_medal / len(reports) if reports else 0.0,
        "reports": reports,
    }
    (out_dir / "grading_report.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "medals.json").write_text(json.dumps(medals, indent=2))
    if getattr(args, "quiet", False):
        return 0
    print(f"{len(graded)}/{len(reports)} valid, {n_medal} medal(s)")
    for r in reports:
        score = "-" if r["score"] is None else f"{r['score']:.4f}"
        medal = ("gold" if r["gold_medal"] else "silver" if r["silver_medal"]
                 else "bronze" if r["bronze_medal"]
                 else "above median" if r["above_median"] else "-")
        note = f"  ({r['error']})" if r["error"] else ""
        print(f"  {r['competition_id']:26} seed {r['seed']}  "
              f"score={score:>8}  {medal}{note}")
    print(f"\nwrote {out_dir / 'grading_report.json'}")
    return 0


def _cmd_triage(args: argparse.Namespace) -> int:
    report = triage_run_group(args.run_group)
    if not report.total:
        print(f"no run directories under {args.run_group}", file=sys.stderr)
        return 2
    if args.verbose:
        for r in report.results:
            print(r)
        print()
    print(report.summary())

    if args.emit_runset:
        medals = {}
        if args.grades:
            raw = json.loads(Path(args.grades).read_text())
            medals = {tuple(k.rsplit("|", 1)[:1] + [int(k.rsplit("|", 1)[1])]): v
                      for k, v in raw.items()}
        blob = {
            "label": args.label or Path(args.run_group).name,
            "fingerprint": {
                "split_id": args.split_id,
                # Carried from the harness so an unsandboxed run can never be
                # silently compared against a sandboxed one.
                "container_config": report.isolation(),
            },
            "runs": report.to_runset_records(medals),
        }
        Path(args.emit_runset).write_text(json.dumps(blob, indent=2))
        print(f"\nwrote run set -> {args.emit_runset}")
        if not args.grades:
            print(
                "note: every run is recorded as no-medal. Pass --grades "
                "<medals.json> from `mlea grade`, or fill in `any_medal` from "
                "`mlebench grade`, before comparing."
            )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    root = Path(args.run_group)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    path = write_report(root, args.out, args.title)
    print(f"wrote {path}")
    return 0


def _resolve_design(args: argparse.Namespace) -> Design:
    if args.design:
        if args.design not in DESIGNS:
            raise SystemExit(
                f"unknown design {args.design!r}; choose from {', '.join(DESIGNS)}"
            )
        design = DESIGNS[args.design]
    else:
        design = Design(
            name="custom",
            n_units=args.units,
            n_seeds=args.seeds,
            base_rate=args.base_rate,
            matching_sd=args.matching_sd,
        )
    overrides = {}
    if args.units is not None and args.design:
        overrides["n_units"] = args.units
    if args.seeds is not None and args.design:
        overrides["n_seeds"] = args.seeds
    if args.heterogeneity is not None:
        overrides["heterogeneity"] = args.heterogeneity
    return replace(design, **overrides) if overrides else design


def _cmd_power(args: argparse.Namespace) -> int:
    design = _resolve_design(args)
    cost = design.cost_runs() * COST_PER_RUN_USD
    print(f"design      : {design.name}")
    if design.description:
        print(f"              {design.description}")
    print(
        f"units       : {design.n_units}  seeds: {design.n_seeds}  "
        f"base rate: {design.base_rate:.1%}  heterogeneity: {design.heterogeneity}"
    )
    if design.matching_sd:
        print(f"matching sd : {design.matching_sd} (matched-pair design)")
    print(f"cost        : {design.cost_runs()} runs ~ ${cost:,.0f} at 24h reference hw")
    print()

    if args.effect is not None:
        res = power_for_effect(design, args.effect, alpha=args.alpha, seed=args.seed)
        print(res.summary())
        return 0

    print(f"minimum detectable effect (at {args.target_power:.0%} power, alpha={args.alpha}):")
    for direction in ("decrease", "increase"):
        mde = minimum_detectable_effect(
            design,
            direction=direction,
            target_power=args.target_power,
            alpha=args.alpha,
            seed=args.seed,
        )
        shown = "NONE at any effect size" if mde is None else f"{mde:+.1%}"
        print(f"  {direction:<9}: {shown}")
    print(
        "  (medal probability is capped at 1, so near a high base rate a drop is "
        "easier to\n   detect than an equal-sized gain. Regressions and "
        "contamination gaps are drops.)"
    )
    print()
    print("power curve (drops):")
    for eff in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55):
        res = power_for_effect(design, -eff, alpha=args.alpha, seed=args.seed)
        bar = "#" * int(round(res.power * 40))
        print(f"  {-eff:+5.0%}  {res.power:5.0%}  {bar}")
    print()
    print("Any observed difference smaller than the MDE is not evidence of a change.")
    return 0


def _cmd_seeds(args: argparse.Namespace) -> int:
    design = _resolve_design(args)
    n = seeds_needed(
        design,
        args.effect,
        target_power=args.target_power,
        alpha=args.alpha,
        max_seeds=args.max_seeds,
        seed=args.seed,
    )
    if n is None:
        print(
            f"No seed count up to {args.max_seeds} reaches {args.target_power:.0%} "
            f"power for a {args.effect:+.0%} effect with {design.n_units} units."
        )
        print("Seeds cannot fix too few competitions -- add units instead.")
        return 0
    d = replace(design, n_seeds=n)
    print(
        f"{n} seed(s) per unit reach {args.target_power:.0%} power for "
        f"{args.effect:+.0%} ({d.cost_runs()} runs "
        f"~ ${d.cost_runs() * COST_PER_RUN_USD:,.0f})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlea",
        description="Comparison and power tooling for MLE-bench sweeps.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compare", help="paired comparison of two run sets")
    c.add_argument("baseline", help="baseline run set JSON")
    c.add_argument("candidate", help="candidate run set JSON")
    c.add_argument("--pairs", help="matched competition pairs JSON (pre/post design)")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 on a statistically significant drop (for CI gating)",
    )
    c.set_defaults(func=_cmd_compare)

    r = sub.add_parser("run", help="run an agent against competitions")
    r.add_argument("--agent-cmd", required=True,
                   help="shell command; gets DATA_DIR, SUBMISSION_PATH, LOGS_DIR, "
                        "CODE_DIR, COMPETITION_ID, SEED, TIME_CAP_SECONDS in env "
                        "and as {placeholders}")
    r.add_argument("--agent-name", default="agent")
    r.add_argument("--data-root", required=True,
                   help="mlebench data dir (default ~/.cache/mle-bench/data). Each "
                        "competition resolves to <root>/<id>/prepared/public when "
                        "that exists, so the agent never sees prepared/private")
    r.add_argument("--competition", action="append", default=[],
                   help="competition id (repeatable)")
    r.add_argument("--competition-set", help="file of competition ids, one per line")
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--time-cap", type=float, default=86400.0, help="seconds per run")
    r.add_argument("--isolation", default="none", choices=["none", "docker"])
    r.add_argument("--checkpoint-marks", help="comma-separated seconds")
    r.add_argument("--out", required=True, help="run group output directory")
    r.add_argument("--submission-glob",
                   help="where the agent writes its own submission, relative to "
                        "CODE_DIR (e.g. 'workspaces/0-run/working/submission.csv' for "
                        "AIDE, 'runs/*/workspace/best_submission/submission.csv' for "
                        "MLEvolve). Mirrored at every checkpoint and at exit, so the "
                        "agent needs no polling loop of its own")
    r.add_argument("--force", action="store_true",
                   help="overwrite completed runs (discards recorded results)")
    r.set_defaults(func=_cmd_run)

    t = sub.add_parser("triage", help="classify why each run in a run group ended")
    t.add_argument("run_group", help="run group directory (one subdir per competition)")
    t.add_argument("-v", "--verbose", action="store_true", help="list every run")
    t.add_argument("--emit-runset", help="write a run set JSON for `mlea compare`")
    t.add_argument("--split-id", default="unknown", help="split id for the run set")
    t.add_argument("--label", help="run set label (default: run group dir name)")
    t.add_argument("--grades", help="medals.json from `mlea grade`, to fill in "
                                    "any_medal rather than recording all as false")
    t.set_defaults(func=_cmd_triage)

    b = sub.add_parser("bench", help="generate gradeable competitions, no Kaggle needed")
    b.add_argument("--out", required=True, help="data root to generate into")
    b.add_argument("--count", type=int, help=f"how many of the {len(SUITE)}-competition "
                                             "suite to generate")
    b.set_defaults(func=_cmd_bench)

    g = sub.add_parser("grade", help="grade submissions against generated competitions")
    g.add_argument("--submission", required=True, help="submissions.jsonl from `mlea run`")
    g.add_argument("--data-root", required=True)
    g.add_argument("--output-dir", required=True)
    g.set_defaults(func=_cmd_grade)

    st = sub.add_parser(
        "selftest", help="run the whole pipeline against generated competitions")
    st.add_argument("--out", default="selftest", help="working directory")
    st.add_argument("--competitions", type=int, default=len(SUITE))
    st.add_argument("--seeds", type=int, default=1)
    st.add_argument("--time-cap", type=float, default=120.0)
    st.set_defaults(func=_cmd_selftest)

    pr = sub.add_parser(
        "probe", help="contamination probe with positive and negative controls")
    pr.add_argument("--out", default="probe", help="working directory")
    pr.add_argument("--competitions", type=int, default=4)
    pr.add_argument("--memory-key", default="values", choices=("names", "values"))
    pr.add_argument("--max-delta", type=float, default=0.01,
                    help="largest clone difficulty difference the probe tolerates")
    pr.add_argument("--time-cap", type=float, default=180.0)
    pr.set_defaults(func=_cmd_probe)

    db = sub.add_parser(
        "dashboard", help="comparative dashboard across every agent in a session")
    db.add_argument("session", help="directory holding runs/<agent>/ (and optionally "
                                    "grades/<agent>/ and data/)")
    db.add_argument("-o", "--out", default="dashboard.html")
    db.add_argument("--title")
    db.add_argument("--note", help="text file whose contents are shown verbatim as "
                                   "the statistical comparison")
    db.set_defaults(func=_cmd_dashboard)

    cf = sub.add_parser(
        "conform",
        help="check our grader against real MLE-bench graders on real schemas")
    cf.add_argument("--out", default="conform")
    cf.add_argument("--competition", action="append",
                    help="repeatable; defaults to every synthesisable one")
    cf.add_argument("--limit", type=int)
    cf.add_argument("--n-train", type=int, default=600)
    cf.add_argument("--n-test", type=int, default=400)
    cf.add_argument("--seed", type=int, default=5)
    cf.add_argument("--tolerance", type=float, default=0.0,
                    help="both graders round to 5 decimals, so exact agreement "
                         "is the right bar")
    cf.add_argument("--time-cap", type=float, default=300.0)
    cf.set_defaults(func=_cmd_conform)

    sk = sub.add_parser("skills", help="profile which ML competences an agent has")
    sk.add_argument("--out", default="skills")
    sk.add_argument("--challenge", action="append",
                    help="repeatable; defaults to all")
    sk.add_argument("--agent", action="append", help="repeatable")
    sk.add_argument("--task", default="binary", choices=("binary", "regression"))
    sk.add_argument("--n-train", type=int, default=1500)
    sk.add_argument("--n-test", type=int, default=600)
    sk.add_argument("--difficulty", type=float, default=0.4)
    sk.add_argument("--n-teams", type=int, default=200)
    sk.add_argument("--seed", type=int, default=3, help="first competition seed")
    sk.add_argument("--competitions", type=int, default=8,
                    help="paired competitions per cell; below 6 no paired test "
                         "can reach p<0.05 at any effect size")
    sk.add_argument("--alpha", type=float, default=0.05)
    sk.add_argument("--time-cap", type=float, default=300.0)
    sk.set_defaults(func=_cmd_skills)

    rp = sub.add_parser("report", help="render a run group as an HTML report")
    rp.add_argument("run_group")
    rp.add_argument("-o", "--out", default="report.html")
    rp.add_argument("--title")
    rp.set_defaults(func=_cmd_report)

    def add_design_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--design", help=f"preset: {', '.join(DESIGNS)}")
        sp.add_argument("--units", type=int, help="number of competitions / pairs")
        sp.add_argument("--seeds", type=int, help="seeds per competition")
        sp.add_argument("--base-rate", type=float, default=0.653)
        sp.add_argument("--heterogeneity", type=float)
        sp.add_argument("--matching-sd", type=float, default=0.0)
        sp.add_argument("--alpha", type=float, default=0.05)
        sp.add_argument("--target-power", type=float, default=0.80)
        sp.add_argument("--seed", type=int, default=0)

    w = sub.add_parser("power", help="power / minimum detectable effect for a design")
    add_design_args(w)
    w.add_argument("--effect", type=float, help="power for this effect (else MDE)")
    w.set_defaults(func=_cmd_power)

    s = sub.add_parser("seeds", help="seeds needed to detect a given effect")
    add_design_args(s)
    s.add_argument("--effect", type=float, required=True)
    s.add_argument("--max-seeds", type=int, default=20)
    s.set_defaults(func=_cmd_seeds)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
