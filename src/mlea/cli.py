"""Command line interface: ``mlea compare`` and ``mlea power``."""

from __future__ import annotations

import argparse
import sys
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
        import json

        blob = {
            "label": args.label or Path(args.run_group).name,
            "fingerprint": {"split_id": args.split_id},
            "runs": report.to_runset_records(),
        }
        Path(args.emit_runset).write_text(json.dumps(blob, indent=2))
        print(f"\nwrote run set -> {args.emit_runset}")
        print(
            "note: every run is recorded as no-medal. Fill in `any_medal` from "
            "`mlebench grade` before comparing."
        )
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

    t = sub.add_parser("triage", help="classify why each run in a run group ended")
    t.add_argument("run_group", help="run group directory (one subdir per competition)")
    t.add_argument("-v", "--verbose", action="store_true", help="list every run")
    t.add_argument("--emit-runset", help="write a run set JSON for `mlea compare`")
    t.add_argument("--split-id", default="unknown", help="split id for the run set")
    t.add_argument("--label", help="run set label (default: run group dir name)")
    t.set_defaults(func=_cmd_triage)

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
