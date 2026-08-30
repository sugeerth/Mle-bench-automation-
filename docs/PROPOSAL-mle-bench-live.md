# Proposal: MLE-bench-Live — a rolling, contamination-controlled split

**Status:** proposal. Supersedes [`PROPOSAL-anytime-eval.md`](PROPOSAL-anytime-eval.md) as the
primary new direction.
**Depends on:** Phase 2 of [`PLAN.md`](PLAN.md).

---

## Why the previous proposal was the wrong thing to build

Before arguing for this one, the honest accounting on the last one.

I proposed anytime evaluation — grade each run at multiple checkpoints — with its headline
payoff being a measured early-stopping policy. Checking the literature afterwards:

- **The MLE-bench paper already ran the time ablation.** Extending the budget from 24 h to
  100 h moved medal rate from 8.7% → 11.8% on MLE-bench-30. Time scaling is weak and gains
  land early. That is the conclusion my proposal was designed to establish, and it is already
  published. You can act on early stopping today by citing it; no new instrumentation needed.
- **The paper already ran the contamination probe I "parked".** They rewrote competition
  descriptions to remove identifying information: 8.5% vs 8.4% medal rate. They also found no
  correlation between GPT-4o's familiarity with a competition and performance on it, and Dolos
  found no plagiarism.

So both of my ideas had already been done. The anytime work is still mildly useful as *triage*
instrumentation — curve shapes distinguish "ran out of clock" from "plateaued" from "overfit
itself" — but it is a cheap Phase 4 nicety, not a headline direction. It has been demoted
accordingly.

**But the second finding is the interesting one, because of *when* it was measured.**

---

## The actual problem

The MLE-bench contamination analysis was run in 2024, on agents scoring **8.5%**.

At 8.5% you are on the floor. Almost nothing is being solved, so there is almost nothing for
memorization to be inflating, and a test asking "does score drop when we remove identifying
information?" has close to zero statistical power. Finding 8.5% vs 8.4% under those conditions
is not evidence that contamination is absent — it is evidence that *the experiment could not
have detected it*.

Today SOTA is **65.3% on the full set and 80.3% on low complexity**. The same test, run now, is
finally informative. Nobody appears to have re-run it.

Meanwhile the benchmark faces two converging pressures:

1. **Saturation.** Low complexity is at 80.3%. At the current rate — 17% → 65% in under two
   years — the full set has maybe one to two years of headroom left.
2. **Contamination risk that grows with capability.** All 75 competitions are public, long-closed
   Kaggle competitions. Winning solutions, write-ups and notebooks are in every frontier model's
   training data. A medal is ambiguous evidence, and gets more ambiguous as models get better
   at recall.

### The neighbouring benchmark already ran this experiment

SWE-bench is the closest analogue, and the result there is stark. Frontier models sit at
**76–81% on SWE-bench Verified**. On **SWE-bench-Live** — the same task format, but built only
from GitHub issues filed after Jan 2024 — GPT-5 tops out at about **23.3%**. Their analysis
attributes roughly 32–33% of "successful" patches to direct solution leakage.

A static benchmark that looked healthy collapsed by ~55 points once the tasks were post-cutoff.
Whether MLE-bench has the same problem is currently **unknown**, and that is the single most
important unknown about any number this repository will ever produce.

---

## The proposal

**Build a continuously refreshed MLE-bench split from Kaggle competitions that closed after the
evaluated model's training cutoff, and report the pre-cutoff / post-cutoff gap.**

The pattern is established — [SWE-bench-Live](https://arxiv.org/abs/2505.23419),
[MMBench-Live](https://arxiv.org/abs/2607.01813), MMLU-CF — but **no MLE-bench equivalent
exists**. That is the gap.

### Why temporal holdout beats obfuscation

I previously proposed obfuscating existing competitions. Temporal holdout is strictly better:

- **No difficulty confound.** Mangling column names removes information a real practitioner
  would legitimately have, so obfuscation measures "can the agent work with unlabelled columns",
  not "did it memorize". A new competition is a genuinely new problem at natural difficulty.
- **It has already been tried and returned null.** Re-running a weak test is a poor use of $4k
  of compute.
- **It fixes saturation too.** New competitions keep arriving, including hard ones. A static
  benchmark can only be beaten once.
- **It is self-renewing.** The split refreshes on a schedule instead of needing curation.

### Why this repo, specifically

This is ~90% automation and ~10% research: poll Kaggle, filter, prepare, schedule, grade,
report. Every one of those is already a component in [`PLAN.md`](PLAN.md). MMBench-Live reports
~$30 and 1–2 h per refresh cycle for a comparable pipeline. **This is the most natural possible
project for an MLE-bench *automation* repository** — the anytime proposal was a measurement
tweak; this is infrastructure only an automation layer can provide.

---

## Design

```
Kaggle API poll (monthly)
   │  competitions that closed in the last N months
   ▼
Eligibility filter
   │  - train labels downloadable
   │  - supervised tabular / text / image
   │  - public leaderboard exists (for medal thresholds)
   │  - rules permit user-side download
   ▼
Auto-prepare  ── reuses upstream `mlebench prepare` re-split logic
   │            (new train/test carved from the public train set)
   ▼
live/YYYY-MM.txt  ── a split file, same format as experiments/splits/*.txt
   │
   ▼
Normal sweep ── same scheduler, same grader, same result store
   │
   ▼
Paired report: matched pre-cutoff vs post-cutoff competitions
                gap = contamination + staleness estimate
```

Four notes on the parts that carry risk:

**Matching protects validity, not power.** A raw pre/post gap confounds contamination with
difficulty drift — Kaggle competitions have changed character over the years. Post-cutoff
competitions must be matched to pre-cutoff ones on modality, dataset size, team count, and
metric type. [`POWER-FINDINGS.md` §2](POWER-FINDINGS.md) shows matching quality is worth only
~11 points of power across its whole plausible range, so this is a *bias* control, not a
variance control: if post-cutoff competitions are systematically harder, that is
indistinguishable from contamination and no number of extra pairs fixes it.

**Medal thresholds are an approximation.** Thresholds come from the real leaderboard, which was
computed on Kaggle's test split, not our re-split. Upstream already makes this approximation, so
we inherit rather than introduce the flaw — but it must be stated in every report.

**"Post-cutoff" is fuzzy.** Training cutoffs are often undisclosed and approximate, and
post-training data may include later material. Treat the gap as a lower bound on contamination,
never a point estimate.

**Recently-closed competitions leak fast.** Public notebooks and write-ups appear within weeks of
a competition closing. The freshness window is real but short, which makes the internet sandbox
non-negotiable for these runs — the one place where the Docker isolation we skipped in the free
tier genuinely matters.

---

## What we get

1. **A validity check on every other number this system produces.** If the gap is small,
   MLE-bench numbers mean what they claim and we can stop worrying. If it is SWE-bench-sized,
   that is the most important finding in the field this year and it changes how everyone reads
   the leaderboard.
2. **A benchmark that does not saturate.** Refreshes faster than agents improve.
3. **A genuine contribution rather than a private harness.** The split files and the ingest
   pipeline are publishable and useful to everyone running MLE-bench.

## Costs and risks

Ingest and preparation are cheap — CPU and bandwidth. The cost is the runs.

⚠️ **An earlier draft costed this at 8 matched pairs ≈ 48 runs ≈ $7–9k. That design does not
work.** Its minimum detectable effect is −54%, so it could detect a SWE-bench-sized catastrophe
and nothing else; a −30% contamination effect would be missed 73% of the time. See
[`POWER-FINDINGS.md` §1](POWER-FINDINGS.md).

Revised: **16 matched pairs × 3 seeds ≈ 96 runs ≈ $14.4k** is the practical floor (MDE −30%),
and 24 pairs ≈ $21.6k is where moderate effects become detectable (MDE −23%). Marginal budget
should buy pairs, not seeds. The reduced time budgets the paper's own ablation justifies apply
on top and cut these materially.

**The binding constraint turns out to be calendar, not money.** Each post-cutoff competition
supplies one pair and Kaggle yields ~15–30 substantive competitions a year, so 16 pairs is
6–12 months of accumulation. Build the ingest pipeline early and run the analysis late.

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Too few new competitions → small n | **High** | Quantified: 16 pairs minimum, ~6–12 months of Kaggle output. Start ingesting now, analyse later. `mlea power` gates the decision |
| Difficulty drift confounds the gap | **High** | Explicit matching + paired tests; report matched covariates. This is the risk matching actually addresses |
| Cutoff dates unknown | Medium | Report as a lower bound; use conservative cutoffs |
| Playground-series competitions inflate volume but are easier/synthetic | Medium | Tag separately, never mix into the headline split |
| Leaderboard threshold approximation | Low | Inherited from upstream; state it |

The small-n problem is the real one and should not be glossed: this produces a slow-moving,
wide-error-bar signal for the first year. It is still worth it, because the question it answers
is binary and load-bearing.

---

## Recommendation

Promote this to the flagship direction — Phase 5 in [`PLAN.md`](PLAN.md), replacing "scheduled
full split75 sweeps" as the thing worth building toward. Keep anytime checkpointing as cheap
Phase 4 triage instrumentation. Drop the obfuscation probe entirely; it was already run and
temporal holdout supersedes it.

**Cheapest first step, and it costs nothing:** re-run the paper's own obfuscation test with a
current agent. Note the design constraint from [`POWER-FINDINGS.md` §6](POWER-FINDINGS.md): a
paired test needs **at least 6 competitions** to be able to return a significant result at all,
so the "3 competitions" an earlier draft suggested cannot work no matter how large the effect.
Use 8–10 as a directional smoke test, and treat a non-significant result as uninformative
rather than as evidence of no contamination.

---

## Sources

- [SWE-bench Goes Live! (2505.23419)](https://arxiv.org/abs/2505.23419) — post-cutoff gap, leakage analysis
- [MMBench-Live (2607.01813)](https://arxiv.org/abs/2607.01813) — refresh cost and cross-version comparability
- [MLE-bench paper (2410.07095)](https://arxiv.org/pdf/2410.07095) — obfuscation test, familiarity analysis, time ablation
- [MLEvolve](https://github.com/InternScience/MLEvolve) — current SOTA
