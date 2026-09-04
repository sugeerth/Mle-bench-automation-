# Proposal: MLE-bench-Live — a rolling, contamination-controlled split

**Status:** ⚠️ **WITHDRAWN as specified.** A feasibility pass against primary sources found
the design dominated by an approach someone has already run, resting on a premise that is
factually false, and yielding roughly one tenth the supply it assumed. The replacement is
[§ What to do instead](#what-to-do-instead) at the bottom. The body is kept because the
problem it identifies is real and unsolved — only the proposed solution is wrong.
**Depends on:** Phase 2 of [`PLAN.md`](PLAN.md).

---

## Why this was withdrawn

Five findings, each independently sufficient.

**1. The freshness premise is false.** The design filters on competition *close* date. But
Kaggle leaks continuously *during* a competition through public notebooks — MLE-bench itself
stores a `kernels.txt` per competition listing the top 50, **3,636 notebooks across 82
competitions**. A competition that ran January to June had five months of public
high-scoring baselines on the open web before it closed. Kaggle's rules additionally
*require* winners to publish a solution write-up within 14 days of close.

This is the precise point where the SWE-bench-Live precedent fails to transfer: a GitHub
issue's fix commit lands at a single instant and genuinely does not exist before it. A
Kaggle competition leaks across its whole run. Filtering on *launch* date fixes the logic
and roughly halves an already-thin supply.

**2. There is no correctness oracle — and this is the blocker.** Both precedents can decide
*automatically* whether a generated task is well-formed. SWE-bench-Live: valid iff the gold
patch flips FAIL_TO_PASS tests. MMBench-Live: the generator produces the answer alongside
the question. That is what makes 50 tasks/month and $30/refresh possible.

MLE-bench-Live has nothing equivalent. There is no ground truth for "did I re-split this
competition correctly and faithfully reimplement its metric?" — and the failure is *silent*:
a leaky split yields a valid-looking task on which agents score suspiciously well. Upstream
carries **8,323 lines of hand-written per-competition `prepare.py`** and 45 distinct metric
implementations, and its own Known Issues list catalogues ~8 competitions with preparation
bugs found by users after release, five of them test-label leakage created by the re-split.
That is the defect rate of an expert-built, hand-reviewed pipeline. The MMBench-Live cost
model prices generation-with-known-answers; this is reverse-engineering someone else's
evaluation, which is a different problem with a different cost curve.

**3. The people who built the pipeline never refreshed it.** The MLE-bench paper proposes
this exact idea as future work — "regularly update MLE-bench with new Kaggle competitions to
stay ahead of contamination issues." **Zero competitions have been added since the initial
commit on 2024-10-08**, 22 months. Only 3 of the repo's 59 commits touch
`mlebench/competitions/` at all, two of them a typo fix and a path fix. That is not
encouragement; it is a measurement of the cost.

**4. The arithmetic does not close.** Enumerating real closed competitions rather than
estimating: ~9–14 clearly eligible "real" competitions per year (excluding the synthetic
monthly Playground Series), falling to ~5–8 after launch-date filtering and ~3–6 after
preparability — call it **3–5 defensible matched pairs per year**. Against the 16-pair floor
in [`POWER-FINDINGS.md`](POWER-FINDINGS.md), that is **3–5 years of accumulation**, not the
6–12 months assumed. Padding with Playground and community competitions to reach 16 faster
destroys both the "real competition" claim and the difficulty matching.

The supply problem is also structural and worsening: roughly 35–40% of recent Featured
competitions are hackathons, judged submissions, agent games or reasoning prizes with no
supervised train/test structure at all. Kaggle in 2026 is not producing the product it
produced in 2018.

**5. Prior art, including a strictly better version.** **MLE-Live / CoMind** already exists —
the name is taken — and it submits to *live, ongoing* competitions, beating 92.6% of human
competitors on average. **MLE-Smith** already automates task ingestion from raw Kaggle
datasets (606 verified tasks, r=0.982 against human-designed tasks). **MLE-Dojo** scaled
coverage to 200+ tasks. **TML-bench** did contamination control by cutoff-date model
selection — and after all that work its post-cutoff split is **4 competitions, two of them
Playground Series and two community competitions**. That is the empirical existence proof of
what this approach yields.

### One more thing worth recording

The medal-threshold approximation is worse than upstream admits, and it is inherited by any
design built on it. `Grader.rank_score` takes a *raw score value* at a rank position on the
original leaderboard and compares the agent's score — computed on a different split — against
it, with no rank mapping or recalibration. Of the 82 preparation scripts, **69 use plain
`sklearn.train_test_split`; zero use stratification; zero are group- or time-aware.** The
Kaggle private test sets those thresholds came from very often *were* split by time, site, or
patient, deliberately. An i.i.d. split is systematically easier, so the approximation is
**biased upward**, not merely noisy. Both teams who built a Kaggle-derived benchmark fresh
(TML-bench, MLE-Dojo) declined to inherit medal thresholds.

---

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
~13 points of power across its whole plausible range, so this is a *bias* control, not a
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

⚠️ **An earlier draft costed this at 8 matched pairs ≈ 48 runs ≈ $7–9k. That design detects
essentially nothing.** Measured against real published per-competition rates rather than an
assumed distribution, its minimum detectable effect is **−90%**: it would miss a −30%
contamination effect 86% of the time, and miss even a full SWE-bench-sized −55% collapse 43%
of the time. See [`POWER-FINDINGS.md` §1](POWER-FINDINGS.md).

Revised: **16 matched pairs × 3 seeds ≈ 96 runs ≈ $14.4k** is the practical floor (MDE −29%),
and 24 pairs ≈ $21.6k is where moderate effects become detectable (MDE −21%). Marginal budget
buys pairs, not seeds — but never below 3 seeds, since 41% of competitions are non-unanimous
across seeds even for a single agent. The reduced time budgets the paper's own ablation justifies apply
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

## What to do instead

<a name="what-to-do-instead"></a>

**Submit to live, ongoing competitions.** It dominates the withdrawn design on every axis
that matters:

| | Rolling re-split of closed competitions | Live submission |
| --- | --- | --- |
| Contamination | partial — leaks during the run | **zero** — the solution does not exist yet |
| Medal thresholds | approximated across a mismatched split, biased upward | **real**, official private leaderboard |
| Preparation cost | ~100 LOC bespoke prep + a bespoke grader, unverifiable | **none** |
| Re-split error | inherited | **none** |
| Shots per year | 3–5 prepared pairs | 10–20 competitions |
| Repeatable | yes | **no** — one shot each |

Repeatability is the one thing lost, which sets the division of labour: **live submission for
the contamination measurement** (a binary, load-bearing question that needs answering once),
and **MLE-bench proper for the regression signal** (where repeatability is the entire point).
Pair each live result against the agent's medal rate on modality- and size-matched MLE-bench
competitions and you get the pre/post design with no ingest pipeline at all.

If a rolling split is still wanted later, take SWE-bench-Live's real lesson rather than its
headline: **freeze the comparable split and roll a separate one**; filter on **launch** date,
not close date; replace medal thresholds with a HumanRank-style relative position computed on
the original leaderboard, as MLE-Dojo does; and budget a human reviewer per competition,
because there is no oracle and pretending otherwise is how a fifth leaky test set ships.

### The cheapest first step is unchanged and still costs nothing

Re-run the paper's own obfuscation test with a current agent. The 2024 result (8.5% vs 8.4%)
was measured at the floor, where it had no power; at 65% it would finally be informative. Note
the constraint from [`POWER-FINDINGS.md` §7](POWER-FINDINGS.md): a paired test needs **at least
6 competitions** to return a significant result at all, so use 8–10, and treat a
non-significant result as uninformative rather than as evidence of no contamination.

---

## Sources

- [SWE-bench Goes Live! (2505.23419)](https://arxiv.org/abs/2505.23419) — post-cutoff gap, leakage analysis
- [MMBench-Live (2607.01813)](https://arxiv.org/abs/2607.01813) — refresh cost and cross-version comparability
- [MLE-bench paper (2410.07095)](https://arxiv.org/pdf/2410.07095) — obfuscation test, familiarity analysis, time ablation
- [MLEvolve](https://github.com/InternScience/MLEvolve) — current SOTA
