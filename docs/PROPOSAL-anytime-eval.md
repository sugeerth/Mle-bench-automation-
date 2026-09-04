# Proposal: Anytime Evaluation for MLE-bench

**Status:** ⚠️ **DEMOTED.** Superseded as the primary direction by
[`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md). Retained as cheap Phase 4 triage
instrumentation only.
**Depends on:** Phase 2 of [`PLAN.md`](PLAN.md)

> **Why demoted.** Checking the literature after writing this: the MLE-bench paper already ran
> the time ablation (24 h → 100 h moves medal rate 8.7% → 11.8% on MLE-bench-30). Time scaling
> is weak and gains land early — which is precisely the conclusion this proposal was designed to
> establish. You can justify early stopping today by citing that result; no new instrumentation
> programme is needed for it. The paper likewise already ran the obfuscation contamination probe
> recorded at the bottom of this document (8.5% vs 8.4%).
>
> What survives is the *triage* value in §"What we get out of it" item 4 — curve shapes
> distinguishing "ran out of clock" from "plateaued" from "overfit itself" — which is genuinely
> useful and genuinely cheap. Build that; skip the rest. The cost-saving argument that motivated
> the whole thing is already settled literature.

---

## The problem with a single number at 24 hours

MLE-bench today asks one question: *after 24 hours on reference hardware, did the agent earn a
medal?* That yields one bit per competition-seed, harvested at enormous cost.

Three things go wrong as a result.

**It conflates capability with budget.** An agent that reaches a silver-medal solution in 90
minutes and an agent that reaches the same solution at hour 23 score identically. In every
practical setting — a research loop, a product, a cost line — those are very different agents.

**It hides the shape of the run.** Nothing in the output distinguishes *steady improvement
that ran out of clock* from *plateaued at hour 2 and burned 22 hours* from *had a good model
at hour 6 and then overfit itself into a worse one*. These call for completely different fixes,
and all three currently look like the same failed run.

**It makes early stopping unsafe.** §5 of the plan identifies "don't use the full 24 h" as the
largest available cost saving. But you cannot responsibly cut the budget when you have no
evidence about what happens in the hours you'd be cutting. The benchmark's own design is what
blocks its biggest efficiency win.

---

## The proposal

**Grade every run at multiple checkpoints along its timeline, not only at the end.**

At fixed wall-clock marks — say 15 min, 30 min, 1 h, 2 h, 4 h, 8 h, 16 h, 24 h — snapshot
whatever the agent currently considers its best submission, and grade all snapshots offline
after the run completes.

The output changes from a bit to a curve: **score as a function of elapsed time and dollars
spent**, per competition, per agent.

### Why this is cheap

The critical property: **grading is offline and nearly free relative to running.** Once the run
has happened, grading eight snapshots instead of one costs eight cheap CPU-seconds of
`mlebench grade-sample`. We are extracting roughly 8× the information from a run we already
paid ~$150 for, at negligible marginal cost.

This is what makes the idea worth doing. It is not a new expensive experiment; it is
instrumentation on an experiment we are already running.

### The one hard part

MLE-bench agents are told to write `submission.csv` and are free to overwrite it whenever. Some
write early and improve in place; some write only at the end. So "the agent's best submission at
time T" is not reliably on disk at time T.

Three options, in order of preference:

1. **Passive snapshotting.** A sidecar copies `submission.csv` (and its mtime) at each mark, if
   it exists. Zero agent changes, zero interference with the run. The cost is coverage: agents
   that write late produce sparse curves, and a snapshot reflects "what was on disk", not
   necessarily "the agent's best".
2. **Cooperative checkpointing.** Extend the agent instructions to ask for a valid
   `submission.csv` maintained at all times. Better data, but it **changes the task** and
   therefore breaks comparability with published MLE-bench numbers. Only acceptable as a
   clearly-labelled separate evaluation mode.
3. **Interrupt-and-ask.** Signal the agent at each mark to produce its current best. Highest
   fidelity, most invasive, most likely to perturb the very thing being measured.

**Recommendation: ship option 1 only.** It is strictly additive — a run with passive
snapshotting produces the exact same final number as a run without it, so headline results stay
comparable and nothing about the benchmark's meaning changes. Treat sparse curves as an honest
finding about agent behaviour rather than a defect to engineer around. Revisit option 2 only if
coverage turns out to be too poor to support the early-stopping decision, and if so, run it as
a separate labelled mode.

---

## What we get out of it

### 1. A defensible early-stopping policy

⚠️ *Largely obsolete — the MLE-bench paper's 24 h → 100 h ablation (8.7% → 11.8%) already
establishes that time scaling is weak. Kept for the reasoning, not as a justification to build.*

The direct payoff. If the curves show that the 95th percentile of last-improvement time is
hour 6, then a 8 h budget captures nearly all the signal at **one third the compute cost** —
and that claim is now backed by measurement rather than hope. Applied to a full 3-seed lite
sweep, that is roughly a $9–13k sweep dropping toward $3–5k.

Note the honest caveat: agents may behave differently when *told* they have 8 hours. The
early-stopping claim is about our internal iteration sweeps, where a slightly-biased but 3×
cheaper signal is a good trade. Published headline numbers stay at the reference 24 h.

### 2. Cost-Pareto comparison

Instead of "agent A scores 34%, agent B scores 31%", the report becomes a curve per agent. An
agent that reaches 28% in one hour is, for most purposes, more interesting than one reaching
34% in a day — and today's benchmark cannot express that at all.

### 3. Much better regression signal, at the same price

This partly solves the statistics problem in [`PLAN.md` §6](PLAN.md#6-comparing-two-agents-without-fooling-yourself).
Binary medal outcomes over 22 competitions are a low-information signal with wide error bars.
A per-competition *curve* is far richer: a change that shifts every curve leftward is clearly an
improvement even if it flips zero medals. More statistical power without more runs is exactly
what an expensive small-n benchmark needs.

### 4. A failure taxonomy that distinguishes causes

Curve shapes map onto the triage classes in [`PLAN.md` §7](PLAN.md#7-failure-triage) directly:

- rising at the cap → `timeout_mid_train`, genuinely needed more time
- flat from early on → plateaued; more budget wouldn't have helped
- rising then falling → self-inflicted regression, likely overfitting to its own validation
- empty until a late single point → agent doesn't checkpoint; a real robustness weakness,
  since any crash loses everything

### 5. Cheap crash recovery

A run that dies at hour 20 currently yields nothing. With snapshots it yields a full curve up
to hour 16. On spot instances — lever #2 in the cost model — this materially changes the
economics of preemption.

---

## Implementation sketch

Small, and mostly inside the worker:

1. **Sidecar in the run container.** A loop that, at each mark, copies `$SUBMISSION_DIR/submission.csv`
   to `checkpoints/t=<seconds>/submission.csv` with a manifest recording mtime and file hash.
   Skip if unchanged since the last mark — most marks will be no-ops and dedupe keeps storage
   trivial.
2. **Result store schema.** One `checkpoints` table: run_id, elapsed_seconds, content hash,
   and the estimated cost at that point.
3. **Grading fan-out.** After the run, `mlebench grade-sample` over each distinct checkpoint
   hash. Embarrassingly parallel, CPU-only, cache by hash so identical snapshots grade once.
4. **Reporting.** Per-competition score-vs-time curve; per-agent aggregate medal-rate-vs-cost
   Pareto frontier; a "time to first medal" and "time to last improvement" distribution.

Estimated effort: **~1 week** for the sidecar and grading fan-out, **~1 week** for reporting.
Deliberately sequenced after Phase 2 in the plan, because it needs the result store and the
per-run cost accounting to already exist.

### Risks

| Risk | Mitigation |
| --- | --- |
| Snapshot coverage too sparse to support early stopping | Measure coverage in Phase 1 before committing to Phase 4; fall back to labelled cooperative mode |
| Sidecar I/O perturbs the run | Copy is a few MB at 8 marks over 24 h; measure, but the prior is that it's negligible |
| Curves get over-interpreted with n=3 seeds | Same MDE discipline as §6 — publish uncertainty bands on curves, not bare lines |
| Storage growth from checkpoints | Content-addressed + skip-if-unchanged; submissions are small relative to the 3.3 TB dataset |

---

## Secondary idea: a contamination probe

⚠️ **Superseded.** This exact test was already run in the MLE-bench paper (rewriting competition
descriptions to remove identifying information: 8.5% vs 8.4%), and temporal holdout is a strictly
better instrument — see [`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md). The
interesting residual point is that the original test ran at an 8.5% medal rate, i.e. at the
floor, where it had almost no power to detect inflation. Kept below for the reasoning.

Recorded here because it's cheap to state and expensive to discover late; **not** proposed for
implementation yet.

Every MLE-bench competition is a *public, long-closed* Kaggle competition. Winning solutions,
write-ups, and notebooks are all in the training data of the models being evaluated. So a
medal is ambiguous evidence: it may reflect ML engineering capability, or it may reflect
recall of a public solution for a dataset the model has seen.

A tractable probe: build **obfuscated variants** of a handful of competitions — rename columns
to opaque identifiers, strip the competition name and framing from the task description, apply
a monotone transform to numeric features, shuffle row order. The underlying ML problem is
unchanged and a genuinely capable agent should score approximately the same. A large drop is
evidence of recall rather than capability.

Why it's only secondary: obfuscation is genuinely hard to do without changing task difficulty
(column semantics are legitimately useful information that a real practitioner would have), so
a naive version measures "can the agent work with unlabelled columns" instead of contamination.
It needs careful design and a control condition, and it should not block the main plan.

Rough cost: 3 competitions × 3 seeds × 2 conditions = 18 runs ≈ $2.5–4k. Worth doing eventually,
because if contamination is large it undermines the interpretation of every other number this
system produces.

---

## Recommendation (revised)

Build **only** the sidecar + grading fan-out, and only for its triage value — roughly one week,
not two, since the reporting can be a single curve-shape classifier rather than a cost-Pareto
analysis suite. Cite the paper's ablation for early stopping instead of re-deriving it.

Drop the obfuscation probe. The flagship direction is
[`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md).
