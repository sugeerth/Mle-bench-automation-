# MLE-bench Automation

Automation harness for running [MLE-bench](https://github.com/openai/mle-bench) (OpenAI's
Kaggle-competition benchmark for ML engineering agents) repeatably, cheaply, and as a
regression signal rather than a one-off leaderboard stunt.

## `mlea` — comparison and power tooling

The one implemented piece. It exists because every other component in the plan spends five
figures per sweep, and this is what decides whether a given sweep could detect the thing it is
being run to detect.

```bash
pip install -e .

# 1. run an agent against competitions
mlea run --agent-cmd 'aide data_dir=$DATA_DIR ...' \
         --data-root ~/.cache/mle-bench/data \
         --competition-set experiments/splits/low.txt \
         --seeds 3 --time-cap 14400 --out runs/nightly

# 2. grade with upstream (the harness emits the JSONL it wants)
mlebench grade --submission runs/nightly/submissions.jsonl

# 3. what is the score actually made of?
mlea triage runs/nightly -v --emit-runset nightly.json --split-id low

# 4. did anything really change? could it even have been detected?
mlea compare baseline.json nightly.json --fail-on-regression
mlea power --design lite-regression

# 5. look at it
mlea report runs/nightly -o report.html
```

### `mlea report` — eval dots

A self-contained HTML page, no build step and no CDN. The headline view is a grid with
**one dot per run**, competitions down and seeds across, answering at a glance the thing a
table of percentages hides: *is this score made of ML results, or of plumbing failures?*

Colour carries the three tiers this repo has argued for throughout — gradeable result,
agent bug, our fault — as **categorical identity, not status**, because they are kinds of
thing rather than good/bad states.

The palette was validated rather than eyeballed, and the first attempt failed: green vs.
orange measured **CVD ΔE 5.6 (protan)** and green vs. red **4.1 (deutan)**, so the chart's
single most important distinction would have been invisible to a red-green colourblind
reader. The shipped blue/red/neutral clears every gate in both light and dark. Each dot
also carries a distinct *shape* (filled circle / diamond / hollow ring) and a text label in
its tooltip, so colour is never the only channel.

### `mlea run` — the eval harness

Runs an agent, enforces the time cap, and collects artifacts in exactly the layout
`mlea triage` reads. Five commitments, each one a thing that is easy to get wrong:

- **The submission on disk at the end is the result.** A run killed at its cap that still
  left a valid `submission.csv` is graded, not discarded.
- **Kill the process group, not the process.** Agents spawn training children; signalling
  only the parent leaves them holding the GPU and poisoning the next run on that node.
  There is a test that spawns a real child and asserts it dies.
- **SIGTERM, grace period, then SIGKILL**, so an agent that traps SIGTERM gets to flush its
  best submission.
- **Agent stdout is never trusted for classification.** It goes to `logs/agent.log`; the
  harness writes `logs/harness.log`. Infra signatures are read only from the latter —
  otherwise an agent prints "Spot instance interruption" and earns itself a free retry.
- **Terminal run directories are immutable** without an explicit `--force`.

It also snapshots the running submission at wall-clock marks (`--checkpoint-marks`),
which delivers the curve-shape triage from [the anytime
proposal](docs/PROPOSAL-anytime-eval.md) as a side effect. Snapshotting is passive, so a
run with it produces a byte-identical final result to one without — there is a test for
that too.

### `mlea triage` — what the medal rate is actually made of

A medal rate blends genuine ML underperformance, agent bugs (malformed CSV, OOM, never
wrote a file), and our own infra faults. Triage separates them from run artifacts, so the
headline number means one thing instead of six:

```
6 run(s)
  infra                     1
  oom                       1
  no_submission             1
  invalid_submission        1
  valid                     2

  capability signal         2  (gradeable submissions)
  agent bugs                3  (counted against the agent, but not ML capability)
  our fault                 1  (excluded; retryable)
  effective denominator     5  of 6

  !! 50% of runs failed mechanically rather than on ML. The medal rate is measuring
     plumbing, not capability -- fix these before reading anything into the score.
```

Three decisions in it worth arguing with:

- **A run killed at the time cap that still left a valid `submission.csv` is a
  result, not a failure.** MLE-bench grades whatever is on disk at the end. Getting this
  backwards discards real results, so it is the first thing the tests pin down.
- **Only infra failures may be retried.** `assert_retry_allowed` raises on anything else —
  retrying an agent failure re-rolls the dice and inflates the medal rate, and that rule
  erodes the moment someone is staring at a red dashboard.
- **Attribution logs cannot settle is marked ambiguous, not guessed.** Exit 137 is SIGKILL,
  sent by both OOM killers and time enforcers; with no cap hit and no OOM line, triage says
  so and names the telemetry that would resolve it.

`--emit-runset` writes a run set JSON that `mlea compare` consumes, with infra failures
already flagged for exclusion.

### `mlea power` / `mlea compare`

Two behaviours worth knowing about:

- **It refuses incomparable comparisons.** Different competition split, container config or
  harness version raises `IncomparableError` instead of returning a number. Quoting a lite
  result against a full-set result is the single most common error in the public MLE-bench
  literature, and this makes it a crash rather than a slide.
- **It flags designs that cannot work.** A paired test over fewer than 6 competitions has a
  p-value floor above 0.05 — no effect size can make it significant. That is reported, rather
  than surfacing as a non-significant p-value that reads like evidence of no effect.

Power is computed against **real published per-competition medal rates**, not an assumed
distribution — upstream ships raw per-seed grading reports under `runs/`, git-LFS tracked and
easy to miss. The shipped table verifies itself by reproducing the paper's headline 16.9%.

It found that the contamination experiment proposed in this repo detects essentially nothing
(MDE **−90%**) before a dollar was spent on it. See [`docs/POWER-FINDINGS.md`](docs/POWER-FINDINGS.md).

Everything else here is **planning documents**.

## Contents

| Document | What it covers |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | The build plan: architecture, phases, cost model, risks, open questions |
| [`docs/PROPOSAL-mle-bench-live.md`](docs/PROPOSAL-mle-bench-live.md) | **Flagship proposal:** a rolling, contamination-controlled split built from post-cutoff Kaggle competitions |
| [`docs/PROPOSAL-anytime-eval.md`](docs/PROPOSAL-anytime-eval.md) | Demoted — anytime checkpointing, kept as cheap triage instrumentation |
| [`docs/SOTA-AND-FREE-TIER.md`](docs/SOTA-AND-FREE-TIER.md) | Current SOTA on the benchmark, and a $0 recipe for getting a pipeline working |
| [`docs/POWER-FINDINGS.md`](docs/POWER-FINDINGS.md) | What our sweeps can and cannot detect — output of the tool below |

## Read this first

**Current SOTA is [MLEvolve](https://github.com/InternScience/MLEvolve) at 65.3% ± 0.8% any-medal**
on the full 75-competition set (12 h budget, 3 seeds) — up from 16.9% in the original Oct 2024
paper. Note that most published MLE-bench numbers are on *different splits* and are not
comparable to each other; see [`docs/SOTA-AND-FREE-TIER.md`](docs/SOTA-AND-FREE-TIER.md).

**The open question that matters most:** MLE-bench's contamination check was run in 2024 on
agents scoring 8.5% — at the floor, where it had almost no power to detect inflation. Nobody has
re-run it at 65%. In the neighbouring benchmark, frontier models score 76–81% on SWE-bench
Verified but ~23% on post-cutoff SWE-bench-Live. Whether MLE-bench has the same problem is
unknown, and it conditions every other number here. See
[`docs/PROPOSAL-mle-bench-live.md`](docs/PROPOSAL-mle-bench-live.md).

MLE-bench is expensive. A full 75-competition sweep at the reference hardware spec
(24 h, 36 vCPU, 440 GB RAM, 1×A10) across the recommended 3 seeds is **225 node-days** of
GPU compute before you count a single LLM token. The entire plan is organised around not
paying that bill more often than you have to — see [Cost model](docs/PLAN.md#5-cost-model).

## Status

- [x] Plan drafted
- [x] SOTA baseline + $0 bootstrap path documented
- [x] `mlea` comparison + power tooling
- [x] `mlea triage` failure classification
- [x] `mlea run` eval harness
- [x] `mlea report` eval-dots visualization
- [x] Power model grounded in real published run data (206 tests total)
- [ ] Plan reviewed
- [ ] Phase 0 run against real prepared data — the harness is written and tested, but has
      only ever run stub agents. Doable for **$0**, see [the free-tier recipe](docs/SOTA-AND-FREE-TIER.md#part-2--running-it-for-0)
