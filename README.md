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

mlea power --design live-gap          # can this experiment detect anything?
mlea power --design lite-regression   # what regression size can a lite sweep catch?
mlea compare baseline.json candidate.json --fail-on-regression
```

Two behaviours worth knowing about:

- **It refuses incomparable comparisons.** Different competition split, container config or
  harness version raises `IncomparableError` instead of returning a number. Quoting a lite
  result against a full-set result is the single most common error in the public MLE-bench
  literature, and this makes it a crash rather than a slide.
- **It flags designs that cannot work.** A paired test over fewer than 6 competitions has a
  p-value floor above 0.05 — no effect size can make it significant. That is reported, rather
  than surfacing as a non-significant p-value that reads like evidence of no effect.

It found that the contamination experiment proposed in this repo was underpowered by roughly
a factor of two before a dollar was spent on it. See [`docs/POWER-FINDINGS.md`](docs/POWER-FINDINGS.md).

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
- [x] `mlea` comparison + power tooling (61 tests)
- [ ] Plan reviewed
- [ ] Phase 0 (walking skeleton) implemented — can be done for **$0**, see [the free-tier recipe](docs/SOTA-AND-FREE-TIER.md#part-2--running-it-for-0)
