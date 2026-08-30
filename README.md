# MLE-bench Automation

Automation harness for running [MLE-bench](https://github.com/openai/mle-bench) (OpenAI's
Kaggle-competition benchmark for ML engineering agents) repeatably, cheaply, and as a
regression signal rather than a one-off leaderboard stunt.

This repository currently contains **planning documents only**. No code has been written yet.

## Contents

| Document | What it covers |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | The build plan: architecture, phases, cost model, risks, open questions |
| [`docs/PROPOSAL-mle-bench-live.md`](docs/PROPOSAL-mle-bench-live.md) | **Flagship proposal:** a rolling, contamination-controlled split built from post-cutoff Kaggle competitions |
| [`docs/PROPOSAL-anytime-eval.md`](docs/PROPOSAL-anytime-eval.md) | Demoted — anytime checkpointing, kept as cheap triage instrumentation |
| [`docs/SOTA-AND-FREE-TIER.md`](docs/SOTA-AND-FREE-TIER.md) | Current SOTA on the benchmark, and a $0 recipe for getting a pipeline working |

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
- [ ] Plan reviewed
- [ ] Phase 0 (walking skeleton) implemented — can be done for **$0**, see [the free-tier recipe](docs/SOTA-AND-FREE-TIER.md#part-2--running-it-for-0)
