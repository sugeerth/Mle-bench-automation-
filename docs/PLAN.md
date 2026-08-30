# MLE-bench Automation — Build Plan

**Status:** draft for review
**Scope:** an automation layer *around* upstream `openai/mle-bench`, not a fork of it.

---

## 1. Goal

Turn MLE-bench from "a thing you run manually for a paper" into "a thing that runs on a
schedule and tells you whether your agent got better or worse."

Concretely, three capabilities:

1. **Reproducible sweeps.** One command launches N agents × M competitions × K seeds, and
   the resulting numbers are comparable to last week's numbers.
2. **Cheap iteration.** A developer changing a prompt gets a directional signal in under an
   hour for well under $100, not in two days for $30k.
3. **Diagnosis, not just a score.** When the medal rate drops, the system says *which*
   competitions moved and *why* the runs failed, without a human reading 200 log files.

### Non-goals

- Forking or re-implementing `mlebench` grading. We call upstream; we don't reinvent it.
- Building a new agent. This repo evaluates agents, it doesn't author one.
- Beating the public leaderboard. Nothing here is tuned to maximise a headline number.

---

## 2. What upstream already gives us

Worth being precise about this, because roughly half of a naive plan is re-implementation
of things that already exist:

- `mlebench prepare --all | --lite | -c <id>` — dataset download + re-split. Needs Kaggle
  API credentials and git-lfs.
- `mlebench grade <jsonl>` and `mlebench grade-sample <path> <competition-id>` — offline
  grading against the held-out split, producing medal thresholds and a grading report.
- `environment/Dockerfile` → `mlebench-env` image, containing the competition data mount
  points, the submission instructions, and a local grading server for agent self-validation.
- `agents/<agent>/` with per-agent Dockerfiles, and `run_agent.py --agent-id --competition-set`
  which creates a run-group directory under `runs/` with per-competition subdirectories and a
  `metadata.json`.
- `experiments/splits/*.txt` — the competition-set files (low/lite, medium, high, split75).
- `experiments/make_submission.py` and `experiments/aggregate_grading_reports.py`.

**So the gap we are filling is everything outside a single-node `run_agent.py` invocation:**
fleet orchestration, data caching, cost control, result storage, statistical comparison,
and failure triage.

---

## 3. Architecture

```
                    ┌──────────────────────────────────────────┐
   sweep.yaml ─────►│  Planner                                  │
   (agent×comp×seed)│  expands matrix, dedupes vs. result store │
                    └───────────────┬──────────────────────────┘
                                    │ run specs
                    ┌───────────────▼──────────────────────────┐
                    │  Scheduler                                │
                    │  leases nodes, enforces budget cap,       │
                    │  retries infra failures (not agent ones)  │
                    └───────────────┬──────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        ┌──────────┐          ┌──────────┐          ┌──────────┐
        │ Worker   │          │ Worker   │   ...    │ Worker   │
        │ node     │          │ node     │          │ node     │
        │          │          │          │          │          │
        │ data cache (read-only mount, prepared once) │          │
        │ docker run mlebench-env + agent image       │          │
        └────┬─────┘          └────┬─────┘          └────┬─────┘
             │ submission.csv, logs, code, checkpoints    │
             └─────────────────────┬─────────────────────┘
                                   ▼
                    ┌──────────────────────────────────────────┐
                    │  Result store (object storage + SQLite/PG)│
                    │  immutable run records, content-addressed │
                    └───────────────┬──────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
        ┌──────────┐          ┌──────────┐          ┌──────────┐
        │ Grader   │          │ Comparer │          │ Triage   │
        │ mlebench │          │ stats vs.│          │ failure  │
        │ grade    │          │ baseline │          │ taxonomy │
        └──────────┘          └──────────┘          └──────────┘
```

### 3.1 Component notes

**Planner.** Reads a declarative `sweep.yaml`. Its one interesting job is *deduplication*:
a run is keyed by `(agent_image_digest, competition_id, seed, container_config_hash,
harness_version)`. If that key is already in the result store with a terminal status, the
Planner skips it. This is what makes "re-run the sweep, I only changed one agent" cheap.

**Scheduler.** Deliberately boring. A work queue, a node pool, a lease with a heartbeat. The
two rules that matter:
- *Retry infra failures, never agent failures.* An OOM-killed container because the node was
  oversubscribed is a retry. An agent that wrote a malformed `submission.csv` is a **result**,
  and retrying it silently inflates your medal rate. This distinction has to be enforced in
  code, not convention.
- *Hard budget cap.* The sweep carries a dollar ceiling. The Scheduler stops leasing nodes
  when projected spend crosses it and marks the sweep `partial`, rather than running to
  completion and surprising someone.

**Data cache.** `mlebench prepare --all` is ~3.3 TB and takes ~2 days. Prepare once, snapshot
the prepared directory to a volume/object store, and mount it **read-only** on every worker.
Two consequences worth stating: preparation is a separate, versioned pipeline with its own
cadence, and workers never need Kaggle credentials.

**Result store.** Append-only. One row per run with the dedupe key, exit status, resource
usage, cost, and content-addressed pointers to `submission.csv`, `logs/`, `code/`, and (see
the proposal) intermediate checkpoints. Grading writes a separate row referencing the run —
so re-grading after an upstream grader fix doesn't mutate run history.

**Comparer.** The part people skip and then regret. See §6. **Implemented** — `mlea compare`
and `mlea power` in `src/mlea/`, the only code in this repo so far. It was built first because
it gates whether the expensive parts are worth running at all.

**Triage.** See §7. **Implemented** — `mlea triage`.

---

## 4. Phases

Each phase is independently useful; stop after any of them and you have something.

### Phase 0 — Walking skeleton *(~1 week)*

> **Partly implemented.** `mlea run` is the harness; `mlea triage` and `mlea compare` are
> the downstream. What remains is pointing it at real prepared competition data — it has
> only ever run stub agents.


One agent, one competition, one seed, on one machine, end to end: prepare → run → grade →
a row in the result store. No scheduler, no parallelism, no cloud.

**Exit criterion:** `make demo` reproduces a known medal result for the `dummy` agent on a
single lite competition, twice, with matching output.

The point of this phase is to discover the actual friction (image build times, mount paths,
credential handling) before designing around imagined friction.

**Phase 0 costs $0.** It can be done entirely on free infrastructure with a free LLM tier —
see [`SOTA-AND-FREE-TIER.md`](SOTA-AND-FREE-TIER.md#part-2--running-it-for-0). There is no
reason to provision anything before this phase has run.

### Phase 1 — Lite sweep, single node *(~1–2 weeks)*

All 22 lite competitions, 1 seed, sequential or lightly parallel on one big node. Result
store and grading pipeline real. Reporting is a static HTML/markdown summary.

**Exit criterion:** a full lite sweep completes unattended overnight and produces a report
with per-competition medal status and total cost.

### Phase 2 — Fleet + budget control *(~2–3 weeks)*

Scheduler, node pool, leases, retry classification, budget cap, data cache as a shared
read-only mount. Parallelism across competitions.

**Exit criterion:** a 22-competition × 3-seed lite sweep finishes in wall-clock time close to
`ceil(66 / node_count) × mean_run_hours`, and a deliberately killed node is recovered without
double-counting or losing a result.

### Phase 3 — Comparison + regression gating *(~2 weeks)*

Baseline registry, variance-aware comparison (§6), and a CI entry point that a PR to an agent
repo can call. This is where the system starts paying for itself.

**Exit criterion:** a knowingly-worse agent (e.g. a truncated context window) is flagged as a
regression; a knowingly-neutral change (a comment-only diff) is not.

### Phase 4 — Triage *(~2 weeks)*

Failure taxonomy (§7), plus the reduced anytime checkpointing kept for curve-shape triage
([`PROPOSAL-anytime-eval.md`](PROPOSAL-anytime-eval.md) — demoted, see that document).

### Phase 5 — MLE-bench-Live *(flagship, ~3–4 weeks + ongoing)*

A rolling, contamination-controlled split built from post-cutoff Kaggle competitions. See
[`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md). This replaces "scheduled full
split75 sweeps" as the thing worth building toward — scheduled sweeps of a possibly-contaminated
static benchmark are expensive noise if the validity question is unanswered.

Scheduled sweeps (nightly lite, weekly medium, monthly full) remain worth running once Phases
2–3 have proven the numbers are stable, but they are maintenance, not a destination.

---

## 5. Cost model

> ⚠️ **These are order-of-magnitude estimates and must be verified against real quotes before
> anyone commits a budget.** They are here to drive design decisions, not procurement.

Reference spec from upstream: 24 h, 36 vCPU, 440 GB RAM, 1×24 GB A10, per competition-seed.
The 440 GB RAM requirement, not the GPU, is what pushes you into expensive instance classes.

| Sweep | Runs | Node-hours | Est. compute | Est. LLM tokens |
| --- | --- | --- | --- | --- |
| Lite, 1 seed | 22 | ~530 | ~$3–4k | ~$0.5–2k |
| Lite, 3 seeds | 66 | ~1,580 | ~$9–13k | ~$1.5–6k |
| Full 75, 3 seeds | 225 | ~5,400 | ~$30–43k | ~$5–20k |

Assumes ~$5.50–8.00/node-hour on-demand and that runs use their full 24 h budget.

### The three levers, in order of impact

1. **Don't use the full 24 h.** Most agents plateau far earlier. This is the single biggest
   saving available and is the direct motivation for the anytime-evaluation proposal — you
   cannot early-stop safely until you can *measure* the score-vs-time curve.
2. **Spot/preemptible instances.** ~60–70% cheaper. Requires the Scheduler to treat
   preemption as an infra failure and requires agent-side checkpointing to avoid losing 20 h
   of work. Do this *after* Phase 2, not before — preemption handling on an unproven
   scheduler is a debugging nightmare.
3. **Tiered gating.** Nightly runs hit lite only. Full sweeps run only for release
   candidates. Cheap signal frequently, expensive signal rarely.

### What we deliberately do not do

Shrink the reference hardware spec to save money. It changes what the benchmark measures and
silently breaks comparability with published numbers. If we ever do run reduced hardware, the
result store must tag those runs as a different `container_config_hash` so they can never be
compared against reference runs by accident. (This is already enforced by the dedupe key.)

---

## 6. Comparing two agents without fooling yourself

MLE-bench's headline metric is **Any Medal (%)** across ~75 competitions with ≥3 seeds. The
statistics here are genuinely nasty and deserve explicit design attention:

- **n is small.** 22 competitions in lite. A 2-competition swing is ~9 percentage points and
  is well within noise.
- **Runs are expensive**, so you cannot buy your way to tight error bars.
- **Per-competition outcomes are binary** (medal / no medal) and highly heterogeneous — some
  competitions are ~always won, some ~never.

Design decisions that follow:

1. **Report mean ± standard error over seeds**, matching upstream, so numbers stay comparable.
2. **Pair the comparison.** Compare agent A vs. B *per competition*, not two independent
   aggregate rates. Competition difficulty is the dominant variance term and pairing removes
   it. Use a paired test over per-competition medal counts.
3. **Publish the minimum detectable effect** alongside every comparison. This is now
   measured, not hypothetical: a 3-seed lite sweep detects a **−22.7%** regression and
   nothing subtler; the full split detects **−12.1%**
   ([`POWER-FINDINGS.md`](POWER-FINDINGS.md)). A 6-point "improvement" on lite is noise.
   `mlea compare` reports the interval and refuses to hide it.
4. **Never let a re-run replace a result.** Re-running a failed competition until it medals is
   the most natural and most corrupting thing a person can do with this system. Every run is
   recorded; aggregate metrics read *all* runs matching the key, not the latest.
5. **Freeze the split.** The competition set is part of the baseline identity. Changing which
   competitions are in a sweep invalidates comparison to prior baselines, and the Comparer
   should refuse to compare across differing split hashes rather than silently doing it.

---

## 7. Failure triage

A large fraction of non-medal outcomes are not "the agent's model was weak" — they're
mechanical: malformed submission, wrong column names, ran out of time mid-training, crashed on
an OOM, never produced a file. Reporting these together with genuine ML underperformance makes
the medal rate uninterpretable.

Proposed taxonomy, assigned automatically from run artifacts:

| Class | Detection | Actionable by |
| --- | --- | --- |
| `no_submission` | no `submission.csv` at exit | agent author |
| `invalid_format` | upstream grading server rejects | agent author |
| `timeout_mid_train` | wall clock hit cap with training in progress | budget / agent |
| `oom` / `crash` | container exit code + log signature | harness or agent |
| `infra` | node preemption, image pull failure, mount error | **us** |
| `valid_no_medal` | graded, below bronze threshold | genuine capability signal |

**Implemented** as `mlea triage` (`src/mlea/triage.py`). Two refinements the table above
missed, both found while writing it:

- `timeout_mid_train` splits in two. A run killed at the cap that *still left a valid
  submission.csv* is a **result**, not a failure — MLE-bench grades whatever is on disk at
  the end. Only a timeout with no gradeable artifact is a failure. Treating every timeout as
  a failure would discard real results.
- Some attributions are not determinable from logs. Exit 137 is SIGKILL, which both OOM
  killers and time enforcers send; host OOM may be the agent over-allocating or us
  oversubscribing the node; `no space left on device` may be either. These are marked
  `ambiguous` with a note naming the telemetry that settles them, rather than guessed.

`assert_retry_allowed` makes the retry rule in §3.1 a raised exception rather than a
convention.

Only `valid_no_medal` is a real capability result. Everything above it is a bug in something,
and the report should separate the two columns rather than blending them into one percentage.

`infra` failures must additionally be excluded from capability metrics entirely — they are our
fault and counting them as agent failures understates the agent.

Classification starts as rules over exit codes and log regexes. An LLM-based classifier over
the tail of the log is a reasonable Phase 4 addition for the residual `crash` bucket, but rules
should handle the common cases — they're cheaper and deterministic.

---

## 8. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Benchmark saturation (low split already at 80.3%) | Medium–High | Rolling split refreshes faster than agents improve |
| Cost overrun | High | Hard budget cap in the Scheduler; tiered gating; lite-first |
| Kaggle API / dataset drift breaks `prepare` | High | Snapshot prepared data; version it; never re-prepare mid-sweep |
| Upstream `mlebench` changes grading | Medium | Pin the upstream commit in the dedupe key; re-grade explicitly, never implicitly |
| Results silently non-comparable | High | Split hash + container config hash in the key; Comparer refuses mismatched comparisons |
| **Benchmark contamination (agents recall public Kaggle solutions)** | **High** | The 2024 check found nothing, but ran at an 8.5% medal rate where it had no power. SWE-bench shows 76–81% static vs ~23% post-cutoff. See [`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md) |
| 3.3 TB storage + egress | Medium | Single prepared snapshot, read-only mounts, region-local workers |
| Flaky agent runs read as regressions | Medium | Seeds ≥3, paired tests, published MDE |

---

## 9. Open questions

These need answers before Phase 2, and I'd like input on them:

0. **Baseline to beat.** Current SOTA is MLEvolve at 65.3% on the full set
   ([details](SOTA-AND-FREE-TIER.md#part-1--what-the-state-of-the-art-actually-is)). Is the
   target to match a published system, or to track a first-party agent over time? These imply
   different splits and very different budgets.
1. **Where does this run?** Cloud provider and whether we have quota for 440 GB-RAM + A10-class
   instances at the parallelism Phase 2 assumes. This changes the Scheduler substantially
   (Kubernetes vs. a plain EC2/GCE pool vs. Slurm).
2. **Which agents are in scope?** Upstream ships AIDE, MLAgentBench, OpenHands, and `dummy`.
   Are we evaluating a first-party agent as the primary target, with the others as baselines?
3. **What is the actual iteration loop we're serving?** "A researcher tweaking prompts daily"
   and "a release gate run monthly" imply very different priorities — the former makes Phase 3
   urgent and Phase 5 nearly irrelevant.
4. **Budget ceiling.** A number here would resolve most of §5's tension immediately.

---

## 10. Recommended first step

Phase 0, on a single machine, this week. It is cheap, it de-risks every later assumption, and
the friction it uncovers is the input to designing Phase 2 properly. Everything past Phase 1
should be re-planned once Phase 0 has actually run.
