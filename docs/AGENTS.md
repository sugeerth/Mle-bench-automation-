# Wiring a real agent into `mlea run`

Verified by reading each project's source and config, not its marketing. Every command below
is copied from the project's own README or run script. Anything unverified says so.

`mlea run --agent-cmd` gets `DATA_DIR`, `SUBMISSION_PATH`, `SUBMISSION_DIR`, `LOGS_DIR`,
`CODE_DIR`, `AGENT_DIR`, `COMPETITION_ID`, `SEED`, `TIME_CAP_SECONDS`, and runs with the cwd
set to `CODE_DIR`.

## The one feature you will need: `--submission-glob`

No agent writes to `$SUBMISSION_PATH`. They each write somewhere of their own choosing, and
several write to a path containing a timestamp. `--submission-glob` mirrors the agent's own
file into the run's canonical location at every checkpoint mark and again at exit, which is
also what makes a score-vs-time curve possible without bolting a polling loop onto the agent
command.

| Agent | `--submission-glob` |
| --- | --- |
| AIDE (upstream) | `workspaces/0-run/working/submission.csv` |
| AIDE (mle-bench fork) | `workspaces/0-run/best_submission/submission.csv` |
| MLEvolve | `runs/*/workspace/best_submission/submission.csv` |
| ML-Master | `workspaces/*/best_submission/submission.csv` *(unverified — see below)* |

## Ranked by how little glue they need

### 1. AIDE — `pip install aideml`, no Docker

The easiest by a wide margin: one pip install, one CLI, every knob an OmegaConf `key=value`,
and Gemini's free tier works natively.

```bash
mlea run \
  --agent-name aide \
  --data-root ~/.cache/mle-bench/data \
  --competition spooky-author-identification \
  --time-cap 14400 \
  --submission-glob 'workspaces/0-run/working/submission.csv' \
  --out runs/aide \
  --agent-cmd 'aide data_dir="$DATA_DIR" desc_file="$DATA_DIR/description.md" \
      exp_name=run log_dir=logs workspace_dir=workspaces \
      copy_data=False generate_report=False agent.steps=500 \
      agent.code.model=gemini-2.5-flash agent.feedback.model=gemini-2.5-flash \
      exec.timeout=1800'
```

Four traps, all verified in AIDE's source:

- **`generate_report` defaults to `True`** and fires a final call to `gpt-4.1` against
  OpenAI. On a Gemini-only or Groq-only run that fails or bills you. Always
  `generate_report=False`.
- **A model named `gpt-*` ignores `OPENAI_BASE_URL`.** Backend routing regex-matches the
  model name first: `gpt-*`/`o<n>` always go to `api.openai.com`, `gemini-*` to Google with
  `GEMINI_API_KEY`, `claude-*` to Anthropic. Only a name matching none of those honours
  `OPENAI_BASE_URL` — that is the Groq/Cerebras path
  (`OPENAI_BASE_URL=https://api.groq.com/openai/v1`, model `llama-3.3-70b-versatile`).
- **The `0-` in the glob is only deterministic if `workspaces/` starts empty.** The
  log-index helper has a walrus-precedence bug that returns 2, not 1, once any
  numeric-prefixed directory exists. `mlea run` gives each run a fresh `code/`, so this
  holds — do not pre-create anything under it.
- **`SEED` cannot be passed to AIDE.** Upstream has no seed key. Seeds vary only through LLM
  sampling, which is real variation but not controlled variation. Use `SEED` for bookkeeping.

**Curve quality:** upstream AIDE rewrites its submission after *every* executed node,
including buggy ones, so the curve is continuous but **non-monotonic** — it tracks the last
node, not the best. The mle-bench fork maintains `best_submission/` instead and gives a clean
monotonic curve. If the curve matters, glob the fork's path or reconstruct best-so-far from
`logs/0-run/journal.json`, which is rewritten every step with each node's metric and buggy flag.

### 2. ML-Master and MLEvolve — best-so-far for free

Both maintain `best_submission/submission.csv` continuously, so their curves are monotonic
without any reconstruction. Both take `agent.code.base_url` / `agent.code.api_key` /
`agent.code.model` as first-class OmegaConf keys, so any OpenAI-compatible free tier works.
Both are clone-and-pin installs with heavy pinned dependency sets, and both assume the
mle-bench `<root>/<id>/prepared/public` layout — which `mlea run` now resolves for you.

Notes that will bite on free notebook infrastructure:

- Both default to `parallel_search_num=3` and will refuse to start on a 2-vCPU box. Pass
  `agent.search.parallel_search_num=1 cpu_number=1`.
- MLEvolve requires `mlebench` importable for its format server, and its run directory is
  timestamped — hence the `runs/*/` glob.
- ML-Master needs a pre-generated `full_instructions.txt` per competition and runs a grading
  server unless you pass `agent.check_format=false`.
- ML-Master requires models that emit `<think>` tags; for anything else pass
  `agent.steerable_reasoning=false`, at some cost in performance.
- ML-Master's exact `workspace_dir` composition is **unverified** — check it before relying
  on the glob above.

### 3. Not recommended

- **R&D-Agent** — pip-installable with a genuine non-Docker conda path, but configured only
  through `.env`, uses its own data layout, needs an embedding model most free tiers lack,
  and **has no stable submission path**: each experiment writes into its own workspace, so
  every snapshot requires resolving the SOTA experiment from the log trace.
- **OpenHands CLI** — trivially installable and headless with clean `LLM_*` env vars, but its
  own README says it is no longer actively maintained, and it has no ML scaffolding and no
  submission convention at all. Useful only as a generic-coding-agent control arm.
- **MLAgentBench** — reads API keys from `*.txt` files at import time, pins `openai==0.28`
  with no `base_url` wiring, and requires hand-building a fake benchmark task directory.
- **MLE-STAR** — the ADK sample was **deleted** from `google/adk-samples` in July 2026. The
  leaderboard entries under that name are closed-source. It also writes its submission once
  at the very end, so it can produce no curve at all.

## Free-tier summary

Gemini via AIDE is the shortest path to a working run at zero cost: `GEMINI_API_KEY` from AI
Studio, `agent.code.model=gemini-2.5-flash`, `generate_report=False`. See
[`SOTA-AND-FREE-TIER.md`](SOTA-AND-FREE-TIER.md) for quotas and the honesty requirements that
apply to any unsandboxed run.
