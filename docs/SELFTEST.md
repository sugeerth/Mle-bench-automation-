# Running the pipeline for real, in about a minute

```bash
pip install -e .
mlea selftest
```

No Kaggle account, no credentials, no cost, no Docker. It generates real gradeable
competitions, runs real models against them, grades real scores, classifies real failures,
compares two agents, renders a report, and then checks that all five stages agree.

## Why this exists

Everything in this repository was, until now, tested against stub agents. It could not be
otherwise: exercising it against a scoreable competition needs a Kaggle account **and** a
rules-acceptance click that cannot be automated — upstream's downloader literally calls
`webbrowser.open(...)` and then `input()`. So the tooling was well-tested against fixtures and
completely untested against anything that could produce a score.

`mlea bench` closes that. The competitions are synthetic but not fake.

## What a generated competition is

| Piece | How it is made |
| --- | --- |
| Features | Gaussian, configurable width |
| Target | A latent function — linear terms, an interaction, and a `tanh` nonlinearity — plus noise scaled by `difficulty` |
| Split | Train/test, answers written only to `prepared/private/` |
| Metric | A real implementation (`roc_auc`, `rmse`, …), numpy-only |
| Leaderboard | **300 real fitted models** — ridge regressions at four capacities, log-uniform regularisation, bootstrap resamples — scored on the same test split |
| Medal thresholds | Rank positions on that leaderboard, using upstream's exact tiers |

The nonlinearity matters: a purely linear target makes ridge an oracle, every team ties, and
medal thresholds become meaningless. There has to be structure a linear model cannot fully
capture.

The layout is upstream's, so `mlea run` resolves it the same way real data would:

```
<root>/<id>/competition.json          metric, thresholds, oracle score
<root>/<id>/leaderboard.json          the 300 simulated scores
<root>/<id>/prepared/public/          train.csv, test.csv, sample_submission.csv, description.md
<root>/<id>/prepared/private/         answers.json
```

## Two properties MLE-bench does not have

**Zero threshold-transfer bias.** Upstream compares a score computed on its own re-split
against a raw value read off Kaggle's leaderboard, which was computed on Kaggle's *private*
split. Of its 82 preparation scripts, **69 use plain i.i.d. `train_test_split`; none stratify;
none are group- or time-aware** — while the private splits those thresholds came from were
often split by time, site or patient deliberately, to prevent leakage. An i.i.d. split is
systematically easier, so the transfer is biased **upward**, not merely noisy. Here the
leaderboard is computed on the same split the agent is graded on, so the bias is zero by
construction.

**A known oracle.** The latent function is known, so the best achievable score is computable —
for a lower-is-better metric that is the irreducible error, not zero. Every generated
competition satisfies `gold` slightly behind `oracle`, which is asserted in the tests. This is
the correctness oracle whose absence is what makes a rolling Kaggle split infeasible
([`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md)), and it is what lets a self-test
*verify* the grader rather than merely exercise it.

## The reference agents

`python -m mlea.baseline --strategy <name>`, wired to the harness contract.

| Strategy | What it does | Expected |
| --- | --- | --- |
| `constant` | Predicts the training mean | 0.5 AUC, no medal |
| `linear` | Ridge on raw features | Better, usually no medal |
| `tuned` | Searches capacity × regularisation on a holdout, rewriting the submission on every improvement | Medals; produces a real score-vs-time curve |
| `broken` | Right rows, column named `prediction` instead of `target` | `invalid_submission` |
| `silent` | Exits 0 having written nothing | `no_submission` |
| `crash` | Non-zero exit with a traceback | `crash` |
| `hungry` | Prints a real CUDA OOM string and exits 1 | `oom` |

The modelling strategies produce genuine quality ordering — measured, on a mid-difficulty
competition with oracle 0.810: constant **0.500**, linear **0.775**, tuned **0.808** (silver).

Submissions are written atomically (`tmp` then `replace`). The harness snapshots the file on a
timer from another thread, and a plain truncate-and-write lets a snapshot catch a half-written
file and score it as malformed — an instrumentation artefact indistinguishable from a real
agent bug.

## What the self-test caught

It found a real gap in the pipeline on its first run. `broken` writes a well-formed CSV with
the right number of rows and a wrong column name. Triage runs *before* grading and sees only a
file on disk, so it classified the run `valid`; the grader rejected it. Nothing was carrying
the grader's verdict back.

`mlea grade` now writes `validation_error` into the run's `metadata.json`, so a later triage
sees it. The order matters: **run → grade → triage**, and the self-test asserts the
before/after difference explicitly.

## A note on the significance result

The self-test's comparison reports `p = 0.0462`, which is exactly `3/65` — the p-value floor
for a 6-unit paired sign-flip test derived in [`POWER-FINDINGS.md` §7](POWER-FINDINGS.md).
A 100-point difference across 6 competitions lands precisely at the smallest p-value the design
can express. The theory and the implementation were written separately and agree.

## What this is not

Not a substitute for MLE-bench. These are tabular, generated and small; they contain no images,
no text, no messy real-world data, and no distribution shift. An agent that does well here has
demonstrated that the *plumbing* works, not that it can do ML engineering.

The next step is still the one that has never happened: point this at a real prepared Kaggle
competition. That costs $0 — see [`SOTA-AND-FREE-TIER.md`](SOTA-AND-FREE-TIER.md).
