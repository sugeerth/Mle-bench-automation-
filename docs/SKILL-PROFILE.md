# Which competence is missing?

```bash
mlea skills --out skills
```

## The problem with one number

MLE-bench reports a single figure. At 65.3% and rising it is close to saturating, and this
repository has already measured two ways that number stops discriminating:

- **Medal rate is binary and saturates.** Once an agent is good enough to medal, a better
  one gets the same number ([`CONTAMINATION-PROBE.md`](CONTAMINATION-PROBE.md)).
- **Percentile compresses at the top.** A perfect solution and a competent one both sit in
  the leaderboard's top few percent.

Neither can say *what* an agent is missing, which is the only thing useful to someone
improving it.

## The design

Every pathology is generated **alongside an otherwise identical clean control** — same seed,
same latent function, same features, target untouched. The pair differs by exactly one thing,
so the score difference isolates one skill.

That pairing is impossible on real competitions. Matching two Kaggle competitions on
difficulty is guesswork, and a systematic difficulty difference is indistinguishable from the
effect you are trying to measure — the same problem that made
[`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md) unworkable. Here the control is
the same problem by construction, and there is a test asserting the oracle score is identical
across every challenge.

Scored in **leaderboard percentile**, never medal rate, for the reason above.

| Pathology | What it injects | The competence it rewards |
| --- | --- | --- |
| `leakage` | A feature that all but gives away the target in train and is pure noise in test | Validating a feature instead of trusting it |
| `shift` | Test features drawn shifted and rescaled relative to train | Regularisation; distrust of extrapolation |
| `outliers` | 2% of cells multiplied 30–80× | Robust preprocessing |
| `missing` | 4% of cells empty | Imputation — without it the agent emits NaN and produces an **ungradeable** submission |

## ⚠️ The first version of this document was wrong

It reported the table below from **one competition per cell**, with no interval:

|  | leakage | missing | outliers | shift |
| --- | ---: | ---: | ---: | ---: |
| `naive` | −47% | BROKE | −86% | +1% |
| `careful` | −61% | −1% | −18% | −26% |
| `expert` | +17% | −1% | −18% | −26% |

From it I drew three conclusions: that *nobody handles shift*, that *clipping costs 18 points
on outliers*, and that *`careful` is worse than `naive` on leakage*.

**Two of those were noise and the third had the sign backwards.** This repository had spent
considerable effort establishing that a benchmark difference without an interval is not a
result, and then reported its own headline finding from n=1. Re-running the identical
measurement over 8 paired competitions:

## Result

Cost of each pathology, in leaderboard percentile points versus the matched control.
8 paired competitions per cell, 95% bootstrap CI, paired sign-flip test.

| agent | pathology | Δ | 95% CI | p | verdict |
| --- | --- | ---: | :---: | ---: | --- |
| `naive` | leakage | **−65.9%** | [−76.8, −53.4] | 0.0117 | real |
| `naive` | missing | **BROKE** | — | — | 8/8 ungradeable |
| `naive` | outliers | **−86.6%** | [−92.6, −79.9] | 0.0117 | real |
| `naive` | shift | +2.6% | [−0.5, +5.8] | 0.1751 | **noise** |
| `careful` | leakage | **−47.8%** | [−63.5, −35.1] | 0.0117 | real |
| `careful` | missing | +0.9% | [−1.6, +3.6] | 0.6031 | **noise** |
| `careful` | outliers | −1.2% | [−8.6, +4.1] | 0.8366 | **noise** |
| `careful` | shift | −4.2% | [−11.0, +2.2] | 0.2996 | **noise** |
| `expert` | leakage | **+12.1%** | [+7.4, +17.9] | 0.0117 | real |
| `expert` | missing | +0.9% | [−1.6, +3.6] | 0.6031 | **noise** |
| `expert` | outliers | −1.2% | [−8.6, +4.1] | 0.8366 | **noise** |
| `expert` | shift | −4.2% | [−11.0, +2.2] | 0.2996 | **noise** |

Eight of twelve cells are indistinguishable from zero. Only leakage and missing survive.

### What actually held

**`naive` fails loudly, in two different ways.** Missing values do not produce a low score;
they produce **no gradeable submission at all** — 8 of 8 — landing in triage as an agent bug
rather than weak ML. Outliers cost it 86 percentile points. Both are real and large.

**Leak detection is worth 78 percentile points.** `naive` −65.9% versus `expert` +12.1%, with
intervals nowhere near each other. `expert` *gains* because the simulated field mostly falls
for the leak, so avoiding it is worth a gold medal. Real competitions with leaks behave
exactly this way.

**`careful` beats `naive` on leakage, not the reverse.** −47.8% versus −65.9%. The single-run
result had this backwards, and the story I told about it — "its capacity search finds the leak
more effectively and latches on harder" — was a plausible mechanism invented to explain noise.

### What did not hold

**"Nobody handles shift" was noise.** Every shift cell is non-significant, the largest being
−4.2% with an interval spanning zero. The confident mechanistic explanation I gave for it —
capacity search on a train-drawn holdout being wrong under covariate shift — may well be true
in general, but this experiment does not show it.

**"Clipping costs 18 points on outliers" was noise.** −1.2%, p=0.84. Conditional clipping
handles outliers essentially for free.

### And the headline claim inverted

The single-run table said **no agent dominates**, which was the whole argument for profiling
over ranking. With intervals, `expert` is not measurably beaten on any pathology — it
dominates, and for *ranking* this field a single score would have sufficed.

The profile still earns its place, but for a smaller and more honest claim: it says **where**
the other agents lose, and by how much, which a single score cannot. Diagnosis, not ranking.

## Significance is not practical significance

A perfectly consistent effect is significant at any size: eight paired differences of the same
sign give p = 3/257 whether they are 1 point or 60. Reporting on significance alone flagged a
−1% effect as a missing competence. A cell must clear both bars — distinguishable from zero
**and** at least 2 percentile points — before it counts against an agent.

## Two implementation notes worth stealing

**Clip conditionally, never unconditionally.** Winsorising every column is its own competence
failure: on clean data it discards real signal, and under covariate shift it clamps
legitimately out-of-range test values back into the train range, destroying the information
the shift moved. A column is clipped only when its own extreme value says something is wrong
with it.

**Detect outliers with the median and MAD, not the mean and standard deviation.** An outlier
inflates the standard deviation, which shrinks its own z-score — the masking effect — so
mean/std detection hides exactly the values it exists to catch. There is a test with five
gross outliers that mean/std misses and MAD finds.

## What this does and does not establish

**Does:** that a paired-control design can attribute a score difference to one named
competence, and that two of these pathologies (leakage, missing) discriminate sharply between
agents that a clean benchmark reports as equivalent. Also, uncomfortably, that a
point-estimate profile is worse than useless — it produced three confident conclusions of
which two were noise and one was backwards.

**Does not:** show that shift or outliers discriminate at all at this sample size — both are
non-significant and may need a larger design or a stronger pathology. Nor does it say anything
about real agents on real competitions. These are tabular and
generated; the three reference agents are ridge regressions differing by a few preprocessing
steps, not frontier ML engineers. What transfers is the **method** — generate the control,
score in percentile, report the profile — not these numbers.
