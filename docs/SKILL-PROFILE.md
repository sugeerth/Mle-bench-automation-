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

## Result

Cost of each pathology, in leaderboard percentile points versus the matched control:

|  | leakage | missing | outliers | shift | robustness |
| --- | ---: | ---: | ---: | ---: | ---: |
| `naive` | −47% | **BROKE** | −86% | +1% | −58% |
| `careful` | −61% | −1% | −18% | −26% | −26% |
| `expert` | **+17%** | −1% | −18% | −26% | −11% |

**No agent dominates.** `expert` wins leakage, `careful` wins missing and outliers, `naive`
wins shift. Four pathologies, three different winners — a single headline score cannot
express this, and would report the three agents as roughly interchangeable on clean data.

### What each row says

**`naive` fails loudly and in different ways.** Missing values do not produce a low score;
they produce **no gradeable submission at all**, which lands in triage as an agent bug rather
than weak ML. Outliers cost it 86 percentile points. That is the whole argument for
separating mechanical failures from capability in the first place.

**`careful` is worse than `naive` on leakage (−61% vs −47%).** Competence is not a ladder.
Its capacity search finds the leak *more* effectively and latches on harder. Preprocessing
without validation makes the leak worse.

**`expert` gains on leakage (+17%).** It drops the suspicious feature — and because the
simulated field mostly falls for the leak, avoiding it is worth a gold medal. Real
competitions with leaks behave exactly this way.

**Nobody handles shift.** Both preprocessing agents lose 26 points where `naive`'s plain
regularised ridge gains 1. Capacity search on a holdout drawn from train is precisely wrong
under covariate shift: the holdout does not contain the shift, so the search selects a model
that extrapolates badly. This is a competence none of the three has, and it is visible only
because the profile separates it out.

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
competence, that the pathologies discriminate sharply between agents that a clean benchmark
reports as equivalent, and that no single ordering of these agents exists.

**Does not:** anything about real agents on real competitions. These are tabular and
generated; the three reference agents are ridge regressions differing by a few preprocessing
steps, not frontier ML engineers. What transfers is the **method** — generate the control,
score in percentile, report the profile — not these numbers.
