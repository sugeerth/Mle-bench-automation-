# Conformance against the real MLE-bench

```bash
mlea conform
```

```
48/48 agreed within 0.0

CONFORMANT — this package's grader matches every real grader tested,
on both valid submissions and the ones a grader should reject.
```

## What this closes

Every previous version of this repository carried the same caveat: **it had never touched
real MLE-bench.** It generated its own competitions, graded them with its own metrics, and
had no way to know whether any of that matched the benchmark it claimed to be a harness for.

It still cannot download a real competition — that needs a Kaggle account and a
rules-acceptance click that cannot be automated. But that turns out to be the *only* thing
it cannot do. Upstream's grading code is importable and runnable without any of it.

So a competition can be generated in a **real competition's submission schema**, and graded
by that **competition's own grader**. Both graders score the same file, and the scores are
compared exactly.

| | |
| --- | --- |
| Real competitions defined upstream | 82 |
| Whose submission schema is readable | 16 |
| That this package can generate data for | **12** |
| Grader comparisons run | 48 |
| Exact agreements | **48** |

The 12 span 11 AUC competitions and one RMSE, and 8 of them are in the lite split. Each is
checked on three valid submissions of differing quality *and* on one malformed submission,
so the check covers rejection behaviour as well as scoring.

## Why only 12 of 82

Upstream has no machine-readable schema. A competition's id and target columns are stated
only inside its own `grade.py`, as arguments to a shared helper:

```python
prepare_for_auroc_metric(submission=..., answers=..., id_col="request_id",
                         target_col="requester_received_pizza")
```

16 competitions state them that way. The rest — mostly image and text tasks — construct their
columns inline, and the file that would show the shape is `sample_submission.csv`, which only
exists after the Kaggle download. Those are **skipped rather than guessed at**: a plausible
wrong schema would produce a conformance check that passes against the wrong thing.

Of the 16, four are scored by metrics this generator cannot produce data for
(levenshtein-distance, map-at-5, weighted-multi-label-log-loss, probabilistic-f1-score).

## Three things the check found

**Upstream rounds every score to 5 decimals.** `Grader.__call__` returns `round(score, 5)`.
My grader was computing full precision and matched scikit-learn to the last bit — which meant
it *disagreed* with upstream on the fifth decimal. Matching the rounding is what makes a score
here directly comparable to a published MLE-bench number, and it is not cosmetic: upstream's
own Known Issues flag three competitions whose leaderboards are dense enough that a
fifth-decimal difference moves a medal.

**A real schema is more than column names.** `petfinder-pawpularity-score` rejects any
`Pawpularity` outside `[1, 100]`. Generating data with the right column names and the wrong
target domain produced submissions the real grader refused before it ever scored them — all
three valid agents failed. Target bounds are now extracted from the grader source too, the
generator maps into them, and agents clip to the range seen in training.

**Raw model output is an invalid submission.** Upstream's AUC helper rejects anything outside
`[0, 1]` outright rather than ranking it. A linear model's raw scores rank perfectly and are
still ungradeable. The reference agents now rank-normalise, which leaves AUC unchanged and is
what a competent agent does anyway.

All three were invisible while the package only graded itself.

## Running it

Upstream is optional and is not a dependency. To enable the check:

```bash
git clone https://github.com/openai/mle-bench.git
pip install tqdm pandas scikit-learn scipy pyyaml diskcache appdirs tenacity py7zr
PYTHONPATH=/path/to/mle-bench mlea conform
```

A full `pip install -e .` of upstream also pulls the Kaggle client, which is needed only for
`prepare` and frequently fails to build. Grading does not need it.

`tests/test_upstream.py` runs the same checks and **skips cleanly** when upstream is not
importable, so the package still installs and tests with nothing but numpy.

## What is still not proven

The competition *data* is synthetic. These are tabular features with a known latent function,
not the images, text and messy real-world distributions of an actual Kaggle competition. What
conformance establishes is that **the grading, the submission contract and the rejection
behaviour are the real ones** — not that an agent scoring well here would score well there.

The remaining step is unchanged and still costs $0: point this at one real prepared
competition. See [`SOTA-AND-FREE-TIER.md`](SOTA-AND-FREE-TIER.md).
