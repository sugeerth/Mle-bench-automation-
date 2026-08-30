# Power findings: what our sweeps can and cannot detect

Produced by `mlea power`. Reproduce with the commands at the bottom.

> **These are model-based, not empirical.** Per-competition medal probabilities are
> drawn from a Beta with the stated base rate and a heterogeneity (concentration)
> of 3.0. Heterogeneity is the parameter power is most sensitive to and it is an
> assumption — re-run with `--heterogeneity` once real per-competition rates exist.
> Dollar figures use $150/run from [`PLAN.md` §5](PLAN.md#5-cost-model) and are
> order-of-magnitude.

---

## 1. The contamination experiment as proposed was a coin flip

[`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md) costed a first experiment at
"8 matched pairs × 3 seeds × 2 conditions ≈ 48 runs ≈ $7–9k". Its actual resolving power:

| pairs | runs | est. cost | power at −20% | at −30% | at −55% | MDE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **8** | 48 | $7,200 | **10%** | **27%** | 83% | **−54%** |
| 12 | 72 | $10,800 | 31% | 63% | 99% | −37% |
| **16** | 96 | $14,400 | 48% | **80%** | 100% | **−30%** |
| 24 | 144 | $21,600 | 70% | 96% | 100% | −23% |
| 32 | 192 | $28,800 | 84% | 99% | 100% | −19% |
| 48 | 288 | $43,200 | 96% | 100% | 100% | −14% |

The 8-pair design has a minimum detectable effect of **−54%**. The SWE-bench-Live gap
is roughly −55 points. So the proposed experiment could detect a catastrophe of exactly
that magnitude and essentially nothing else: a −30% contamination effect — which would
still be a major finding about the benchmark — would be missed **73% of the time**.

Spending $7.2k on a design whose only outcome is "the disaster case, maybe" is not worth
doing. **16 pairs is the practical floor** at ~$14.4k, and 24 pairs is where the result
starts being informative about moderate effects.

### The binding constraint is calendar, not budget

Kaggle yields perhaps 15–30 substantive competitions a year, and each post-cutoff
competition supplies one pair. So a 16-pair experiment is roughly **6–12 months of
accumulation**, and 24 pairs is a year or more. The rolling split has to start collecting
long before it can answer anything — which is an argument for building the ingest
pipeline early and running the analysis late, not for deferring the whole thing.

## 2. Matching affects validity, not power — the proposal had this wrong

The proposal called matched-pair quality "the whole experiment" and listed it as a High
risk to the result. On power, that is simply not what the simulation shows:

| matching sd | power at −30% | MDE |
| ---: | ---: | ---: |
| 0.00 (perfect) | 86% | −29% |
| 0.10 | 84% | −29% |
| 0.15 | 80% | −30% |
| 0.25 (poor) | 75% | −32% |

Going from perfect matching to poor matching costs ~11 points of power. Variance here is
dominated by between-competition heterogeneity and binomial seed noise, not by matching
residual.

**Matching still matters — for a different reason.** It protects against *bias*: if
post-cutoff competitions are systematically harder, that difficulty difference is
indistinguishable from contamination and the experiment returns a confidently wrong
answer. That is a validity threat, not a variance threat, and no amount of extra pairs
fixes it. The proposal has been corrected to say so.

## 3. At a fixed budget, buy competitions rather than seeds

All rows cost 96 runs:

| design | power at −30% |
| --- | ---: |
| 48 pairs × 1 seed | **89%** |
| 24 pairs × 2 seeds | 86% |
| 16 pairs × 3 seeds | 80% |
| 12 pairs × 4 seeds | 75% |
| 8 pairs × 6 seeds | 57% |

Power for detecting a between-arm mean shift is driven by the number of units, and the
paired permutation test's resolution floor depends on units alone — no seed count rescues
too few competitions (`seeds_needed` returns `None` for such designs, which is tested).

**But do not read this as "use 1 seed".** MLE-bench convention is ≥3 seeds, single-seed
per-competition rates are 0/1 with no within-competition variance estimate, and results
would not be comparable to published numbers. The honest reading is that seeds beyond 3
are a poor purchase for *this* question, and marginal budget should go to breadth.

## 4. Regression gating: what a lite sweep can actually catch

| design | units | seeds | est. cost | MDE (drop) |
| --- | ---: | ---: | ---: | ---: |
| lite-regression | 22 | 3 | $19,800 | **−22.7%** |
| full-regression | 75 | 3 | $67,500 | **−12.1%** |

A 3-seed lite sweep detects roughly a **23-point** regression and nothing subtler. This
is the quantified version of [`PLAN.md` §6](PLAN.md#6-comparing-two-agents-without-fooling-yourself):
a 6-point "improvement" on lite is noise, and reporting it as a result is wrong. Every
comparison must publish its MDE alongside the difference, which `mlea compare` does.

## 5. Drops are easier to detect than gains near a high base rate

Medal probability is capped at 1. At the 80.3% lite base rate, an improvement is squashed
by that ceiling while an equal-sized regression is fully visible — on lite, a −22.7% drop
is detectable at 80% power while **no** improvement reaches 80% power at any effect size.

Convenient, since regressions and contamination gaps are both drops. But it means
"we could not detect an improvement" is close to uninformative on lite, and demonstrating
that an agent got *better* needs the full split or a lower-base-rate subset.

## 6. Designs below 6 units cannot produce a significant result at all

A paired sign-flip test over `n` units has a p-value floor of `3/(2ⁿ+1)`. At n=5 that is
0.091 and at n=4 it is 0.176 — above α=0.05 regardless of effect size. Such a design
cannot return a significant result however large the true difference. `mlea` flags this
(`underpowered_by_construction`) rather than reporting a non-significant p-value that
looks like evidence of no effect.

---

## Reproducing

```bash
pip install -e .
mlea power --design live-gap
mlea power --design lite-regression
mlea power --design full-regression
mlea power --design live-gap --units 16
mlea power --design live-gap --units 16 --effect -0.30
mlea seeds --design live-gap --units 16 --effect -0.30
```

## What would change these numbers

The heterogeneity assumption is the big one. After the first real lite sweep, fit
per-competition medal rates and re-run every number here — if competitions are more
uniform than assumed, all MDEs improve; if more polarised, they get worse.
