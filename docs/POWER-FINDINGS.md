# Power findings: what our sweeps can and cannot detect

Produced by `mlea power`. Reproduce with the commands at the bottom.

> **Grounded in real data, not a modelling assumption.** Per-competition medal
> probabilities are resampled from published MLE-bench runs — upstream ships raw
> per-seed grading reports under `runs/`, git-LFS tracked (which is why they are
> invisible to ordinary raw fetches and easy to miss). The shipped table is
> `src/mlea/data/mlebench_per_competition_medals.csv`, 31 experiments; it is verified
> by reproducing the paper's headline 16.9% for `models-o1-preview-aide`, and there is
> a test that fails if it ever stops doing so.
>
> Dollar figures use $150/run from [`PLAN.md` §5](PLAN.md#5-cost-model) and remain
> order-of-magnitude.

---

## 0. The correction that produced these numbers

An earlier version of this document modelled per-competition medal rates as a Beta with
a **concentration of 3.0**, chosen by judgement. That was wrong, and wrong in the
optimistic direction.

Method-of-moments fits to the published runs, binomial noise subtracted:

| Experiment | Competitions | Seeds | Mean rate | Fitted concentration |
| --- | ---: | ---: | ---: | ---: |
| scaffolding-gpt4o-aide | 75 | 39 | 0.087 | **0.99** |
| models-o1-preview-aide | 75 | 21 | 0.170 | **0.80** |
| aira-dojo | 75 | 20 | 0.316 | **0.70** |
| pievolve | 53 | 6 | 0.803 | 0.64 |
| famou-agent | 75 | 9 | 0.559 | 0.36 |

Median **0.70** across nine experiments; the three with the most seeds give 0.99, 0.80,
0.70. The old 3.0 understated between-competition variance by roughly a factor of four.

And a Beta fits the shape poorly regardless, because the real distribution is
**zero-inflated**, not smooth. For o1-preview with AIDE, over ~21 seeds:

| Per-competition medal rate | Competitions |
| --- | ---: |
| **exactly 0** | **42** |
| (0, 0.25] | 15 |
| (0.25, 0.75] | 11 |
| (0.75, 1) | 5 |
| **exactly 1** | **2** |

56% of competitions are *never* medalled. So the model now **resamples the observed
rates directly** and assumes no shape at all. The Beta path survives only for
hypothetical designs at rates nobody has published, with the default concentration
corrected to 0.7.

**Cross-check:** the corrected parametric model and the data-driven one land in the same
place — lite MDE −20.3% (fitted Beta) vs −20.7% (empirical resample), against −23.4%
under the old assumption. Two independent routes agreeing is the reason to believe
either. There is a test asserting they stay within 4 points.

---

## 1. The contamination experiment as proposed is far worse than first thought

[`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md) originally costed a first
experiment at "8 matched pairs × 3 seeds ≈ 48 runs ≈ $7–9k". On the assumed model that
looked like an MDE of −54%. On real rates:

| pairs | runs | est. cost | power at −20% | at −30% | at −55% | MDE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **8** | 48 | $7,200 | **4%** | **14%** | 57% | **−90%** |
| 12 | 72 | $10,800 | 24% | 55% | 97% | −40% |
| **16** | 96 | $14,400 | 48% | **83%** | 100% | **−29%** |
| 24 | 144 | $21,600 | 78% | 97% | 100% | −21% |
| 32 | 192 | $28,800 | 91% | 100% | 100% | −17% |
| 48 | 288 | $43,200 | 99% | 100% | 100% | −12% |

An MDE of **−90%** means the 8-pair design can detect essentially nothing: it would miss
a −30% contamination effect **86%** of the time, and even a full SWE-bench-sized −55%
collapse would be missed **43%** of the time. It is not a coin flip on the disaster case;
it is worse than a coin flip.

**16 pairs remains the practical floor** (MDE −29%), and 24 pairs is where moderate
effects become detectable. The earlier conclusion survives the correction — the
correction just makes the rejected design look even worse.

### The binding constraint is calendar, not budget

Each post-cutoff competition supplies one pair, and Kaggle yields perhaps 15–30
substantive competitions a year, so 16 pairs is **6–12 months of accumulation**. Build the
ingest pipeline early and run the analysis late.

## 2. Matching affects validity, not power

| matching sd | power at −30% | MDE |
| ---: | ---: | ---: |
| 0.00 (perfect) | 87% | −27% |
| 0.10 | 85% | −29% |
| 0.15 | 83% | −29% |
| 0.25 (poor) | 74% | −33% |

Across its whole plausible range, matching quality is worth ~13 points of power. It
protects against **bias** — post-cutoff competitions that are systematically harder are
indistinguishable from contamination — which no number of extra pairs fixes.

## 3. At a fixed budget, buy competitions rather than seeds

All rows cost 96 runs:

| design | power at −30% |
| --- | ---: |
| 48 pairs × 1 seed | **92%** |
| 24 pairs × 2 seeds | 89% |
| 16 pairs × 3 seeds | 83% |
| 12 pairs × 4 seeds | 71% |
| 8 pairs × 6 seeds | 40% |

Power for a between-arm mean shift is driven by units, and the permutation test's
resolution floor depends on units alone.

**But §4 is the reason not to read this as "use 1 seed".**

## 4. Seeds are not optional, and the data says how many

Directly from the published runs — the fraction of competitions where the same agent
does *not* get the same answer on every seed:

| Experiment | Seeds run | Non-unanimous competitions | Expected flips at 3 seeds |
| --- | ---: | ---: | ---: |
| models-o1-preview-aide | ~21 | 31/75 (**41%**) | 13.5/75 (18%) |
| aira-dojo | ~20 | 39/75 (**52%**) | 19.0/75 (25%) |
| scaffolding-gpt4o-aide | ~39 | 21/75 (**28%**) | 8.7/75 (12%) |

**Between a quarter and a half of competitions are genuinely unstable across seeds.** At
the conventional 3 seeds, 12–25% of competitions would be expected to disagree with
themselves. A single-seed sweep is not a cheaper measurement of the same thing — it is a
coin flip on a fifth of the benchmark. Spend marginal budget on units, but never below 3
seeds.

## 5. Regression gating: what a sweep can actually catch

| design | units | seeds | est. cost | MDE (drop) |
| --- | ---: | ---: | ---: | ---: |
| lite-regression | 22 | 3 | $19,800 | **−20.7%** |
| full-regression | 75 | 3 | $67,500 | **−9.4%** |
| o1-preview-full | 75 | 3 | $67,500 | −21.5% |

A 3-seed lite sweep detects roughly a **21-point** regression and nothing subtler — the
quantified form of [`PLAN.md` §6](PLAN.md#6-comparing-two-agents-without-fooling-yourself).
The third row is the same design at the paper's 16.9% baseline: detecting a *drop* is much
harder near the floor, because there is little left to lose.

## 6. Drops are easier to detect than gains near a high base rate

Medal probability is capped at 1. At the lite pool's 77% mean, a regression is fully
visible while an improvement is squashed by the ceiling — on lite, **no** improvement
reaches 80% power at any effect size. Convenient, since regressions and contamination
gaps are both drops, but it means "we could not detect an improvement" is nearly
uninformative on lite.

## 7. Designs below 6 units cannot produce a significant result at all

A paired sign-flip test over `n` units has a p-value floor of `3/(2ⁿ+1)`: 0.091 at n=5,
0.176 at n=4 — above α=0.05 whatever the effect. `mlea` flags this
(`underpowered_by_construction`) rather than reporting a non-significant p-value that
reads like evidence of no effect.

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

## What would still change these numbers

The reference arm. All presets resample `pievolve` (mean 0.803, closest published run to
current SOTA territory), which has only ~6 seeds per competition — so its per-competition
rates carry real binomial noise, which inflates apparent heterogeneity somewhat. The
high-seed experiments are all much weaker agents. **A high-seed run of a current
frontier agent would be the single most valuable input to this document**, and none is
published.
