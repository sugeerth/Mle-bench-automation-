# Where the benchmark goes blind, and what to run instead of all of it

`docs/BENCHMARK-QUALITY.md` applies classical test theory. This applies the
method that has largely replaced it, and the first thing it does is overturn two
conclusions from that document — one about what was possible, one about what to
do.

Classical test theory reports one reliability number for a whole population. That
single number is what is wrong with it. A benchmark is not uniformly sharp: it
resolves ability well in some range and not at all outside it, and the range
moves as agents improve. CTT cannot see that, because it has no ability scale —
only a total score, whose meaning depends on which items happen to be in the
suite.

Two things are implemented here, both current practice rather than textbook
history:

* A **two-parameter item-response model** (`mlea irt`), fitted by marginal
  maximum likelihood, which puts competitions and agents on one scale and turns
  "how reliable is this benchmark" into "what can it resolve, where".
* **Anchor-point score estimation** (`mlea anchor`), the method from the
  `tinyBenchmarks` line of work, which estimates the *full* benchmark score from
  a small subset — and is measured here against four alternatives including this
  repository's own earlier method.

---

## 0. Two corrections to the previous round

**"A full item-response model is not identifiable on 25 agents."** That was
wrong. The parameters being estimated are **per item**, not per agent, and each
of the 25×75 cells carries 3 to 39 Bernoulli trials — 13,542 agent-runs in all.
25 is a small number of *people*, which is what the rule of thumb is counting; it
is not a small amount of data. Parameter recovery at exactly this shape is in
`tests/test_irt.py`:

| quantity | recovery at 25 agents × 75 competitions | bias |
| --- | --- | --- |
| agent ability | r = 0.99 | RMSE 0.25 SD |
| competition difficulty (inside the measured ability range) | r > 0.90 | — |
| competition discrimination | r ≈ 0.80 | **23–32% shallow** |

The discrimination attenuation is real and is not hidden: information goes as the
*square* of discrimination, so every standard error below is too **wide**, not
too narrow. The model's own bias is measured by parametric bootstrap
(`--calibrate`): generate data from the fitted parameters at the real seed counts,
refit, regress the estimates on what produced them. The slope is this design's
attenuation — **0.77 ± 0.04** on the published data, and 0.68 on independently
simulated data at the same shape. The bias shrinks steadily as subjects are
added: 0.68 at 25 subjects, 0.86 at 400, 0.98 at 1500. That is consistency,
demonstrated rather than cited.

The practical consequence, in the direction that matters: every resolution figure
below is a **lower bound** on how sharp MLE-bench actually is.

**"Retire the 11 never-solved competitions: 19% of cost for provably zero
information."** This was the previous round's headline recommendation and it is
wrong, in a way CTT structurally could not detect. Under CTT a competition nobody
solves and a competition that tells you nothing are the same item — both have
zero variance. The item-response model separates them, and says the never-solved
eleven are the **only** competitions positioned above the current frontier:

```
hardest competition that can be placed at all      difficulty  +2.73
the 11 never-solved competitions                   difficulty  +4.26
strongest published agent (disarray, 77.8% medals) ability     +2.32
```

They carry no information about *today's* agents. They are also the benchmark's
entire reserve of headroom. Retiring them saves 19% of cost and costs 0.7
percentage points of resolution limit now — and removes every item that would
start discriminating as agents improve. The right move is to stop running them on
every evaluation, not to delete them.

## 1. The model is justified, on cells it never saw

More parameters always fit better in sample, so the comparison is held-out
predictive log-likelihood per trial, five folds:

```
   2pl        -0.198
   rasch      -0.279
   item_mean  -0.409
```

`item_mean` is what classical test theory implicitly assumes — every agent is the
average agent. Rasch adds ability but holds every competition equally
discriminating. The second parameter earns its keep by a wide margin, and both
model variants beat the CTT-implied baseline decisively.

## 2. MLE-bench is sharpest at a 39% medal rate, and has ~3.6 points left

One unit of ability is one standard deviation of published-agent ability. At the
standard three seeds per competition:

```
    ability  medal rate      SE   min gap
      -1.79        2.6%   0.230      2.1pp
      -0.97        8.4%   0.106      3.5pp
      -0.16       24.5%   0.063      4.9pp
       0.25       36.7%   0.061      5.3pp   <- near the minimum (exactly at +0.33, 38.9%)
       1.07       58.3%   0.085      4.1pp
       1.89       71.0%   0.095      4.2pp
       2.30       77.1%   0.110      3.9pp   <- strongest published agent
       2.71       81.3%   0.149      3.4pp
       3.12       84.2%   0.189      3.3pp
```

`min gap` is the medal-rate difference two agents need before the
benchmark can call them apart at 95% confidence — `2.77 × SE` converted through
the slope of the expected-rate curve.

* The benchmark is sharpest at an expected medal rate of **38.9%** (SE 0.061).
* Its resolution **halves** by a **78.7%** medal rate.
* The strongest published agent sits at **77.8%**. The halving point is already
  underneath it.
* At SE ≤ 0.15 it resolves ability up to an expected medal rate of **81.4%** —
  about **3.6 percentage points of headroom** above current SOTA.

Marginal reliability is 0.986, which agrees with the 0.983 alpha from the CTT
analysis. The agreement is the point: the model reproduces the number CTT gives
and then says the thing CTT cannot.

## 3. The lever is seeds, not harder competitions

The reflex when a benchmark saturates is to build harder tasks. For MLE-bench
that is not yet the binding constraint — the limit at the top is estimation
noise, and noise is bought off with seeds:

```
   seeds per competition    resolves to SE <= 0.15 up to
            3                    81.2% medal rate
            6                    90.8% medal rate
           12                    96.4% medal rate
           24                    97.8% medal rate
```

Doubling seeds buys about ten points of headroom; doubling again buys six more.
The competitions to run them on already exist.

At **constant budget**, using upstream's complexity tiers as the cost model
(low 1, medium 6, high 15; 75 competitions × 3 seeds = 1425 units):

```
   all 75, 3 seeds (status quo)            75 x 3.0   cost 100%   resolves to 81.4%
   drop the 11 never-solved, 3 seeds       64 x 3.0   cost  81%   resolves to 80.7%
   drop never-solved, reinvest the saving  64 x 3.7   cost  99%   resolves to 81.6%
   drop never-solved + 2 flat, reinvest    62 x 3.8   cost 100%   resolves to 81.7%
   all 75, 6 seeds                         75 x 6.0   cost 200%   resolves to 91.1%
```

So the retire-and-reseed swap is worth +0.3 points at the same price — real but
small, and nothing like the win the previous round's dead-item framing implied.
Extending the benchmark's useful life materially costs money: roughly 2× budget
for 10 points. Stated plainly because the opposite is the kind of claim a tool
like this exists to prevent.

## 4. Estimating the full score from a fifth of the benchmark

The ranking question — "does a subset rank agents the same way" — is the weaker
one. A ranking is only useful against agents you have also run, while a *score* is
comparable to a published one. So: run 20 of the 75; what is your number on all
75, and how wrong is it?

Two estimators, on identical subsets:

* **naive** — the mean over the items you ran. What everyone reports.
* **p-IRT** — estimate ability from the observed items, then add back the model's
  prediction for the ones you skipped.

Five selection strategies, all seeing only the training agents: `random`,
`stratified` by observed difficulty (what a lite split approximates),
`greedy_alpha` (this repository's earlier ranking-oriented method),
`max_information` (the Fisher-optimal adaptive-testing choice), and `anchor`
(cluster items by fitted response curve, take one per cluster — the
tinyBenchmarks construction).

Scored **leave-one-agent-out**: the model is fitted and the subset chosen without
ever seeing the agent whose score is then predicted.

```
k=20  (error in predicting the full 75-competition medal rate)
   stratified       pirt   MAE  2.31pp  max 12.26pp  bias -0.57pp
   anchor           pirt   MAE  2.64pp  max 12.75pp  bias -0.76pp
   random           pirt   MAE  2.72pp  max 12.87pp  bias +0.05pp
   max_information  pirt   MAE  2.82pp  max 14.01pp  bias -0.39pp
   greedy_alpha     pirt   MAE  3.19pp  max 22.87pp  bias +0.73pp
   stratified       naive  MAE  3.44pp  max 12.44pp  bias +2.59pp
   random           naive  MAE  4.88pp  max 20.22pp  bias +1.13pp
   anchor           naive  MAE 10.53pp  max 20.78pp  bias +10.46pp
   max_information  naive  MAE 20.31pp  max 40.22pp  bias +16.56pp
   greedy_alpha     naive  MAE 24.92pp  max 56.00pp  bias +22.48pp
```

**The estimator matters roughly twenty times more than the selection.** At k=20,
p-IRT spans 2.31 to 3.19pp across all five selectors — a range of 0.9 points.
Naive spans 3.44 to 24.92pp — a range of 21.5 points. Choosing how to *estimate*
is the entire game; choosing *which* competitions is a rounding error by
comparison.

The mechanism is bias, not noise. A selection rule optimised for information
picks competitions near the middle of the difficulty range, so the raw mean over
them overstates the full-suite rate by 16.6 points — systematically, in the same
direction, every time. p-IRT knows how hard the subset was and corrects for it;
every p-IRT bias in the table is inside ±1.3 points.

Across subset sizes, best result at each k:

| k | best | MAE | best naive | MAE |
| --- | --- | --- | --- | --- |
| 5 | stratified + p-IRT | 4.14pp | stratified + naive | 5.88pp |
| 10 | stratified + p-IRT | 3.41pp | stratified + naive | 4.02pp |
| 20 | stratified + p-IRT | 2.31pp | stratified + naive | 3.44pp |
| 30 | stratified + p-IRT | 1.80pp | random + naive | 3.48pp |
| 45 | greedy_alpha + p-IRT | 0.87pp | random + naive | 2.97pp |

And this repository's own `greedy_alpha` — built last round to preserve a
*ranking* — is the **worst** p-IRT selector at every small k (8.44pp at k=5
against stratified's 4.14pp). It does not transfer to estimation, and finding
that out required running it against alternatives rather than assuming the
earlier method generalised.

## What to take from this

1. **Report the resolution curve, not a reliability number.** A benchmark has a
   range it resolves and a point beyond which it does not, and only the second
   tells you when it expires.
2. **Never-solved tasks are the headroom, not the dead weight.** Stop re-running
   them every evaluation; do not retire them. They are the only items positioned
   above the frontier.
3. **Estimate, do not average, when you run a subset.** It is free, it removes a
   16-point bias, and it makes a cheap run comparable to a full one.
4. **Measure your own method's bias.** A parametric bootstrap at the real design
   costs a minute and says which outputs to lean on. Here: ability yes,
   difficulty yes within range, discrimination with a 25% correction.
5. **A benchmark that saturates may need seeds rather than harder tasks.** Check
   which before building new tasks — the costs differ by an order of magnitude.

## Limits

* The 25 experiments are a convenience sample of published results, not a random
  draw. Several are AIDE variants, so they are not independent, and the ability
  distribution is whatever people chose to publish.
* A 2PL has no upper asymptote. A competition can be unmedallable for reasons
  unrelated to ability — ambiguous grading, a broken prepared split — and a 3PL
  or 4PL would model that as a ceiling instead of as extreme difficulty. Not
  fitted here: with 11 items at zero and no agent above the frontier, the
  asymptote and the difficulty are not separable on this data. It is the obvious
  next model, not a detail.
* Difficulty is only a measurement inside the measured ability range. Outside it,
  the model is reading the flat tail of a response curve, so the 75 competitions
  come back in three classes: **62 located** (the data pins them), **11
  unidentified** (never solved, difficulty is a prior bound — and all 11 land on
  the same value, so it is one bound rather than 11 measurements), and **2
  extrapolated** (slope too flat to place —
  `tensorflow-speech-recognition-challenge` fits at difficulty +10.4 with a slope
  of 0.33, the model saying "this barely tracks ability" in a form that reads as
  "this is impossibly hard"). Only the 62 are used for any curve or any
  prediction.
* The cost model is upstream's human-hour complexity tiers, not measured GPU
  spend. Treat cost figures as an ordering.
* Medal rate is a coarse outcome. Leaderboard percentile would carry more
  information per run, but is not published per-competition for most experiments.

## Commands

```bash
mlea irt                                  # the resolution curve
mlea irt --compare-models                 # 2PL vs Rasch vs item-mean, held out
mlea irt --calibrate                      # measure this design's own bias
mlea irt --seeds 6 --target-se 0.15       # what more seeds would buy
mlea anchor -k 20                          # the head-to-head, one subset size
mlea anchor -k 5 10 20 30 45 -v            # the full table above (~6 min)
```
