# Better benchmarking: measuring the instrument

Every other document here asks how good an agent is. This one asks the question
underneath that: **can the benchmark tell?**

An agent-by-competition score matrix is a respondent-by-item matrix. That is not
an analogy — it is the same object, and the statistics that decide whether such
a matrix carries information have been settled for a century. Applying them to
MLE-bench takes about fifty lines and answers questions the benchmark's own
leaderboard cannot:

* How much of the ability spread it reports would survive re-running it on a
  different sample of competitions?
* How many ability levels can it actually resolve, as opposed to print?
* Which competitions move the ranking, and which are paid for in full and move
  nothing?
* What would a quarter of the budget buy?

Reproduce everything below with `mlea instrument`. The data is 25 published
MLE-bench experiments that reported all 75 competitions, shipped in
`src/mlea/data/` with provenance (see `docs/POWER-FINDINGS.md`), spanning medal
rates from 1.3% to 77.8%.

---

## 1. MLE-bench is a good instrument, and it is eroding from both ends

Across the full population of published agents:

```
75 items x 25 agents, ability 0.012-0.778
reliability (alpha)   0.983   (independent items would reach 0.340 at this size)
split-half            0.989
separation G          7.54
distinguishable levels  10.4
dead items            11, 19% of suite cost
```

Reliability 0.983 is high by any standard, and the benchmark separates about ten
distinct ability levels. That is a real result and it deserves saying: MLE-bench
is not a noisy benchmark.

The `independent items would reach 0.340` figure is the reason to trust it.
Cronbach's alpha is unbiased in the mean under independence but **wide** when
subjects are few: shuffling each competition's column independently — which
destroys any shared ability while preserving every competition's own difficulty
— still produces alpha up to 0.34 one time in twenty at this matrix's shape. A
reliability reported without that reference band is not interpretable, and
benchmark papers essentially never report either number.

Now restrict to the eight strongest agents, which is the comparison anyone
actually runs today:

```
75 items x 8 agents, ability 0.440-0.778
reliability (alpha)   0.943   (independent items would reach 0.479 at this size)
separation G          4.08
distinguishable levels   5.8
dead items            29, 37% of suite cost
```

**29 of 75 competitions carry zero information about these eight agents.** The
composition is the interesting part:

| | floor (nobody medals) | ceiling (everybody medals) |
|---|---|---|
| low complexity | 2 | 10 |
| medium | 5 | 5 |
| high | 4 | 3 |

Eleven competitions have never been medalled by *any* of the 25 published agents,
at any seed. Eighteen more are now solved every time by every strong agent.
MLE-bench is saturating from the top and remains unreached at the bottom, and
the middle it can still resolve is 46 competitions wide.

Reliability is a property of an instrument **and the population it is used on**.
This is why: same 75 competitions, 0.983 across all agents and 0.943 across the
strong ones, ten resolvable levels down to six.

## 2. MLE-bench Lite is more than half dead at the frontier

Lite — upstream's 22 low-complexity competitions — is the split most people can
afford. Measured across all 25 agents it holds up well:

```
22 items x 25 agents      alpha 0.959   levels 6.8   dead 2
```

Measured across the eight strongest agents that ran it, it does not:

```
22 items x 8 agents, ability 0.652-0.909
reliability (alpha)   0.791   (independent items would reach 0.509 at this size)
separation G          1.94
distinguishable levels   2.9
dead items            12, 55% of suite cost
```

**Twelve of Lite's 22 competitions cannot distinguish any of the eight strongest
agents**, ten because all eight medal on them and two because none ever has.
What remains resolves about three ability levels. Reliability 0.79 is still above
the 0.51 a shuffled matrix reaches at this size, so Lite is measuring *something*
— but a benchmark that sorts frontier agents into three bins is being read to
three decimal places in papers.

## 3. Lite is a fine *cost* choice and a neutral *information* one

Lite was selected by how long an experienced ML engineer would need, not by how
much each competition tells you. It is worth asking what that cost.

Selecting a subset by information and then scoring it on the same agents proves
nothing: any subset chosen to reproduce a ranking will reproduce that ranking. So
every figure below selects on half the agents and scores the ranking of the other
half, averaged over random splits (`cross_validated_fidelity`). Fidelity is
Kendall's tau-b against the full-75 ranking.

```
   MLE-bench Lite (22 items, cost 22): 0.915 +- 0.043
   a random 22 of the same suite:      0.913 +- 0.042
```

At equal item count, Lite is indistinguishable from drawing 22 competitions at
random. But it is roughly six times cheaper than a random 22, because a random
draw pulls in medium and high tiers. Compared at equal *cost* — which is the
comparison that matters — Lite is close to the best available:

```
cost-fidelity frontier (total cost 475; tau-b vs the full suite, held out)
     budget  share  items  held-out tau-b
         24     5%     22  0.927 +- 0.043
         48    10%     26  0.943 +- 0.024
         95    20%     34  0.975 +- 0.029
        142    30%     42  0.980 +- 0.022
        238    50%     58  0.966 +- 0.028
        356    75%     67  1.000 +- 0.000
```

Cost here is upstream's own complexity tiering, weighted by the midpoints of the
human-hour bands it is defined by (low 1, medium 6, high 15). It is a proxy for
human effort, not a GPU bill; treat it as an ordering. The 50% row falling below
the 30% row is greedy selection being greedy, not a real non-monotonicity —
reported rather than smoothed away.

Two conclusions:

* **Lite is well chosen for its budget.** An information-optimal subset at the
  same cost gains 0.012 tau. There is no easy win here.
* **The win is at 20% of budget.** 34 competitions chosen for information
  reproduce the full-75 ranking at tau-b 0.975 on agents the selection never saw.
  Most of the benchmark's cost buys ranking agreement it already had.

And separately from any subset choice: the 11 never-medalled competitions are
19% of the suite's cost for provably zero contribution to any ranking. Four of
them are in the expensive high tier. They are worth keeping as a *frontier* —
the thing no agent can do yet is the most interesting part of a benchmark — but
they should not be in the denominator of a headline medal rate, and they should
not be re-run on every evaluation.

Dead is relative to the population measured. A competition dead today is a
competition the next agent might open, so this analysis is re-run, not cached.

## 4. The same analysis, turned on this repository's own suite

The generated suite in `mlea.bench` existed to exercise the plumbing. Pointed at
itself, the instrument said it could not do the other job at all:

```
6 items x 6 agents
reliability (alpha)   0.990
distinguishable levels  13.9
ladder recovery: tau-b +0.183
   constant=0.063  naive=0.937  careful=0.862  expert=0.862
```

Reliability 0.990, fourteen resolvable levels, and it **ranks a more competent
agent below a less competent one**. This is the textbook failure the whole
exercise is meant to catch: high reliability with no validity. The suite agreed
with itself perfectly about an ordering that was wrong.

The reference agents are the one case with a known-true ordering — `mlea.baseline`
builds `constant < naive < careful < expert` by adding one competence per rung —
so "does the suite recover it" is checkable rather than assumed. (`linear` and
`tuned` are deliberately excluded from that check: `linear` fits the identical
model to `naive` on clean data, and `tuned` searches capacity and penalty while
skipping data hygiene, so it is not *above* `careful`, it is elsewhere. Asserting
an order that does not exist would make a correct suite look broken.)

Three defects, each found by the item analysis and each with a mechanical cause:

1. **No headroom.** The latent function was a linear term, one pairwise
   interaction and one tanh — all of which the highest-capacity model in the
   simulated field represents *exactly*. The field reached the oracle, the entire
   leaderboard compressed into a band 0.019 AUC wide, and leaderboard percentile
   became noise. `latent_complexity` adds nonlinear functions of random
   projections that no fixed low-order basis captures; headroom went 0.019 →
   0.070 AUC on the competition measured.
2. **A leaderboard that was not trying.** Every simulated team drew its
   regularisation blind, so the top of the leaderboard was a lucky draw. Under
   `field_strength`, a fraction of the field selects capacity and penalty on a
   holdout, the way the upper half of a real leaderboard does.
3. **A ladder with duplicate rungs.** On clean data `naive` and `linear` fit the
   same model, as do `careful` and `expert`. Six named agents were three distinct
   ones until a data pathology was present for them to differ on, so the
   discriminating suite pairs every difficulty with a challenge.

Rebuilt on those three findings:

```
10 items x 6 agents
reliability (alpha)   0.856
distinguishable levels   3.6
ladder recovery: tau-b +1.000
   constant=0.004  naive=0.450  careful=0.594  expert=0.694
   off-ladder: linear=0.636  tuned=0.333
```

Reliability went **down**, from 0.990 to 0.856, and that is the improvement. The
old number was measuring the gap between a constant predictor and everything
else. The new suite recovers the known competence ordering exactly, has no dead
items and no inverted ones, and reports honestly that reaching reliability 0.90
would take about 16 items rather than 10.

Two competitions were removed, on a mechanism rather than on a correlation that
came out the wrong way. Under `leakage` or `shift` with a squared-error metric,
predicting the constant training mean scores at the **top** of the leaderboard:
extrapolating a linear fit through shifted features, or through a feature that is
pure noise at test time, loses to declining to model at all. Measured
discrimination was −0.92 and −0.27; an item a constant predictor wins cannot rank
modelling ability. Both are kept as skill diagnostics (`mlea skills`) and kept
out of a score. The instrument flags this class as an **inverted item**, and it
found one in real MLE-bench too, among the strong agents:
`tensorflow-speech-recognition-challenge`.

---

## What to take from this

For anyone building or reporting on an ML-agent benchmark:

1. **Report reliability with a null band.** One number with a reference
   distribution. It is cheaper than any ablation in the paper and it decides
   whether the ranking means anything.
2. **Report it on the ability band being compared**, not on the whole history of
   agents. Reliability across a 1%-to-78% population says nothing about whether
   two frontier agents are distinguishable.
3. **Publish the item table.** Which competitions discriminate, which are at
   floor, which at ceiling. A benchmark whose dead-item share is rising is
   saturating, and that is visible years before the headline rate hits 90%.
4. **Check a known-true ordering when one exists.** Reliability without validity
   is a suite agreeing with itself, which a broken suite does perfectly.
5. **Publish a cost-fidelity frontier alongside the lite split.** Otherwise
   "lite" is chosen on intuition, and nobody can tell whether the cheap version
   answers the same question.

## Limits

* The 25 experiments are a convenience sample of what has been published, not a
  random draw from a population of agents. Several are AIDE variants, so their
  results are not independent, which inflates reliability to an unknown degree.
* Medal rate is a coarse per-competition outcome — three seeds gives four
  possible values. 5.7% of item variance across the full population is finite-seed
  binomial noise, rising to 13–16% among the strong agents and in Lite, where
  fewer seeds were run. Discrimination is reported both as measured and corrected
  for that attenuation.
* The cost weights are human-hour proxies from upstream's tiering, not measured
  agent spend. A real cost vector can be passed to `cost_frontier`.
* Ability here is the unweighted mean across competitions, matching how MLE-bench
  reports. A full item-response model would estimate item difficulty and
  discrimination jointly; with 25 subjects that is not identifiable, so classical
  test theory is the honest ceiling on this data.

## Commands

```bash
mlea instrument                              # full published MLE-bench
mlea instrument --top 8                      # the frontier band
mlea instrument --split lite --top 8         # Lite, where it is actually used
mlea instrument --frontier                   # cost-fidelity curve, and Lite on it
mlea instrument --source suite               # this repository's generated suite
mlea instrument --source suite --suite legacy
mlea instrument --matrix scores.csv          # any subject-by-item matrix
```
