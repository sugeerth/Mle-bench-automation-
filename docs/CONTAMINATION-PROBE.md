# A contamination probe that is known to work

```bash
mlea probe --out probe
```

## The problem with the existing evidence

MLE-bench's contamination check returned a null: rewriting competition descriptions to
remove identifying information moved the medal rate from 8.5% to 8.4%. That is the headline
reason to believe the benchmark is not contaminated.

Two things are wrong with treating it as such.

**It was measured at the floor.** 8.5% means almost nothing was being solved, so there was
almost nothing for memorisation to inflate. The test had close to zero power. SOTA is now
65.3%, and nobody has re-run it.

**It was never shown to be able to detect anything.** The test rewrote *descriptions*. It
did not alter the data. A model that recognises a dataset by its contents rather than its
framing would pass straight through it, and the null result would look identical. **No
published contamination test for this benchmark has ever demonstrated that it fires when
contamination is present.**

That question cannot be settled on real competitions. Matching two Kaggle competitions on
difficulty is guesswork, and any systematic difficulty difference is indistinguishable from
memorisation — the bias that made [`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md)
unworkable. On generated competitions it is settled by construction.

## The design

A two-by-two: a **simulated memoriser** and an **honest solver**, each on an **original** and
on a **clone**.

A clone is the same problem wearing a different surface. Two strengths:

| Transform | What it changes | Difficulty delta |
| --- | --- | --- |
| `relabel` | Column names, column order, row order, row ids. Values untouched. | **exactly 0.000000** |
| `rescale` | All of the above, plus a positive per-column affine map on the values. | measured, `< 0.001` |

`relabel` preserves difficulty *exactly* — a permutation-invariant model scores identically,
which is asserted in the tests. That is the property no pair of real competitions can have.

The memoriser is handed the answers for the originals only, and recognises rows by
fingerprint. Keyed on **rows**, not ids: a model that has seen a dataset recalls the mapping
from rows to labels and can apply it however the rows are presented. Keying on ids would make
any reissue of ids look like a successful defence, which it is not.

## Results

```
   agent       seen the original
   memorizer               100%
   honest                    0%

   agent      transform   still recalled   pctile gap   medal gap
   memorizer  relabel              100%       +0.0%        +0%
   memorizer  rescale                0%       +8.9%        +0%
   honest     relabel                0%       -2.4%        +0%
   honest     rescale                0%       +0.8%        +0%

   PASS  negative control: an honest solver shows no gap (worst |2.4%|)
   PASS  positive control: rescale separates the memoriser (+8.9% vs +0.8%)
   PASS  the memoriser does recall what it was shown (100% of originals)
   PASS  renaming and reordering alone does NOT defeat value-keyed recall
   PASS  altering the values does defeat it (0% of rescale clones recalled)
```

### Three findings

**1. Renaming does not defeat recall.** The memoriser recalled **100%** of `relabel` clones —
different column names, different column order, different row order, different ids. Only
altering the values defeated it. A probe that rewrites descriptions or renames columns is
testing whether the model reads labels, not whether it remembers data. **The 2024 obfuscation
test was of exactly that strength.**

**2. Medal rate cannot see contamination.** The memoriser's advantage was worth **+8.9
percentile points and +0% medal rate**. Any-medal is binary and saturates: an agent good
enough to medal without recall and one that recalls perfectly are the same number. So
MLE-bench's own headline metric is blind to any contamination effect that does not flip a
medal — and the stronger agents get, the more effects fall into that blind spot. **A probe
must score on percentile or raw score, never on medal rate.**

**3. Percentile compresses at the top too.** A perfect memoriser and a competent honest
solver both sit in the leaderboard's top few percent, so a total advantage shrinks to single
digits. This is the "crowded leaderboard" pathology upstream already flags for three of its
competitions, and it means contamination is hardest to detect exactly where agents are
strongest — which is now.

## What this does and does not establish

**Does:** that a clone-based probe fires when memorisation is present, that value-level
transformation is required to defeat recall, and that medal rate is the wrong unit to measure
it in. These are properties of the *method*, and they transfer.

**Does not:** anything about whether MLE-bench is actually contaminated. These competitions
are generated; no model has seen them. The memoriser is simulated — it is handed the answers
rather than having learned them. This is a **positive control**, which is the thing missing
from the existing evidence, not a measurement of the real benchmark.

## What it implies for measuring the real thing

1. Re-run the obfuscation test at today's capability, not at 8.5%.
2. Transform the **data**, not the description. Renaming is not enough.
3. Score on percentile or raw score. Medal rate cannot see it.
4. Use at least 6 competitions — below that a paired test cannot reach p<0.05 at any effect
   size ([`POWER-FINDINGS.md` §7](POWER-FINDINGS.md)).
5. Report the probe's own positive control alongside the result. A null from a probe of
   unknown sensitivity is not evidence of absence, which is the whole lesson of the 2024
   result.
