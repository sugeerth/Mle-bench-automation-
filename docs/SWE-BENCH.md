# SWE-bench: the same treatment, and one thing it does better

```bash
pip install swebench && mlea swe-conform
```

```
15/15 agreed on the full FAIL_TO_PASS and PASS_TO_PASS breakdown

CONFORMANT — our resolver matches the real SWE-bench grader on every
branch of the rule: resolved, unresolved, regressed, and patch-apply failure.
```

## Grading needs no Docker

SWE-bench is a Docker-heavy benchmark, and that obscures a useful fact: **containers exist to
*produce* a test log safely. Deciding what a log means is log parsing plus set comparison**,
and runs anywhere.

So upstream's real `get_eval_report` — its 57 log parsers, its FAIL_TO_PASS / PASS_TO_PASS
logic, its resolution rule — is importable and callable with a plain `pip install swebench`.
Every instance generated here is graded by it, and this package's independent implementation
is checked against it on the full breakdown, not just the boolean.

This is the same finding as on the MLE-bench side: the part everyone assumes needs the heavy
infrastructure turns out not to.

## The thing SWE-bench has that MLE-bench does not

**A mechanical correctness oracle.** An instance is well-formed iff its gold patch flips every
FAIL_TO_PASS test from failing to passing while leaving every PASS_TO_PASS test passing. That
is checkable automatically, for free.

MLE-bench has no equivalent — nothing can decide whether a competition was re-split correctly,
which is why a leaky split produces a valid-*looking* task and why upstream added zero
competitions in 22 months ([`PROPOSAL-mle-bench-live.md`](PROPOSAL-mle-bench-live.md)).

So the generator here uses it. `make_instance` does not *declare* FAIL_TO_PASS and
PASS_TO_PASS — it **measures** them, by running the suite before and after the gold patch, and
refuses to emit an instance that fails its own oracle. There are tests for both refusals: a
gold patch that fixes nothing, and one that fixes the bug and breaks a passing test.

## The rule, and the half that gets skipped

* every FAIL_TO_PASS test must now pass, **and**
* every PASS_TO_PASS test must *still* pass.

The second clause is what makes SWE-bench hard. A patch that fixes the reported bug and
quietly breaks something else is not a resolution — and a harness that checks only the first
clause scores it as one.

So there is a `regressive` reference agent that does exactly that, and a test asserting it is
**not** resolved, with `f2p_failure` empty (the bug *was* fixed) and `p2p_failure` non-empty.

| Reference agent | Patch | Outcome |
| --- | --- | --- |
| `gold` | the measured fix | resolved |
| `noop` | empty | unresolved |
| `regressive` | fixes the bug, breaks a passing test | **regressed** |
| `broken` | a diff against a file that does not exist | patch did not apply |
| `irrelevant` | adds a README | unresolved |

Five agents × three instances, every branch of the rule exercised by a real subprocess rather
than a fixture.

## The bug conformance found

When a patch fails to apply, this package originally marked **every** FAIL_TO_PASS and
PASS_TO_PASS test as failed. Upstream leaves both sets empty.

Both agree the instance is unresolved, so the verdict was never wrong — which is exactly why
comparing only the boolean would have missed it. Upstream is right on the structure: an
unapplied patch means **no tests ran**, so there is no evidence about any of them. Claiming
they all failed would corrupt any downstream aggregate — "which tests does this agent break
most often" would attribute failures to an agent whose patch never executed.

Fixed, with the reasoning in the code and a test named for it. One thing is kept beyond
upstream: `patch_applied` stays on the report, because upstream folds "did not apply" and
"ran and did not fix it" into the same bare `resolved=False`, and those are different failures
with different owners — the same mechanical-versus-capability distinction
[`mlea triage`](../README.md) draws on the MLE-bench side.

## What this is and is not

**Is:** the real resolution logic, verified. The rule, the log format, the parser, the
apply-failure semantics and the regression clause are upstream's, and a disagreement fails
the check.

**Is not:** real SWE-bench instances. These are three small generated repositories with real
bugs, not Django or sympy. The generated instances exist to exercise the harness against the
real grader — SWE-bench's actual difficulty is in navigating a large unfamiliar codebase, and
nothing here measures that.

Running real instances needs the Docker images, which is a different kind of cost from the
Kaggle credential problem on the MLE-bench side: it is heavy but not blocked. That is the
obvious next step here.
