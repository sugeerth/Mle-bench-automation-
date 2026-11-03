# MLE-bench: current SOTA, and how to run it for $0

**Last verified:** August 2026. Leaderboards and free tiers both move fast — re-check before relying on any number here.

---

## Part 1 — What the state of the art actually is

### The headline number

**MLEvolve — 65.3% ± 0.8% any-medal on the full 75-competition set**, 12 h budget per task,
Gemini-3.1-Pro-preview, 3 seeds. It is #1 among open-source methods and is source-available
at [`InternScience/MLEvolve`](https://github.com/InternScience/MLEvolve).

Its breakdown by complexity is the more useful part:

| Split | MLEvolve any-medal |
| --- | --- |
| Low (lite, 22 comps) | 80.3 ± 1.5 |
| Medium | 64.0 ± 0.9 |
| High | 46.7 ± 0.0 |

Nearby on the same leaderboard: proprietary MARS+ / AIBuildAI at 62.7–63.1%, open-source
ML-Master 2.0 at 56.4%, Leeroo at 50.7%.

### For trajectory

The original MLE-bench paper (Oct 2024) had AIDE + o1-preview at **16.9%** pass@1 on the full
set (34.1% at pass@8). So the full-set medal rate has gone from ~17% to ~65% in under two years.

### ⚠️ Most published MLE-bench numbers are not comparable to each other

This is the single most important thing to understand before quoting any figure. A search for
"MLE-bench SOTA" returns, all at once:

| Claim | Split | Comparable to MLEvolve's 65.3%? |
| --- | --- | --- |
| MLEvolve 65.3% | full 75 | — (this is the reference) |
| R&D-Agent (GPT-5) 68.2 ± 2.6% | **lite, 22 comps** | ❌ no |
| R&D-Agent 35.1% | full 75 | ✅ yes |
| MLE-STAR 44% | full 75 | ✅ yes |
| EurekAgent 85.71% | **a selected lite subset** | ❌ definitely not |

Lite is the *low-complexity* subset, so lite numbers run 15–25 points higher than full-set
numbers for the same system. A "selected subset" of lite is higher still. Time budget (12 h vs.
24 h) and pass@k vs. pass@1 also move the number substantially.

Whenever you see an MLE-bench percentage, ask for **split, budget, seeds, and k** before
comparing it to anything. This is exactly the failure mode [`PLAN.md` §6](PLAN.md#6-comparing-two-agents-without-fooling-yourself)
guards against with a split hash in the run key — and it is not a hypothetical concern, it is
what the public literature currently looks like.

---

## Part 2 — Running it for $0

### What free gets you, honestly

**Achievable at $0:** a working end-to-end pipeline — prepare → run agent → grade → real medal
verdict — on roughly **5–8 of the smallest lite competitions, 1 seed, short budget**. That is a
genuine directional signal and it is enough to build and debug everything in
[`PLAN.md`](PLAN.md) Phase 0 and most of Phase 1.

**Not achievable at $0:** any number comparable to the leaderboard above. SOTA is 75
competitions × 3 seeds × 12–24 h on 440 GB / A10-class nodes with frontier models. That is
~$30–43k of compute. No free tier is within two orders of magnitude of it, and nobody should
imply otherwise.

So the goal of the free path is **"is my harness correct and is my agent roughly working"**,
not "what is my benchmark score".

### The stack

| Layer | Free option | Limits (verify before relying) |
| --- | --- | --- |
| Compute | **Kaggle Notebooks** | ~30 h/week GPU (T4 ×2 / P100, 16 GB VRAM), 12 h max session, ~20 GB working disk |
| Compute | Google Colab free | Less predictable; aggressive idle disconnects |
| Compute | **Your own laptop** | No quota at all — viable for the text/tabular competitions below |
| LLM | Google AI Studio (Gemini Flash) | ~1,500 requests/day, ~10–15 RPM, no card |
| LLM | Groq | ~30 RPM, ~1,000 req/day, ~100k tokens/day; very fast |
| LLM | Cerebras | ~1M tokens/day |
| LLM | OpenRouter free models | ~20 RPM, ~50 req/day until $10 credit is purchased |
| LLM | Ollama, local | Unlimited, but a 7–14 B model on a T4 is a weak ML engineer |
| Data | Kaggle API | Free; needs an account and per-competition rule acceptance |

Kaggle Notebooks is the best single choice: the free GPU quota is predictable, and the
competition data is *already on Kaggle*, so you skip the slowest part of `mlebench prepare`.

**Caveat on free LLM tiers:** most send your traffic to the provider for training by default.
Fine for a public Kaggle benchmark, not fine if you ever point this harness at private data.

### The key technical unlock: you don't need Docker

MLE-bench's official harness is Docker-based, and neither Kaggle nor Colab gives you Docker.
That looks like a blocker but isn't:

- `mlebench prepare` and `mlebench grade-sample` are **plain pip-installed CLI commands**. They
  work fine with no container.
- Docker only provides the agent's *sandbox* — resource limits, internet blocking, isolation.

So on free infrastructure you run the agent directly in the notebook and grade with the real
grader. You get **correct scores** and lose **isolation guarantees**. For "is my pipeline
right?", that trade is entirely acceptable, as long as you never present the resulting numbers
as benchmark-comparable. See the honesty caveat at the end.

### Pick small competitions

`mlebench prepare --lite` pulls ~158 GB and will not fit in a 20 GB notebook. Prepare **one
competition at a time** with `-c`, and start with the small ones.

From the 22-competition lite split, ordered roughly by how painless they are:

**Tiny, CPU-only — start here (no GPU quota burned at all):**
- `random-acts-of-pizza` — small text/JSON, binary classification
- `detecting-insults-in-social-commentary` — small text
- `spooky-author-identification` — small text, multiclass
- `nomad2018-predict-transparent-conductors` — small tabular regression
- `leaf-classification` — small tabular + images

**Small, GPU-helpful:**
- `aerial-cactus-identification` — small image binary classification
- `denoising-dirty-documents` — small image-to-image

**Avoid on free tier** (multi-GB or long-training): `siim-isic-melanoma-classification`,
`ranzcr-clip-catheter-line-classification`, `histopathologic-cancer-detection`,
`new-york-city-taxi-fare-prediction`, `aptos2019-blindness-detection`.

### Step by step

```bash
# 1. Install upstream (works anywhere, no Docker needed)
git clone https://github.com/openai/mle-bench.git && cd mle-bench
pip install -e .

# 2. Kaggle credentials -> ~/.kaggle/kaggle.json, chmod 600
#    On Kaggle Notebooks these are already available via the Add-ons menu.
#    You must ALSO click "Late Submission"/accept rules on each competition's
#    Kaggle page once, or the API download returns 403.

# 3. Prepare ONE small competition (not --lite, which is ~158GB)
mlebench prepare -c random-acts-of-pizza

# 4. Sanity-check the grader before involving any agent at all
mlebench grade-sample <path-to-sample-submission.csv> random-acts-of-pizza

# 5. Now run an agent against the prepared data, pointed at a free LLM endpoint
#    (most scaffolds accept an OpenAI-compatible base_url + api_key)

# 6. Grade what it produced
mlebench grade-sample <agent-output>/submission.csv random-acts-of-pizza
```

**Do steps 3–4 before touching an agent.** Getting a known submission graded end to end is the
entire value of Phase 0, it costs nothing, and it will surface the Kaggle-credentials and
rule-acceptance friction immediately rather than three hours into an agent run.

### Suggested free budget

| | |
| --- | --- |
| 5 CPU-only competitions × ~2 h | on your own laptop — no quota consumed |
| 2 GPU competitions × ~4 h | ~8 h of Kaggle's ~30 h weekly quota |
| LLM calls | comfortably inside Gemini Flash's ~1,500/day |
| **Total** | **$0**, one weekend, ~1 week of quota headroom to spare |

### Two honesty requirements

1. **Never report free-tier runs as MLE-bench scores.** Different hardware, no sandbox, shorter
   budget, 1 seed. In terms of [`PLAN.md`](PLAN.md), these runs carry a different
   `container_config_hash` and `split_hash` and must never be compared against reference runs.
2. **Without Docker there is no internet block.** The agent can, in principle, look up the
   public winning solution for the competition it is being graded on. That is the contamination
   problem in [`PROPOSAL-anytime-eval.md`](PROPOSAL-anytime-eval.md#secondary-idea-a-contamination-probe),
   with the guardrail removed. Fine for pipeline debugging; disqualifying for any real number.

### If you later want free-but-sandboxed

Several providers offer one-time credits that get you real Docker-capable machines without a
recurring bill — enough for a proper lite sweep. That is the natural bridge between this page
and Phase 2 of the plan, but it is credits, not free, so it is out of scope here.

---

## Sources

- [MLEvolve](https://github.com/InternScience/MLEvolve) — verified 65.3% figure and split breakdown
- [MLE-bench paper (arXiv 2410.07095)](https://arxiv.org/pdf/2410.07095) — original 16.9% / 34.1% baselines
- [MLE-bench repo](https://github.com/openai/mle-bench) — CLI, splits, Docker environment
- Free-tier limits compiled from provider comparisons, Aug 2026 — **treat as approximate**
