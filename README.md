# Kivi — semantic memory

A working local application: Kivi learns durable understanding from a person's dictations, uses it while they write and while they ask, and can show its work for every answer.

`RUN.md` is the review method. This file explains what was built, why, what it costs, and where it fails.

> **Part One of the assignment — the product positioning statement and the product vision — is written by the candidate and lives in `docs/`.** This document describes the implementation only.

---

## The two modes, and the boundary between them

The assignment asks what belongs in regular dictation and what belongs in Hey Kivi. The answer is built into the product as two surfaces sharing one memory:

**Dictate** — the person speaks, Kivi writes. Memory has exactly two jobs here: shape *how* the text is written using preferences the person has already stated, and learn quietly from what was said. It never answers, never retrieves, never uses a tool, and never interrupts to ask permission. Someone mid-thought is writing, not talking to an assistant. New understanding appears as a calm note after the fact — "Kivi noticed you prefer detailed written feedback" — with one click to drop it.

**Hey Kivi** — the person asks, Kivi answers, retrieves, polishes, or declines. This is where memory becomes visible and arguable: every answer carries a *Why?* showing the memories used and the original dictations behind them, and a *That's wrong* that archives the memory on the spot.

The rule in one line: **dictation may only change the shape of what you wrote; Hey Kivi may act on what it knows.** That boundary is why the interface has a mode switch rather than a single chat box.

## Architecture

```text
Dictation  ──► semantic extractor  (frames OR local LLM)      ← swappable sensor
               typed memory candidates (one Candidate dataclass)
                        │
                        ▼
               memory controller     (utility · confidence · NLI · decay)   ← fixed policy
                        │
               SQLite: memories · evidence · decisions · embeddings · growth
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  Dictate: style shaping          Hey Kivi: hybrid retrieval
  + quiet learning                → grounded answer or refusal, with citations
```

The load-bearing line is between the sensor and the controller. Sensors may be wrong, replaceable and probabilistic. The controller is none of those: admission, conflict resolution, reinforcement, decay and abstention are arithmetic in `engine.py` and `config.py`, and they do not change when backends change. **An LLM proposes; the controller decides.** The evaluation asserts this — the same 10 controller checks and 5 admission checks pass identically on both paths.

### What the controller computes

| Stage | Function |
|---|---|
| Utility | `0.45·importance + 0.35·confidence + 0.20·future_value(type)` |
| Admission | store iff `utility ≥ 0.55` **and** `confidence ≥ 0.50` |
| Supersede | archive the conflicting memory iff the new one is active and `confidence ≥ 0.72` |
| Reinforcement | noisy-OR `c' = 1 − (1 − c_old)(1 − 0.45·c_new)`, plus `min(0.08, 0.006·recurrence)` utility |
| Decay | `c_eff = c · exp(−(decay_rate / (1 + ln(1 + recurrence))) · age_days)` |
| Retrieval | `0.55·lexical + 1.60·cosine + 0.35·c_eff + 0.45·utility − 0.25·[tentative]`, answer floor `1.10` |

No term is keyed to a query word or a predicate name. Every threshold is a field on `KiviConfig`, so the policy is one file rather than scattered literals.

### What is remembered, and what is not

Five types — `preference`, `fact`, `project_state`, `task`, `event` — with different future value and different decay. A memory is `active`, `tentative`, or `archived`; nothing is ever deleted, so a superseded or forgotten memory keeps its evidence and the decision that removed it.

Conflict handling is structural rather than enumerated. `is_exclusive()` decides from the *shape* of the relation whether a second value replaces the first: `manager_is` and `prefers_feedback` are single-valued, `needs_to_do` and `has_task` are not. This matters because predicates are synthesised from the sentence, so no list of predicate names could cover them.

### Backends

| Layer | No install | With a model |
|---|---|---|
| Extraction | grammatical frames (`frames.py`) | instruct model returning structured candidates + explicit rejections |
| Conflict | structural exclusivity rule | NLI entailment/contradiction — DeBERTa-MNLI cross-encoder if `transformers` is present, else the instruct model as a classifier |
| Retrieval | hashed bag-of-words + lexical | `nomic-embed-text` vectors, hybrid |
| Query routing | nearest intent exemplar | LLM intent classification with slot filling |
| Answering | person-shifted stored claim | grounded generation, citations mandatory, abstains otherwise |

All default to `auto`. Fallback is never silent — `doctor`, `inspect --kind backends` and the engineering view all report it.

## Why extraction is domain-free

The first version of the deterministic extractor matched this corpus by name — `golden goose`, `sarvam`, `predictive maintenance`, `\bi work at (...)`. It scored 11/11 on its own evaluation. Run against a *different* user's 500 dictations it learned **8 memories** and abstained on nearly every question, because "I work **out of** the Pune office" does not match `i work at`.

That is exactly the condition the reviewer's corpus creates, and the evaluation could not see it.

`frames.py` now keys on how English first-person statements are *shaped* — "my X is Y", "I prefer X", "for the X I am using Y", "I need to X" — and synthesises the predicate from the sentence. A designer's dictations produce `manager_is` and `prefers_feedback`; an engineer's produce `manager_is` and `prefers_answers`, from the same code.

| On a corpus it was not written against | Before | After |
|---|---|---|
| Memories learned from 500 records | 8 | **23** |
| Reviewer-style questions answered | 1 / 7 | **6 / 7** |

`python3 -m tests.foreign_corpus` runs this permanently. It fails the build if the system stops generalising.

## Evaluation

`python3 -m kivi_memory --deterministic evaluate --records 500` runs the whole pipeline and writes every case with its provenance to `data/evaluation_results.json`. Three families:

- **6 answer checks** — including one that must abstain. Graded by substring **with an NLI contradiction veto**: an answer must contain the expected terms *and* not be judged to contradict a reference claim, so `"python"` appearing inside "you are not using Python" cannot pass.
- **10 controller checks** — provenance on every memory, rejection actually happening, reinforcement rather than duplication, forgetting that archives and logs, evidence spanning more than one dictation, sublinear growth, and three probes asserting that a weak candidate is rejected, a hedged claim stays tentative, and a weak contradiction does *not* overturn an established memory.
- **5 admission checks** — a small hand-labelled set with fixed confidence/importance values, saying where the admission line should sit.

**Deterministic path: 20/21** (5/6 · 10/10 · 5/5), about 4 seconds end to end.

The one remaining answer miss is real and left visible: a hashed bag-of-words cannot bridge "assignment" to "internship task", so Kivi abstains rather than guessing. That is the correct failure, and it is the case the embedding backend exists to fix.

### Cost, latency, growth

Measured on the 500-record corpus, written into every results file:

| | Deterministic | Local model |
|---|---|---|
| Full ingest | ~2 s | ~230 model calls after caching |
| Query latency | 2–8 ms | dominated by the model |
| External API calls | 0 | 0 (localhost) |
| Cost | $0.00 | $0.00 |

Growth is sampled ten times during ingest into a `growth_samples` table: **23 active memories from 500 dictations — 4.6 per 100, selectivity 0.954**, about 3.4 KB of database per dictation. Memory grows far more slowly than the log, which is the whole claim — Kivi is not keeping a copy of what you said.

Caching keeps the model path affordable: the extraction prompt deliberately omits the timestamp so repeated dictations are content-addressable, taking the corpus from 578 calls to 233.

### What calibration found

`python3 -m kivi_memory calibrate` grid-searches the three admission thresholds across 60 configurations and reports the surface with a per-threshold sensitivity breakdown.

The first version of this reported **FLAT** — every setting scored identically, meaning the evaluation could not falsify the thresholds at all. That was a finding about the evaluation, not a validation of the numbers. Adding the labelled admission set and widening the grid fixed it: scores now range **17–19 across 60 settings**, so `best` is a defensible choice rather than three constants someone liked. The tool still prints its own verdict, and will say FLAT again if the surface ever stops discriminating.

## Limitations

Stated plainly, because several of them are the interesting part.

- **The no-model path abstains on paraphrase.** Two evaluation questions and some foreign-corpus questions fail this way. It is a real ceiling on bag-of-words retrieval, not a bug, and it is why the model path exists.
- **The frame extractor is still a rule system.** It is domain-free now, but it only understands the sentence shapes it has frames for. Unusual phrasing is silently missed rather than mis-learned — a safe failure, not a harmless one.
- **`OllamaNLI` is an instruct model prompted as a classifier.** It is real inference and beats the old whitelist, but a dedicated cross-encoder is better calibrated — `pip install torch transformers` and `--nli cross-encoder`.
- **Retrieval scans all active memories.** Fine at this scale, needs an ANN index at production scale.
- **NLI may only archive memories for `preference`, `fact` and `project_state`.** Tasks and events are multi-valued; a model that judged two to-dos "contradictory" would quietly delete a task list. That constraint is in the controller, not the prompt.
- **The seed corpus is synthetic and its anchors are known**, so its accuracy measures wiring and policy, not real-world extraction quality. `tests/foreign_corpus.py` exists because of this.
- **The interface has no audio.** Dictation is typed. The assignment does not require speech recognition, and nothing downstream depends on where the text came from.

## AI use

The corpus is synthetic and generated by code (`seed.py`, `tests/foreign_corpus.py`), with a fixed seed. Implementation, refactoring and documentation were done with AI assistance. The product positioning statement and product vision in `docs/` are the candidate's own work, as the assignment requires.

## Layout

```text
kivi_memory/
  frames.py       domain-free extraction frames
  extractors.py   rule and LLM sensors, both emitting one Candidate type
  engine.py       the memory controller: ingest, dictate, ask, forget, growth
  core.py         Candidate, utility, decay, exclusivity, tokenisation
  config.py       every threshold and backend switch in one place
  nli.py          entailment / contradiction, cross-encoder or prompted
  embeddings.py   Ollama vectors with a deterministic offline fallback
  intent.py       query routing: LLM classifier or nearest exemplar
  answerer.py     grounded generation and dictation polishing
  llm.py          local Ollama and OpenAI-compatible transports, with caching
  db.py           schema and migrations
  evaluation.py   the reproducible evaluation and threshold calibration
  server.py       the local HTTP API
web/              the interface
tests/            fake_ollama.py, run_llm_path.py (21 checks), foreign_corpus.py
docs/             product positioning and vision (candidate's own writing)
```
