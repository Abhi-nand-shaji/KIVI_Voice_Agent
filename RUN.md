# RUN.md

## Primary review method

**A completely local application.** Python 3.12 + SQLite, a local web interface at `http://127.0.0.1:8000`. No deployment, no hosted service, no account.

There are two ways to run it. Both use the same interface, the same controller, the same database and the same audit trail — only the sensing layer differs.

| | Install cost | What it uses |
|---|---|---|
| **A. Deterministic** (default if nothing is installed) | nothing beyond Python | grammatical extraction frames, lexical retrieval, template answers |
| **B. Local model** (recommended) | `ollama` + two models, ~5 GB | LLM extraction, NLI conflict detection, embeddings, grounded answers |

**Path A is guaranteed to work with zero setup.** Path B answers more questions. A third option — an OpenAI-compatible API key instead of a local download — is in §5 if that is easier than pulling a model.

---

## 1. Required runtimes and versions

- **Python 3.12+** (3.11 works). Verify: `python3 --version`. On Windows use `py -3` in place of `python3`.
- **SQLite** — bundled with Python's standard library; nothing to install.
- Optional, for path B: **Ollama** ≥ 0.1.30 — https://ollama.com/download
- Optional, for a dedicated NLI cross-encoder: `pip install torch transformers`

There are **no required third-party Python packages.** `pip install` is not needed for path A.

## 2. Required environment variables

**None.** Every setting has a CLI flag and a default. `.env.example` lists the optional overrides.

The only variable that changes behaviour meaningfully is `KIVI_LLM_API_KEY`, and only if you choose the hosted-model option in §5. It is never required and no credential is committed to this repository.

## 3. Exact commands to install dependencies

```bash
# Path A — nothing to install.
python3 --version                      # confirm 3.11+

# Path B — local models (optional)
ollama serve &                         # leave running
ollama pull qwen2.5:7b-instruct        # extraction, NLI, routing, answers (~4.7 GB)
ollama pull nomic-embed-text           # embeddings (~275 MB)
```

## 4. Exact commands to create, migrate and seed the database

```bash
cd <repo root>
python3 -m kivi_memory reset                                  # create + migrate a fresh SQLite database
python3 -m kivi_memory seed --records 500 --out data/seed_corpus.jsonl
```

`reset` creates `data/kivi.db` and applies the full schema plus migrations. `seed` generates the reproducible 500-record corpus (fixed seed) and ingests it.

For path A explicitly, add `--deterministic`:

```bash
python3 -m kivi_memory reset
python3 -m kivi_memory --deterministic seed --records 500 --out data/seed_corpus.jsonl
```

**Ingest time.** Path A: about 2 seconds. Path B: roughly one model call per unique dictation — about 230 calls on this corpus after caching — which is a few minutes on a GPU and considerably longer on CPU. To try path B quickly, seed 60 records instead of 500. Responses are cached in `data/llm_cache.db`, so a repeat run is near-instant.

## 5. Exact commands to start every required process

There is **one** process.

```bash
python3 -m kivi_memory serve --port 8000
```

It prints which backend each layer resolved to. For path A explicitly:

```bash
python3 -m kivi_memory --deterministic serve --port 8000
```

Before starting, `doctor` reports exactly what is wired up and why:

```bash
python3 -m kivi_memory doctor
```

<details>
<summary><b>Optional: hosted model instead of a local download</b></summary>

If pulling 5 GB is inconvenient, any OpenAI-compatible endpoint works. Set the key and everything else follows:

```bash
export KIVI_LLM_API_KEY=sk-...            # the variable name, precisely
export KIVI_LLM_BASE_URL=https://api.openai.com/v1   # optional, this is the default
python3 -m kivi_memory doctor
```

Defaults to `gpt-4o-mini` and `text-embedding-3-small`; override with `KIVI_LLM_MODEL` / `KIVI_EMBED_MODEL`. Same prompts, same controller, same audit trail — only the transport changes. See `.env.example`.
</details>

## 6. The URL to open

**http://127.0.0.1:8000**

The interface opens on **Dictate**. Three surfaces, switched at the top:

- **Dictate** — ordinary use. Type what you would have said, press *Write it*, get written text back. Memory's only job here is to shape how it is written, and to learn quietly.
- **Hey Kivi** — ask questions. Answers carry a *Why?* that shows the memories used and the original dictations behind them.
- **What Kivi knows** — everything learned, in plain sentences, each with *Why do you know this?* and *Forget this*.

An **Engineering view** link in the footer reveals the controller decision log, confidence and utility numbers, NLI labels and backend detail. Ordinary use never requires it.

## 7. Primary interactions to try

**In Dictate**, paste this and press *Write it*:

```text
um so I need to send the retention writeup to Neha before friday and you know
I prefer detailed written feedback on my drafts
```

Expect: cleaned-up text; a quiet line saying it was written using what Kivi knows; and a calm note that it noticed a new preference, with an option to drop it. Nothing blocks or asks permission.

**In Hey Kivi**, try these:

```text
What response style do I prefer?
Which language am I using for the Golden Goose prototype?
Hey Kivi, find the dictation I did around 5 PM yesterday in Slack and polish it for the meeting.
What did Kivi deliberately ignore?
What is my passport number?
Forget that I prefer bulleted summaries.
```

`What is my passport number?` must decline — that is the point of it. Press *Why?* on any answer to see provenance; press *That's wrong* to correct it.

These four are paraphrases that appear nowhere in the source, included to show that query understanding is a classifier rather than a chain of substring tests:

```text
Show me what you threw away
Which things did you decide not to keep?
Tell me everything you have learned about me
Pull up the note I recorded on Monday evening in Notion
```

## 8. Exact command to run the candidate evaluation

```bash
python3 -m kivi_memory --deterministic evaluate --records 500 --out data/evaluation_results.json
```

Runs the complete pipeline — generate corpus, ingest, ask, grade — and writes full results. About 4 seconds.

Expected on the deterministic path: **20/21** (5/6 answer checks, 10/10 controller checks, 5/5 admission checks). The one answer miss is real and documented in the README: a bag-of-words retriever cannot bridge "response style" to "concise technical explanations". The system abstains rather than guessing. Drop `--deterministic` to run with the model backends.

Related commands:

```bash
python3 -m kivi_memory evaluate --compare --records 500   # deterministic vs local model, same corpus
python3 -m kivi_memory --deterministic calibrate --records 200   # grid-search the controller thresholds
python3 -m tests.run_llm_path                             # 21 checks, exercises the model paths with no model installed
python3 -m tests.foreign_corpus                           # ingest a DIFFERENT user's 500 records and answer questions about them
```

`tests/foreign_corpus.py` is the one to run if you have limited time. It is the check that this system was not written against its own test data.

## 9. Exact procedure for importing another corpus

```bash
python3 -m kivi_memory reset
python3 -m kivi_memory import path/to/your_corpus.jsonl
python3 -m kivi_memory serve --port 8000
```

Add `--deterministic` to import without any model.

**Format.** JSONL (one object per line) is documented; a JSON array or `{"records": [...]}` is also accepted. One record:

```json
{"id":"record_001","created_at":"2026-08-31T17:07:00","app":"slack","raw_asr":"raw asr text","formatted_text":"Formatted text.","metadata":{"anything":"here"}}
```

**Field names are flexible**, because your log's column names are not published. Each field is matched against these aliases, first match wins:

| Field | Accepted keys |
|---|---|
| formatted output | `formatted_text`, `formatted`, `llm_formatted`, `formatted_output`, `text`, `content`, `transcript`, `final_text`, `output` |
| raw ASR | `raw_asr`, `asr`, `raw`, `raw_text`, `asr_text`, `asr_output`, `transcription` |
| timestamp | `created_at`, `timestamp`, `time`, `date`, `recorded_at`, `started_at`, `ts` |
| application | `app`, `application`, `app_name`, `source`, `surface`, `client`, `target_app` |
| id | `id`, `record_id`, `uuid`, `_id`, `dictation_id` |

Anything else on the record is preserved under `metadata.source_fields` rather than dropped. Only a timestamp and some text are genuinely required; a missing id is derived by content hash.

## 10. Where evaluation results and memory state can be inspected

**In the interface:** *What Kivi knows* for memory state, and the *Engineering view* in the footer for the decision log and backends.

**On the command line:**

```bash
python3 -m kivi_memory inspect --kind stats          # counts, statuses, actions, backends
python3 -m kivi_memory inspect --kind memories --limit 25   # each with evidence and provenance
python3 -m kivi_memory inspect --kind decisions --limit 25  # every ADD / UPDATE / REINFORCE / REJECT / NO_OP / FORGET with its reason
python3 -m kivi_memory inspect --kind transcripts --limit 25
python3 -m kivi_memory inspect --kind backends
```

Decisions carry `extractor`, `nli_label` and `nli_probability`, so you can see which sensor proposed a memory and what inference said about it.

**Files:**

| Path | Contents |
|---|---|
| `data/kivi.db` | transcripts, memories, evidence, decisions, embeddings, growth samples |
| `data/evaluation_results.json` | every case, its answer, provenance, grade, plus latency / growth / model usage / cost |
| `data/calibration.json` | threshold grid-search surface and sensitivity |
| `data/seed_corpus.jsonl` | the reproducible 500-record corpus |
| `data/llm_cache.db` | cached model responses (path B only) |

Directly, if you prefer SQL:

```bash
sqlite3 data/kivi.db "SELECT action, count(*) FROM decisions GROUP BY action;"
sqlite3 data/kivi.db "SELECT canonical_text, status, confidence, recurrence FROM memories ORDER BY utility DESC LIMIT 20;"
sqlite3 data/kivi.db "SELECT transcripts, active_memories, db_bytes FROM growth_samples;"
```

## 11. Exact procedure for resetting the system

```bash
python3 -m kivi_memory reset       # drop and recreate the database
rm -f data/llm_cache.db            # also discard cached model responses (optional)
```

`reset` is destructive and immediate; there is no confirmation prompt and no other state to clear.

---

## Choosing backends individually

```bash
python3 -m kivi_memory --extractor llm --nli llm --retrieval hybrid --answerer llm serve
python3 -m kivi_memory --extractor hybrid ask "..."     # union of frames and model, frames win ties
python3 -m kivi_memory --nli cross-encoder ask "..."    # needs torch + transformers
```

| Flag | Values | Default |
|---|---|---|
| `--extractor` | `rule`, `llm`, `hybrid`, `auto` | `auto` |
| `--nli` | `off`, `llm`, `cross-encoder`, `auto` | `auto` |
| `--retrieval` | `lexical`, `hybrid`, `auto` | `auto` |
| `--answerer` | `template`, `llm`, `auto` | `auto` |
| `--deterministic` | forces all four to the no-install path | — |

`auto` uses a model when one is reachable and falls back when it is not. The fallback is never silent: it is reported by `doctor`, by `inspect --kind backends`, and by a banner in the interface's engineering view.
