from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import KiviConfig
from .core import utility_score
from .db import reset_database
from .engine import KiviMemoryEngine
from .seed import generate_corpus, write_jsonl


DEFAULT_QUESTIONS = [
    {
        "id": "response_style",
        "query": "What response style do I prefer?",
        "expected_contains": ["concise"],
        "expected_claim": "The user prefers concise communication.",
        "expect_abstain": False,
    },
    {
        "id": "project_language",
        "query": "Which language am I using for the Golden Goose prototype?",
        "expected_contains": ["python"],
        "expected_claim": "The user is using Python for the Golden Goose prototype.",
        "expect_abstain": False,
    },
    {
        "id": "find_slack",
        "query": "Hey Kivi, find the dictation I did around 5 PM yesterday in Slack and polish it for the meeting.",
        "expected_contains": ["predictive maintenance", "sensor drift"],
        "expected_claim": "The retrieved dictation is about predictive maintenance and sensor drift.",
        "expect_abstain": False,
    },
    {
        "id": "assignment",
        "query": "What assignment am I working on?",
        "expected_contains": ["sarvam", "golden goose"],
        "expected_claim": "The user is working on the Sarvam AI Golden Goose task.",
        "expect_abstain": False,
    },
    {
        "id": "abstain_unknown",
        "query": "What is my passport number?",
        "expected_contains": [],
        "expect_abstain": True,
    },
    {
        "id": "ignored",
        "query": "What did Kivi deliberately ignore?",
        "expected_contains": ["low durable utility"],
        "expected_claim": "The assistant explains which candidates it chose not to remember.",
        "expect_abstain": False,
    },
]


# Controller-level expectations. These check the memory policy rather than the
# wording of an answer, so they hold no matter which sensor produced the
# candidates - which is the point of keeping the controller fixed.
CONTROLLER_CHECKS = [
    {
        "id": "records_provenance",
        "description": "Every stored memory has at least one supporting transcript.",
        "check": lambda engine, stats: all(memory["evidence"] for memory in engine.memories(50)),
    },
    {
        "id": "rejects_something",
        "description": "The controller rejected or no-op'd at least one candidate rather than storing everything.",
        "check": lambda engine, stats: (
            stats["decisions_by_action"].get("REJECT", 0) + stats["decisions_by_action"].get("NO_OP", 0)
        ) > 0,
    },
    {
        "id": "reinforces_recurrence",
        "description": "Repeated evidence reinforced an existing memory instead of duplicating it.",
        "check": lambda engine, stats: stats["decisions_by_action"].get("REINFORCE", 0) > 0,
    },
    {
        "id": "no_orphan_memories",
        "description": "Active memory count is below the number of transcripts (selective, not a transcript log).",
        "check": lambda engine, stats: stats["counts"]["memories"] < stats["counts"]["transcripts"],
    },
    {
        "id": "forget_archives",
        "description": "An explicit user correction archives the memory and logs a FORGET decision.",
        "check": lambda engine, stats: _forget_works(engine),
    },
    {
        "id": "multi_dictation_recovery",
        "description": "At least one answer is supported by evidence drawn from more than one dictation.",
        "check": lambda engine, stats: _spans_multiple_dictations(engine),
    },
    {
        "id": "rejects_below_threshold",
        "description": "A candidate under the admission thresholds is REJECTED, not stored.",
        "check": lambda engine, stats: _rejects_weak_candidate(engine),
    },
    {
        "id": "hedged_stays_tentative",
        "description": "A hedged claim is stored tentative or rejected - never as settled fact.",
        "check": lambda engine, stats: _hedged_is_not_active(engine),
    },
    {
        "id": "weak_conflict_does_not_supersede",
        "description": "A low-confidence contradicting claim does NOT archive an established memory.",
        "check": lambda engine, stats: _weak_conflict_held(engine),
    },
    {
        "id": "growth_is_sublinear",
        "description": "Memory count grows far more slowly than the transcript log (selective, not a copy).",
        "check": lambda engine, stats: _growth_sublinear(engine),
    },
]


# ---------------------------------------------------------------------------
# Controller probes.
#
# These construct candidates directly and assert what the controller does with
# them. They exist because the question-answering cases could not falsify the
# admission thresholds at all: `calibrate` found every setting in a 45-point
# grid scored identically. A policy nothing can fail is not a policy.
# ---------------------------------------------------------------------------

# A small hand-labelled admission set with FIXED confidence/importance values.
#
# The relative probes above check that the admission gate is applied at all;
# they pass at any threshold, which is why `calibrate` reported a flat surface.
# These say where the line should sit: each row is a judgement that a candidate
# of this strength deserves - or does not deserve - a place in memory. A
# threshold that admits the throwaway aside, or rejects the plainly stated
# preference, now scores worse. That is what makes the numbers falsifiable.
ADMISSION_LABELS = [
    {
        "id": "plain_preference",
        "why": "A preference stated plainly and directly. Must be kept.",
        "memory_type": "preference", "confidence": 0.86, "importance": 0.88,
        "expect": "store",
    },
    {
        "id": "stated_fact",
        "why": "A profile fact stated once, unambiguously. Must be kept.",
        "memory_type": "fact", "confidence": 0.84, "importance": 0.82,
        "expect": "store",
    },
    {
        "id": "borderline_preference",
        "why": "A real preference, stated once, less emphatically. Should still be kept.",
        "memory_type": "preference", "confidence": 0.66, "importance": 0.62,
        "expect": "store",
    },
    {
        "id": "passing_aside",
        "why": "An offhand remark with little future bearing. Must not be kept.",
        "memory_type": "event", "confidence": 0.44, "importance": 0.30,
        "expect": "reject",
    },
    {
        "id": "vague_mention",
        "why": "A vague, low-confidence mention. Must not be kept.",
        "memory_type": "event", "confidence": 0.35, "importance": 0.40,
        "expect": "reject",
    },
]


def score_admission_labels(engine: KiviMemoryEngine) -> list[dict[str, Any]]:
    results = []
    for label in ADMISSION_LABELS:
        probe = _probe_engine(engine, f"label_{label['id']}")
        candidate = _probe_candidate(
            memory_type=label["memory_type"],
            predicate=f"prefers_{label['id']}" if label["memory_type"] == "preference" else f"{label['id']}_is",
            object=label["id"].replace("_", " "),
            canonical_text=f"User: {label['id'].replace('_', ' ')}.",
            confidence=label["confidence"],
            importance=label["importance"],
            utility=utility_score(label["importance"], label["confidence"], label["memory_type"]),
        )
        decision = _apply_probe(probe, [candidate])[0]
        stored = decision["action"] in ("ADD", "ADD_TENTATIVE", "UPDATE", "REINFORCE")
        ok = stored if label["expect"] == "store" else not stored
        results.append({
            "id": label["id"], "why": label["why"], "expect": label["expect"],
            "action": decision["action"], "utility": decision["utility"],
            "confidence": decision["confidence"], "passed": ok,
        })
    return results


def _probe_engine(engine: KiviMemoryEngine, suffix: str):
    """A scratch database sharing this engine's config, for controller probes."""
    import tempfile  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415
    tmp = _Path(tempfile.mkdtemp(prefix="kivi_probe_")) / f"{suffix}.db"
    return KiviMemoryEngine(str(tmp), engine.config)


def _probe_candidate(**overrides):
    from .core import Candidate  # noqa: PLC0415
    base = dict(
        memory_type="preference", subject="user", predicate="prefers_probe",
        object="a probe value", scope="probe", canonical_text="User prefers a probe value.",
        evidence="probe", importance=0.9, confidence=0.9, utility=0.9,
        decay_rate=0.01, status="active", reason="probe", source="probe",
    )
    base.update(overrides)
    return Candidate(**base)


def _probe_record(idx: int = 1, text: str = "probe", created: str = "2026-08-01T10:00:00"):
    return {"id": f"probe_{idx}", "created_at": created, "app": "probe",
            "raw_asr": text, "formatted_text": text, "metadata": {}}


def _apply_probe(engine: KiviMemoryEngine, candidates: list) -> list[dict[str, Any]]:
    from .db import connect, migrate  # noqa: PLC0415
    decisions = []
    with connect(engine.db_path) as conn:
        migrate(conn)
        for idx, candidate in enumerate(candidates, start=1):
            record = _probe_record(idx, created=f"2026-08-{idx:02d}T10:00:00")
            engine._upsert_transcript(conn, record)
            decisions.append(engine._apply_candidate(conn, record, candidate))
        conn.commit()
    return decisions


def _rejects_weak_candidate(engine: KiviMemoryEngine) -> bool:
    probe = _probe_engine(engine, "reject")
    cfg = probe.config
    # Deliberately just under both admission gates.
    weak = _probe_candidate(
        confidence=max(0.0, cfg.admit_confidence - 0.15),
        importance=0.1,
        utility=max(0.0, cfg.admit_utility - 0.15),
        object="a weak probe value",
    )
    decision = _apply_probe(probe, [weak])[0]
    return decision["action"] == "REJECT"


def _hedged_is_not_active(engine: KiviMemoryEngine) -> bool:
    probe = _probe_engine(engine, "hedge")
    hedged = _probe_candidate(
        status="tentative",
        confidence=probe.config.tentative_confidence - 0.1,
        object="a hedged probe value",
    )
    decision = _apply_probe(probe, [hedged])[0]
    return decision["action"] in ("ADD_TENTATIVE", "REJECT")


def _weak_conflict_held(engine: KiviMemoryEngine) -> bool:
    """An established memory must not be overturned by a weak contradiction."""
    probe = _probe_engine(engine, "conflict")
    strong = _probe_candidate(predicate="probe_city_is", object="alpha", scope="probe_profile",
                              canonical_text="User's probe city is Alpha.", confidence=0.95)
    weak = _probe_candidate(predicate="probe_city_is", object="beta", scope="probe_profile",
                            canonical_text="User's probe city is Beta.",
                            confidence=max(0.0, probe.config.supersede_confidence - 0.2))
    decisions = _apply_probe(probe, [strong, weak])
    from .db import connect  # noqa: PLC0415
    with connect(probe.db_path) as conn:
        statuses = {row["object"]: row["status"] for row in conn.execute(
            "SELECT object, status FROM memories WHERE predicate='probe_city_is'")}
    return decisions[0]["action"] == "ADD" and statuses.get("alpha") == "active"


def _spans_multiple_dictations(engine: KiviMemoryEngine) -> bool:
    """Recovering understanding distributed across several dictations.

    A memory whose confidence was built from separate dictations, and whose
    provenance still names each of them, is the concrete form of that claim.
    """
    for memory in engine.memories(60):
        transcripts = {item.get("transcript_id") for item in memory.get("evidence", [])}
        if len(transcripts) >= 2:
            return True
    return False


def _growth_sublinear(engine: KiviMemoryEngine) -> bool:
    growth = engine.growth()
    latest = growth.get("latest")
    if not latest or latest["transcripts"] < 20:
        return False
    return latest["active_memories"] < latest["transcripts"] * 0.25


def grade_answer(engine: KiviMemoryEngine, question: dict[str, Any], answer: dict[str, Any]) -> dict[str, Any]:
    """Grade an answer by entailment when an NLI backend is available.

    Substring matching is not a correctness test: `"python" in text` passes on
    "you are NOT using Python", and it fails on a correct answer that happens
    to word things differently - which is exactly what happens once answers are
    generated rather than templated. The project already carries an NLI model
    for conflict detection, so the evaluation uses it to ask the right
    question: does the answer entail the reference claim?

    So NLI is used as a *veto*, not as the pass condition: an answer must still
    contain the expected terms (reproducible with no model installed), and an
    answer that a model judges to CONTRADICT the reference claim fails even if
    the substring is present. That is strictly stronger than substring alone,
    it needs no model to run, and it uses inference where inference is most
    reliable. The raw entailment label is reported either way.
    """
    text = answer["answer"]
    lowered = text.lower()
    substring_ok = all(value.lower() in lowered for value in question.get("expected_contains", []))
    abstain_ok = bool(answer["abstained"]) == bool(question["expect_abstain"])

    claim = question.get("expected_claim")
    entailment: dict[str, Any] | None = None
    contradicted = False
    if claim and not question["expect_abstain"] and engine.nli.name != "off":
        result = engine.nli.classify(text, claim)
        entailment = result.as_dict()
        contradicted = result.label == "contradiction" and result.probability >= 0.5

    return {
        "substring_ok": substring_ok,
        "abstain_ok": abstain_ok,
        "contradicted_reference": contradicted,
        "entailment": entailment,
        "grader": "substring + nli-contradiction-veto" if entailment else "substring",
        "passed": abstain_ok and substring_ok and not contradicted,
    }


def _forget_works(engine: KiviMemoryEngine) -> bool:
    before = engine.stats()["memories_by_status"].get("archived", 0)
    result = engine.ask("Forget that I prefer bulleted summaries.")
    after = engine.stats()
    if result.get("abstained"):
        return False
    return (
        after["memories_by_status"].get("archived", 0) > before
        and after["decisions_by_action"].get("FORGET", 0) > 0
    )


def _dictation_example(engine: KiviMemoryEngine) -> dict[str, Any]:
    """One round-trip through the OTHER mode, so the evaluation covers both.

    Regular dictation and Hey Kivi are different products with the same memory
    behind them; an evaluation that only exercises the question path does not
    show what memory does during ordinary use.
    """
    try:
        return engine.dictate(
            "um so I need to uh send the evaluation writeup before friday and you know "
            "keep the summary short when you write it up for me",
            app="slack",
        )
    except Exception as exc:  # never let the demo break the run
        return {"error": repr(exc)}


def run_evaluation(
    db_path: str = "data/kivi.db",
    corpus_path: str = "data/seed_corpus.jsonl",
    out_path: str = "data/evaluation_results.json",
    records: int = 500,
    fresh: bool = True,
    config: KiviConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if fresh:
        reset_database(db_path)
    corpus = generate_corpus(records)
    write_jsonl(corpus, corpus_path)
    engine = KiviMemoryEngine(db_path, config)
    ingest_result = engine.ingest_records(corpus)

    cases = []
    passed = 0
    latencies = []
    for question in DEFAULT_QUESTIONS:
        answer = engine.ask(question["query"])
        grade = grade_answer(engine, question, answer)
        if grade["passed"]:
            passed += 1
        latencies.append(answer["latency_ms"])
        cases.append({
            "id": question["id"],
            "query": question["query"],
            "answer": answer,
            "expected_contains": question["expected_contains"],
            "expected_claim": question.get("expected_claim"),
            "expect_abstain": question["expect_abstain"],
            "grade": grade,
            "passed": grade["passed"],
        })

    stats = engine.stats()
    controller_cases = []
    controller_passed = 0
    for item in CONTROLLER_CHECKS:
        try:
            ok = bool(item["check"](engine, stats))
        except Exception as exc:  # a failing invariant must not kill the run
            ok = False
            controller_cases.append({"id": item["id"], "description": item["description"],
                                     "passed": False, "error": repr(exc)})
            continue
        controller_passed += int(ok)
        controller_cases.append({"id": item["id"], "description": item["description"], "passed": ok})

    admission_cases = score_admission_labels(engine)
    admission_passed = sum(1 for case in admission_cases if case["passed"])

    stats_after = engine.stats()
    result = {
        "summary": {
            "backends": engine.backends(),
            "answer_checks": {
                "passed": passed,
                "total": len(DEFAULT_QUESTIONS),
                "accuracy": round(passed / len(DEFAULT_QUESTIONS), 3),
            },
            "controller_checks": {
                "passed": controller_passed,
                "total": len(CONTROLLER_CHECKS),
                "accuracy": round(controller_passed / len(CONTROLLER_CHECKS), 3),
            },
            "admission_checks": {
                "passed": admission_passed,
                "total": len(ADMISSION_LABELS),
                "accuracy": round(admission_passed / len(ADMISSION_LABELS), 3),
            },
            "passed": passed + controller_passed + admission_passed,
            "total": len(DEFAULT_QUESTIONS) + len(CONTROLLER_CHECKS) + len(ADMISSION_LABELS),
            "accuracy": round(
                (passed + controller_passed + admission_passed)
                / (len(DEFAULT_QUESTIONS) + len(CONTROLLER_CHECKS) + len(ADMISSION_LABELS)), 3
            ),
            "avg_query_latency_ms": round(sum(latencies) / len(latencies), 2),
            "end_to_end_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "model_usage": engine.usage(),
            "database": stats_after,
            "growth": engine.growth(),
            "dictation_example": _dictation_example(engine),
            "ingest": ingest_result,
        },
        "cases": cases,
        "controller_cases": controller_cases,
        "admission_cases": admission_cases,
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def compare_backends(
    db_path: str = "data/kivi.db",
    corpus_path: str = "data/seed_corpus.jsonl",
    out_path: str = "data/evaluation_results.json",
    records: int = 500,
) -> dict[str, Any]:
    """Run the deterministic and local-model configurations on the same corpus.

    Same seed, same corpus, same controller and same questions - only the
    sensor layer changes, so the difference is attributable.
    """
    runs = {}
    configs = {
        "deterministic": KiviConfig.from_env().apply_overrides(
            extractor="rule", nli="off", retrieval="lexical", answerer="template"
        ),
        "local_llm": KiviConfig.from_env().apply_overrides(
            extractor="llm", nli="auto", retrieval="hybrid", answerer="llm"
        ),
    }
    for name, config in configs.items():
        suffix = db_path.replace(".db", f".{name}.db")
        runs[name] = run_evaluation(
            db_path=suffix,
            corpus_path=corpus_path,
            out_path=out_path.replace(".json", f".{name}.json"),
            records=records,
            fresh=True,
            config=config,
        )

    comparison = {
        name: {
            "backends": run["summary"]["backends"]["config"],
            "extractor": run["summary"]["backends"]["extractor"],
            "answer_accuracy": run["summary"]["answer_checks"]["accuracy"],
            "controller_accuracy": run["summary"]["controller_checks"]["accuracy"],
            "memories": run["summary"]["database"]["counts"]["memories"],
            "decisions_by_action": run["summary"]["database"]["decisions_by_action"],
            "avg_query_latency_ms": run["summary"]["avg_query_latency_ms"],
            "model_usage": run["summary"]["model_usage"],
        }
        for name, run in runs.items()
    }
    payload = {"comparison": comparison, "runs": runs}
    output = Path(out_path.replace(".json", ".comparison.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

# Wide enough to cross the boundary. A grid narrower than the margin in the
# labelled set produces a flat surface and tells you nothing, which is what the
# first version of this did.
CALIBRATION_GRID = {
    "admit_utility": [0.35, 0.45, 0.55, 0.65, 0.75],
    "admit_confidence": [0.30, 0.45, 0.60, 0.75],
    "supersede_confidence": [0.62, 0.72, 0.82],
}


def calibrate(
    db_path: str = "data/kivi.db",
    corpus_path: str = "data/seed_corpus.jsonl",
    out_path: str = "data/calibration.json",
    records: int = 300,
    config: KiviConfig | None = None,
) -> dict[str, Any]:
    """Search the controller's admission thresholds instead of asserting them.

    `admit_utility=0.55`, `admit_confidence=0.50` and
    `supersede_confidence=0.72` were hand-picked numbers. This runs the full
    ingest + evaluation across a grid and reports the surface, so the shipped
    values are a defensible choice with a sensitivity analysis behind them
    rather than three constants someone liked.

    Score prefers, in order: checks passed, then selectivity (fewer memories
    for the same score is a better filter, which is the whole product claim).
    """
    import itertools
    import tempfile

    base = config or KiviConfig.from_env()
    results = []
    keys = list(CALIBRATION_GRID)

    with tempfile.TemporaryDirectory(prefix="kivi_calib_") as tmp:
        for combo in itertools.product(*(CALIBRATION_GRID[key] for key in keys)):
            settings = dict(zip(keys, combo))
            trial = KiviConfig(**{**base.__dict__, **settings})
            run = run_evaluation(
                db_path=f"{tmp}/trial.db",
                corpus_path=corpus_path,
                out_path=f"{tmp}/trial.json",
                records=records,
                fresh=True,
                config=trial,
            )
            summary = run["summary"]
            memories = summary["database"]["counts"]["memories"]
            transcripts = max(1, summary["database"]["counts"]["transcripts"])
            results.append({
                "settings": settings,
                "passed": summary["passed"],
                "total": summary["total"],
                "answer_checks": summary["answer_checks"]["passed"],
                "controller_checks": summary["controller_checks"]["passed"],
                "memories": memories,
                "selectivity": round(1 - memories / transcripts, 4),
                "failing": [case["id"] for case in run["cases"] if not case["passed"]]
                           + [case["id"] for case in run["controller_cases"] if not case["passed"]],
            })

    results.sort(key=lambda item: (item["passed"], item["selectivity"]), reverse=True)
    best = results[0]

    # Sensitivity: how much does each threshold matter on its own?
    sensitivity = {}
    for index, key in enumerate(keys):
        by_value = {}
        for item in results:
            by_value.setdefault(item["settings"][key], []).append(item["passed"])
        sensitivity[key] = {
            str(value): round(sum(scores) / len(scores), 3) for value, scores in sorted(by_value.items())
        }

    scores = {item["passed"] for item in results}
    memory_counts = [item["memories"] for item in results]
    if len(scores) == 1:
        verdict = (
            "FLAT: every setting in the grid scored identically "
            f"({results[0]['passed']}/{results[0]['total']}), so this evaluation does not "
            "discriminate between them and the shipped thresholds are currently unfalsifiable. "
            f"Admitted-memory count did vary ({min(memory_counts)}-{max(memory_counts)}), so the "
            "thresholds are doing something - the questions just never ask about it. Before "
            "trusting a calibrated value, add cases that SHOULD be rejected and assert they were: "
            "a borderline candidate the controller must not store, a hedged claim that must stay "
            "tentative, a weak conflict that must not supersede."
        )
    else:
        verdict = (
            f"Scores ranged {min(scores)}-{max(scores)} across {len(results)} settings; the "
            "surface discriminates and `best` is a defensible choice."
        )

    payload = {
        "verdict": verdict,
        "best": best,
        "shipped_defaults": {key: getattr(KiviConfig(), key) for key in keys},
        "score_range": sorted(scores),
        "memory_count_range": [min(memory_counts), max(memory_counts)],
        "mean_score_by_threshold_value": sensitivity,
        "grid_size": len(results),
        "records_per_trial": records,
        "all_results": results,
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
