"""The reproducible evaluation harness.

Generates a fixed-seed corpus, ingests it, asks every question, runs every
controller and admission check, and writes one JSON file with the answers, the
provenance behind each, the grades, and the cost/latency/growth numbers.

`compare_backends` runs the identical corpus and identical checks through two
backend configurations, which is the only way to say anything defensible about
whether the model earns its place.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import KiviConfig
from .db import reset_database
from .engine import KiviMemoryEngine
from .eval_cases import (
    ADMISSION_LABELS,
    CONTROLLER_CHECKS,
    DEFAULT_QUESTIONS,
    _dictation_example,
    grade_answer,
    score_admission_labels,
)
from .seed import generate_corpus, write_jsonl


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
