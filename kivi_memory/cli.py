from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ANSWERER_MODES, EXTRACTOR_MODES, KiviConfig, NLI_MODES, RETRIEVAL_MODES
from .db import connect, migrate, reset_database
from .engine import KiviMemoryEngine
from .evaluation import calibrate, compare_backends, run_evaluation
from .llm import get_client
from .seed import generate_corpus, read_jsonl, write_jsonl
from .server import serve


def add_backend_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "backends",
        "Local model wiring. Defaults are 'auto': use the local Ollama daemon when it is "
        "reachable and the model is pulled, otherwise fall back to the deterministic path. "
        "No API key is used anywhere.",
    )
    group.add_argument("--extractor", choices=EXTRACTOR_MODES, default=None)
    group.add_argument("--nli", choices=NLI_MODES, default=None)
    group.add_argument("--retrieval", choices=RETRIEVAL_MODES, default=None)
    group.add_argument("--answerer", choices=ANSWERER_MODES, default=None)
    group.add_argument("--ollama-host", dest="ollama_host", default=None)
    group.add_argument("--llm-model", dest="llm_model", default=None)
    group.add_argument("--embed-model", dest="embed_model", default=None)
    group.add_argument("--nli-model", dest="nli_model", default=None)
    group.add_argument("--no-cache", dest="use_cache", action="store_false", default=None,
                       help="Bypass the local response cache (slower, still offline).")
    group.add_argument("--deterministic", action="store_true",
                       help="Shorthand for --extractor rule --nli off --retrieval lexical --answerer template.")


def build_config(args: argparse.Namespace) -> KiviConfig:
    config = KiviConfig.from_env()
    if getattr(args, "deterministic", False):
        config.apply_overrides(extractor="rule", nli="off", retrieval="lexical", answerer="template")
    config.apply_overrides(
        extractor=getattr(args, "extractor", None),
        nli=getattr(args, "nli", None),
        retrieval=getattr(args, "retrieval", None),
        answerer=getattr(args, "answerer", None),
        ollama_host=getattr(args, "ollama_host", None),
        llm_model=getattr(args, "llm_model", None),
        embed_model=getattr(args, "embed_model", None),
        nli_model=getattr(args, "nli_model", None),
        use_cache=getattr(args, "use_cache", None),
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Kivi semantic memory prototype")
    parser.add_argument("--db", default="data/kivi.db", help="SQLite database path")
    add_backend_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("migrate", help="Create or update the SQLite schema")
    sub.add_parser("reset", help="Delete and recreate the SQLite database")
    sub.add_parser("doctor", help="Check the local Ollama daemon, models, and resolved backends")
    sub.add_parser("metrics", help="Print every recorded measurement: latency, model usage, cost, growth")

    seed = sub.add_parser("seed", help="Generate and ingest the synthetic 500-record corpus")
    seed.add_argument("--records", type=int, default=500)
    seed.add_argument("--out", default="data/seed_corpus.jsonl")
    seed.add_argument("--report", default="data/ingest_report.json",
                      help="Where to write latency / model usage / growth for this ingest.")

    ingest = sub.add_parser("import", help="Import a JSONL corpus")
    ingest.add_argument("path", help="JSONL file with transcript records")
    ingest.add_argument("--report", default="data/ingest_report.json",
                        help="Where to write latency / model usage / growth for this ingest.")

    ask = sub.add_parser("ask", help="Ask Hey Kivi against the learned memory")
    ask.add_argument("query")

    evaluate = sub.add_parser("evaluate", help="Run the reproducible candidate evaluation")
    evaluate.add_argument("--records", type=int, default=500)
    evaluate.add_argument("--corpus", default="data/seed_corpus.jsonl")
    evaluate.add_argument("--out", default="data/evaluation_results.json")
    evaluate.add_argument("--no-fresh", action="store_true", help="Do not reset before evaluation")
    evaluate.add_argument("--compare", action="store_true",
                          help="Run the deterministic and local-LLM configurations back to back.")

    calib = sub.add_parser("calibrate", help="Grid-search the controller's admission thresholds")
    calib.add_argument("--records", type=int, default=300)
    calib.add_argument("--corpus", default="data/seed_corpus.jsonl")
    calib.add_argument("--out", default="data/calibration.json")

    inspect = sub.add_parser("inspect", help="Print database state as JSON")
    inspect.add_argument("--kind", choices=["stats", "memories", "decisions", "transcripts", "backends"],
                         default="stats")
    inspect.add_argument("--limit", type=int, default=50)

    server = sub.add_parser("serve", help="Start the local web interface")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    config = build_config(args)

    if args.command == "migrate":
        with connect(args.db) as conn:
            migrate(conn)
        print(f"Migrated {args.db}")
    elif args.command == "reset":
        reset_database(args.db)
        print(f"Reset {args.db}")
    elif args.command == "doctor":
        client = get_client(config)
        engine = KiviMemoryEngine(args.db, config)
        print(json.dumps({"ollama": client.doctor(), "resolved_backends": engine.backends()},
                         indent=2, sort_keys=True))
    elif args.command == "metrics":
        engine = KiviMemoryEngine(args.db, config)
        payload = {
            "database": engine.stats(),
            "growth": engine.growth(),
            "model_usage_this_process": engine.usage(),
            "backends": engine.backends(),
            "saved_reports": {},
        }
        for label, path in (("last_ingest", "data/ingest_report.json"),
                            ("evaluation", "data/evaluation_results.json"),
                            ("calibration", "data/calibration.json")):
            try:
                payload["saved_reports"][label] = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload["saved_reports"][label] = f"not present: {path}"
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif args.command == "seed":
        records = generate_corpus(args.records)
        write_jsonl(records, args.out)
        engine = KiviMemoryEngine(args.db, config)
        result = engine.ingest_records(records)
        report = engine.write_ingest_report(result, args.report, source=args.out)
        print(json.dumps({"corpus": args.out, **result, "report_written_to": args.report,
                          "latency": report["latency"], "model_usage": report["model_usage"],
                          "growth": report["growth"]}, indent=2, sort_keys=True))
    elif args.command == "import":
        records = read_jsonl(args.path)
        engine = KiviMemoryEngine(args.db, config)
        result = engine.ingest_records(records)
        report = engine.write_ingest_report(result, args.report, source=args.path)
        print(json.dumps({**result, "report_written_to": args.report,
                          "latency": report["latency"], "model_usage": report["model_usage"],
                          "growth": report["growth"]}, indent=2, sort_keys=True))
    elif args.command == "ask":
        result = KiviMemoryEngine(args.db, config).ask(args.query)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "evaluate":
        if args.compare:
            result = compare_backends(
                db_path=args.db, corpus_path=args.corpus, out_path=args.out, records=args.records
            )
            print(json.dumps(result["comparison"], indent=2, sort_keys=True))
        else:
            result = run_evaluation(
                db_path=args.db,
                corpus_path=args.corpus,
                out_path=args.out,
                records=args.records,
                fresh=not args.no_fresh,
                config=config,
            )
            print(json.dumps(result["summary"], indent=2, sort_keys=True))
    elif args.command == "calibrate":
        result = calibrate(db_path=args.db, corpus_path=args.corpus, out_path=args.out,
                           records=args.records, config=config)
        print(json.dumps({"verdict": result["verdict"], "best": result["best"],
                          "shipped_defaults": result["shipped_defaults"],
                          "memory_count_range": result["memory_count_range"],
                          "mean_score_by_threshold_value": result["mean_score_by_threshold_value"],
                          "grid_size": result["grid_size"]}, indent=2, sort_keys=True))
    elif args.command == "inspect":
        engine = KiviMemoryEngine(args.db, config)
        if args.kind == "stats":
            payload = engine.stats()
        elif args.kind == "backends":
            payload = engine.backends()
        elif args.kind == "memories":
            payload = engine.memories(args.limit)
        elif args.kind == "decisions":
            payload = engine.decisions(args.limit)
        else:
            payload = engine.transcripts(args.limit)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "serve":
        serve(args.db, args.host, args.port, config)
