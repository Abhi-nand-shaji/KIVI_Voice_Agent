"""Runtime configuration for the Kivi memory prototype.

No API keys anywhere. Every model-backed path talks to a locally running
Ollama daemon over plain HTTP, and every path has a deterministic fallback so
the prototype still runs on a reviewer's machine with nothing installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


EXTRACTOR_MODES = ("rule", "llm", "hybrid", "auto")
NLI_MODES = ("off", "llm", "cross-encoder", "auto")
RETRIEVAL_MODES = ("lexical", "hybrid", "auto")
ANSWERER_MODES = ("template", "llm", "auto")


@dataclass
class KiviConfig:
    # --- backend selection -------------------------------------------------
    extractor: str = "auto"
    nli: str = "auto"
    retrieval: str = "auto"
    answerer: str = "auto"

    # --- model transport ---------------------------------------------------
    # Default: a local Ollama daemon, no key, nothing leaves the machine.
    # Alternative: any OpenAI-compatible endpoint, for a reviewer who would
    # rather supply a key than download a 4.7 GB model. Same prompts, same
    # controller, same audit trail - only the transport changes.
    api_style: str = "ollama"          # "ollama" | "openai"
    llm_base_url: str = ""             # e.g. https://api.openai.com/v1
    llm_api_key: str = ""              # from KIVI_LLM_API_KEY, never committed
    ollama_host: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    embed_model: str = "nomic-embed-text"
    nli_model: str = ""            # defaults to llm_model
    cross_encoder_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli"

    # --- generation --------------------------------------------------------
    temperature: float = 0.0
    num_ctx: int = 4096
    request_timeout: float = 120.0
    health_timeout: float = 3.0
    max_retries: int = 2

    # --- controller thresholds (the deterministic part) --------------------
    admit_utility: float = 0.55
    admit_confidence: float = 0.50
    supersede_confidence: float = 0.72
    tentative_confidence: float = 0.62
    nli_contradiction_threshold: float = 0.65
    nli_equivalence_threshold: float = 0.70
    semantic_conflict_similarity: float = 0.55

    # Memory types where a contradiction means "the old one is now false".
    # Tasks and events are naturally multi-valued - two different to-dos in the
    # same scope are not in conflict, and letting a model call them one would
    # quietly delete the user's task list. The controller, not the model,
    # decides where inference is allowed to archive something.
    nli_conflict_types: tuple = ("preference", "fact", "project_state")

    # --- retrieval weights -------------------------------------------------
    w_lexical: float = 0.55
    w_semantic: float = 1.60
    w_confidence: float = 0.35
    w_utility: float = 0.45
    tentative_penalty: float = 0.25
    retrieval_floor: float = 0.75
    answer_threshold: float = 1.10

    # --- caching -----------------------------------------------------------
    cache_path: str = "data/llm_cache.db"
    use_cache: bool = True

    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "KiviConfig":
        cfg = cls(
            extractor=_env("KIVI_EXTRACTOR", "auto"),
            nli=_env("KIVI_NLI", "auto"),
            retrieval=_env("KIVI_RETRIEVAL", "auto"),
            answerer=_env("KIVI_ANSWERER", "auto"),
            ollama_host=_env("KIVI_OLLAMA_HOST", "http://127.0.0.1:11434"),
            llm_model=_env("KIVI_LLM_MODEL", "qwen2.5:7b-instruct"),
            embed_model=_env("KIVI_EMBED_MODEL", "nomic-embed-text"),
            nli_model=_env("KIVI_NLI_MODEL", ""),
            cross_encoder_model=_env("KIVI_CROSS_ENCODER_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli"),
            temperature=_env_float("KIVI_TEMPERATURE", 0.0),
            num_ctx=_env_int("KIVI_NUM_CTX", 4096),
            request_timeout=_env_float("KIVI_REQUEST_TIMEOUT", 120.0),
            cache_path=_env("KIVI_CACHE_PATH", "data/llm_cache.db"),
            api_style=_env("KIVI_API_STYLE", "ollama"),
            llm_base_url=_env("KIVI_LLM_BASE_URL", ""),
            llm_api_key=_env("KIVI_LLM_API_KEY", ""),
        )
        # Supplying a key or a base URL is itself the signal to use the
        # OpenAI-compatible transport, so a reviewer only has to set one thing.
        if cfg.api_style == "ollama" and (cfg.llm_api_key or cfg.llm_base_url):
            cfg.api_style = "openai"
            cfg.llm_base_url = cfg.llm_base_url or "https://api.openai.com/v1"
            if cfg.llm_model == "qwen2.5:7b-instruct":
                cfg.llm_model = _env("KIVI_LLM_MODEL", "gpt-4o-mini")
            if cfg.embed_model == "nomic-embed-text":
                cfg.embed_model = _env("KIVI_EMBED_MODEL", "text-embedding-3-small")
        cfg.use_cache = _env("KIVI_USE_CACHE", "1") not in ("0", "false", "no")
        return cfg

    def apply_overrides(self, **overrides) -> "KiviConfig":
        for key, value in overrides.items():
            if value is None:
                continue
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    @property
    def resolved_nli_model(self) -> str:
        return self.nli_model or self.llm_model

    def describe(self) -> dict:
        return {
            "api_style": self.api_style,
            "endpoint": self.llm_base_url or self.ollama_host,
            "api_key_present": bool(self.llm_api_key),
            "extractor": self.extractor,
            "nli": self.nli,
            "retrieval": self.retrieval,
            "answerer": self.answerer,
            "ollama_host": self.ollama_host,
            "llm_model": self.llm_model,
            "embed_model": self.embed_model,
            "nli_model": self.resolved_nli_model,
        }
