"""Natural-language-inference layer.

The previous controller detected conflicts with a hand-written whitelist of
"exclusive" predicates plus an exact string comparison on the object. That is a
uniqueness constraint on a schema, not inference: it cannot see that
"I've moved to Chennai" contradicts "User is based in Bangalore" unless both
happen to land on the same predicate, and it treats "bangalore" and
"bangalore, india" as two unrelated facts.

This module answers the actual question the controller needs answered:

    Given a stored memory (premise) and a new observation (hypothesis),
    is the hypothesis ENTAILED by, CONTRADICTED by, or NEUTRAL to the premise?

Two backends:

* `CrossEncoderNLI` - a genuine MNLI cross-encoder (DeBERTa-v3) with a softmax
  over three logits. Used when `transformers` + `torch` are installed.
* `OllamaNLI` - the local instruct model prompted as a three-way classifier
  with a calibrated probability. Weaker than a dedicated cross-encoder, but it
  needs nothing beyond the daemon already running for extraction.

`NullNLI` returns neutral for everything, which reduces the controller to its
original whitelist behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import KiviConfig
from .llm import OllamaClient


LABELS = ("entailment", "neutral", "contradiction")


@dataclass
class NLIResult:
    label: str
    probability: float
    backend: str
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "probability": round(float(self.probability), 3),
            "backend": self.backend,
            "rationale": self.rationale,
        }


class BaseNLI:
    name = "null"

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        raise NotImplementedError

    def contradicts(self, premise: str, hypothesis: str, threshold: float) -> tuple[bool, NLIResult]:
        result = self.classify(premise, hypothesis)
        return (result.label == "contradiction" and result.probability >= threshold), result

    def equivalent(self, premise: str, hypothesis: str, threshold: float) -> tuple[bool, NLIResult]:
        """Bidirectional entailment - used to merge paraphrase duplicates."""
        forward = self.classify(premise, hypothesis)
        if forward.label != "entailment" or forward.probability < threshold:
            return False, forward
        backward = self.classify(hypothesis, premise)
        ok = backward.label == "entailment" and backward.probability >= threshold
        merged = NLIResult(
            label="equivalence" if ok else backward.label,
            probability=min(forward.probability, backward.probability),
            backend=self.name,
            rationale=f"forward={forward.probability:.2f} backward={backward.probability:.2f}",
        )
        return ok, merged

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "real_nli": False}


class NullNLI(BaseNLI):
    name = "off"

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        return NLIResult("neutral", 0.0, self.name, "NLI disabled")


NLI_SYSTEM = (
    "You are a natural language inference classifier. "
    "Given a PREMISE and a HYPOTHESIS about the same user, decide whether the "
    "hypothesis is entailed by the premise, contradicts it, or is neutral.\n"
    "Rules:\n"
    "- entailment: the premise makes the hypothesis true, or they state the same fact in different words.\n"
    "- contradiction: both cannot be true of the same user at the same time.\n"
    "- neutral: they are about different things, or both can hold at once.\n"
    "Two different values of a single-valued attribute (current city, current manager, "
    "current preferred language) contradict each other. Two values of a multi-valued "
    "attribute (skills, interests, tasks) do not.\n"
    'Reply with JSON only: {"label": "entailment|neutral|contradiction", '
    '"probability": 0.0-1.0, "rationale": "one short clause"}'
)

NLI_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(LABELS)},
        "probability": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "probability"],
}


class OllamaNLI(BaseNLI):
    name = "ollama"

    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client = client
        self.model = model

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        payload = self.client.chat_json(
            NLI_SYSTEM,
            f"PREMISE: {premise}\nHYPOTHESIS: {hypothesis}",
            model=self.model,
            schema=NLI_SCHEMA,
            cache_kind="nli",
        )
        if not payload:
            return NLIResult("neutral", 0.0, self.name, "NLI backend unavailable")
        label = str(payload.get("label", "neutral")).strip().lower()
        if label not in LABELS:
            label = "neutral"
        try:
            probability = float(payload.get("probability", 0.0))
        except (TypeError, ValueError):
            probability = 0.0
        probability = max(0.0, min(1.0, probability))
        return NLIResult(label, probability, f"{self.name}:{self.model}", str(payload.get("rationale", ""))[:180])

    def describe(self) -> dict[str, Any]:
        return {"backend": f"ollama:{self.model}", "real_nli": True, "kind": "prompted-classifier"}


class CrossEncoderNLI(BaseNLI):
    """A real MNLI cross-encoder, if torch + transformers are available."""

    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        raw = getattr(self.model.config, "id2label", {}) or {}
        self.id2label = {int(k): str(v).lower() for k, v in raw.items()}

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        torch = self._torch
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1).tolist()
        best = max(range(len(probabilities)), key=lambda i: probabilities[i])
        label = self.id2label.get(best, LABELS[best] if best < len(LABELS) else "neutral")
        if "entail" in label:
            label = "entailment"
        elif "contradict" in label:
            label = "contradiction"
        else:
            label = "neutral"
        return NLIResult(label, probabilities[best], f"{self.name}:{self.model_name}")

    def describe(self) -> dict[str, Any]:
        return {"backend": f"cross-encoder:{self.model_name}", "real_nli": True, "kind": "mnli-softmax"}


def get_nli(config: KiviConfig, client: OllamaClient | None) -> BaseNLI:
    mode = config.nli
    if mode == "off":
        return NullNLI()
    if mode == "cross-encoder":
        return CrossEncoderNLI(config.cross_encoder_model)
    if mode == "llm":
        if client is None:
            return NullNLI()
        return OllamaNLI(client, config.resolved_nli_model)
    # auto: prefer a real cross-encoder, then the local instruct model, then off
    try:
        return CrossEncoderNLI(config.cross_encoder_model)
    except Exception:
        pass
    if client is not None and client.available() and client.has_model(config.resolved_nli_model):
        return OllamaNLI(client, config.resolved_nli_model)
    return NullNLI()
