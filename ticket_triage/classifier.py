"""Ticket classification: tags each ticket with a category, urgency, and confidence.

Two implementations are provided:

- ``RuleBasedClassifier``: deterministic keyword/regex scoring. Zero external
  dependencies, zero cost, always available. Used as the default and as the
  automatic fallback if the LLM classifier is unavailable or errors out.
- ``LLMClassifier``: delegates to an Anthropic model for higher-quality,
  context-aware classification. Only used when ``ANTHROPIC_API_KEY`` is set
  and the ``anthropic`` package is installed.

Both share the ``BaseClassifier`` interface so the pipeline can swap them
(or add new ones, e.g. a fine-tuned sklearn/transformers model) without any
other code changing.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections import Counter

from .models import Category, ClassificationResult, Ticket, Urgency

# ---------------------------------------------------------------------------
# Keyword tables for the rule-based classifier. Each category maps to a list
# of (pattern, weight) tuples. Patterns are matched case-insensitively against
# the ticket's subject + body.
# ---------------------------------------------------------------------------

_CATEGORY_KEYWORDS: dict[Category, list[tuple[str, float]]] = {
    Category.BILLING: [
        (r"\binvoice\b", 2.0), (r"\bcharged?\b", 1.5), (r"\brefund\b", 2.0),
        (r"\bbilling\b", 2.0), (r"\bpayment\b", 1.5), (r"\bsubscription\b", 1.0),
        (r"\bcredit card\b", 1.5), (r"\bprice\b", 1.0), (r"\boverdue\b", 1.5),
        (r"\bdouble.?charged\b", 2.5), (r"\breceipt\b", 1.0),
    ],
    Category.TECHNICAL_ISSUE: [
        (r"\berror\b", 1.5), (r"\bnot working\b", 2.0), (r"\bcrash(?:ed|ing)?\b", 2.0),
        (r"\bslow\b", 1.0), (r"\btimeout\b", 1.5), (r"\bfail(?:ed|ing|ure)?\b", 1.5),
        (r"\bloading\b", 1.0), (r"\bbroken\b", 1.5), (r"\b500\b", 1.5), (r"\bunable to\b", 1.0),
    ],
    Category.ACCOUNT_ACCESS: [
        (r"\bpassword\b", 2.0), (r"\blogin\b", 2.0), (r"\bsign in\b", 2.0),
        (r"\blocked out\b", 2.5), (r"\b2fa\b", 1.5), (r"\btwo.?factor\b", 1.5),
        (r"\breset\b", 1.0), (r"\baccount access\b", 2.0), (r"\bcan'?t access\b", 1.5),
        (r"\busername\b", 1.0),
    ],
    Category.BUG_REPORT: [
        (r"\bbug\b", 2.5), (r"\bglitch\b", 2.0), (r"\bunexpected behavior\b", 1.5),
        (r"\breproduce\b", 1.5), (r"\bregression\b", 1.5), (r"\bstack trace\b", 2.0),
        (r"\bconsole error\b", 1.5),
    ],
    Category.FEATURE_REQUEST: [
        (r"\bfeature request\b", 3.0), (r"\bwould be great if\b", 1.5),
        (r"\bcan you add\b", 1.5), (r"\bsuggest(?:ion)?\b", 1.0),
        (r"\bit would help if\b", 1.0), (r"\bplease consider\b", 1.0),
        (r"\bwish list\b", 1.5),
    ],
    Category.CANCELLATION: [
        (r"\bcancel\b", 2.5), (r"\bunsubscribe\b", 2.0), (r"\bclose my account\b", 2.5),
        (r"\bdowngrade\b", 1.5), (r"\bterminate\b", 1.5), (r"\bdelete my account\b", 2.0),
    ],
    Category.COMPLAINT: [
        (r"\bterrible\b", 2.0), (r"\bawful\b", 2.0), (r"\bworst\b", 2.0),
        (r"\bunacceptable\b", 2.0), (r"\bdisappointed\b", 1.5), (r"\bfrustrat(?:ed|ing)\b", 1.5),
        (r"\bcomplaint\b", 2.5), (r"\bridiculous\b", 1.5), (r"\bangry\b", 1.5),
    ],
    Category.SECURITY: [
        (r"\bhacked\b", 3.0), (r"\bbreach\b", 2.5), (r"\bunauthorized\b", 2.0),
        (r"\bphishing\b", 2.5), (r"\bsuspicious (?:login|activity)\b", 2.5),
        (r"\bdata leak\b", 2.5), (r"\bvulnerab\w*\b", 2.0), (r"\bcompromised\b", 2.5),
    ],
    Category.GENERAL_INQUIRY: [
        (r"\bhow do i\b", 1.0), (r"\bquestion\b", 0.8), (r"\bwondering\b", 0.5),
        (r"\bcould you (?:tell|explain)\b", 0.8), (r"\bwhat is\b", 0.5),
    ],
}

_URGENCY_KEYWORDS: list[tuple[str, Urgency, float]] = [
    # Negation phrases are listed first and weighted to outweigh the plain
    # positive keyword they contain (e.g. "not urgent" also matches \burgent\b,
    # so its LOW weight must exceed \burgent\b's HIGH weight to win the vote).
    (r"\bnot (?:that )?urgent\b", Urgency.LOW, 3.0),
    (r"\bisn'?t urgent\b", Urgency.LOW, 3.0),
    (r"\bno urgency\b", Urgency.LOW, 3.0),
    (r"\bnot (?:an? )?emergency\b", Urgency.LOW, 3.5),
    (r"\bnot critical\b", Urgency.LOW, 3.0),
    (r"\bemergency\b", Urgency.CRITICAL, 3.0),
    (r"\burgent(?:ly)?\b", Urgency.HIGH, 2.5),
    (r"\basap\b", Urgency.HIGH, 2.5),
    (r"\bimmediately\b", Urgency.HIGH, 2.0),
    (r"\bcritical\b", Urgency.CRITICAL, 2.5),
    (r"\bproduction (?:is )?down\b", Urgency.CRITICAL, 3.5),
    (r"\ball users? (?:are )?affected\b", Urgency.CRITICAL, 3.0),
    (r"\bsecurity breach\b", Urgency.CRITICAL, 3.5),
    (r"\bdata loss\b", Urgency.CRITICAL, 3.0),
    (r"\bcan'?t (?:log ?in|access)\b", Urgency.HIGH, 1.5),
    (r"\bright now\b", Urgency.HIGH, 1.5),
    (r"\btoday\b", Urgency.MEDIUM, 0.8),
    (r"\bwhen (?:you get a chance|possible)\b", Urgency.LOW, 1.5),
    (r"\bno rush\b", Urgency.LOW, 2.0),
    (r"\bjust curious\b", Urgency.LOW, 1.5),
    (r"\bminor\b", Urgency.LOW, 1.0),
]


class BaseClassifier(ABC):
    name: str = "base"

    @abstractmethod
    def classify(self, ticket: Ticket) -> ClassificationResult:
        ...


class RuleBasedClassifier(BaseClassifier):
    """Deterministic keyword-scoring classifier. No external dependencies."""

    name = "rule_based_v1"

    def classify(self, ticket: Ticket) -> ClassificationResult:
        text = ticket.full_text.lower()

        # --- Category scoring ---
        scores: Counter[Category] = Counter()
        matched_terms: dict[Category, list[str]] = {}
        for category, patterns in _CATEGORY_KEYWORDS.items():
            for pattern, weight in patterns:
                hits = re.findall(pattern, text)
                if hits:
                    scores[category] += weight * len(hits)
                    matched_terms.setdefault(category, []).append(pattern.strip("\\b"))

        if scores:
            top_category, top_score = scores.most_common(1)[0]
            total = sum(scores.values())
            # confidence reflects how dominant the winning category is,
            # scaled down a bit if the absolute signal is weak.
            dominance = top_score / total if total else 0.0
            strength = min(top_score / 4.0, 1.0)
            confidence = round(0.4 * strength + 0.6 * dominance, 2)
            confidence = max(0.35, min(confidence, 0.97))
        else:
            top_category = Category.UNKNOWN
            confidence = 0.2

        # --- Urgency scoring ---
        urgency_scores: Counter[Urgency] = Counter()
        for pattern, urgency, weight in _URGENCY_KEYWORDS:
            hits = re.findall(pattern, text)
            if hits:
                urgency_scores[urgency] += weight * len(hits)

        if urgency_scores:
            urgency = max(urgency_scores.items(), key=lambda kv: (kv[1], kv[0].rank))[0]
        else:
            urgency = Urgency.MEDIUM  # sensible default: neither dismiss nor over-escalate

        rationale_bits = []
        if top_category in matched_terms:
            rationale_bits.append(
                f"category cues: {', '.join(sorted(set(matched_terms[top_category]))[:4])}"
            )
        rationale_bits.append(f"urgency signal: {urgency.value}")
        rationale = "; ".join(rationale_bits) if rationale_bits else "no strong keyword signal"

        return ClassificationResult(
            category=top_category,
            urgency=urgency,
            confidence=confidence,
            rationale=rationale,
            classifier_name=self.name,
        )


_LLM_SYSTEM_PROMPT = """You are a support-ticket triage classifier for a customer support team.
Given a ticket's subject and body, respond with ONLY a JSON object (no markdown fences, no prose)
with exactly these keys:
  "category": one of ["billing","technical_issue","account_access","bug_report",
                       "feature_request","cancellation","complaint","security",
                       "general_inquiry","unknown"]
  "urgency": one of ["low","medium","high","critical"]
  "confidence": a number between 0 and 1 reflecting how certain you are
  "rationale": a short (<20 word) explanation of the classification
"""


class LLMClassifier(BaseClassifier):
    """Anthropic-model-backed classifier for higher accuracy on ambiguous tickets.

    Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY`` environment
    variable. Falls back gracefully is handled by the caller (pipeline), not here:
    this class raises if it cannot run so the pipeline can decide what to do.
    """

    name = "llm_v1"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'anthropic' package is required for LLMClassifier. "
                "Install it with `pip install anthropic`."
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot use LLMClassifier.")

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model

    def classify(self, ticket: Ticket) -> ClassificationResult:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=_LLM_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Subject: {ticket.subject}\n\nBody: {ticket.body}",
                }
            ],
        )
        raw_text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()
        raw_text = re.sub(r"^```(json)?|```$", "", raw_text.strip()).strip()

        try:
            data = json.loads(raw_text)
            category = Category(data["category"])
            urgency = Urgency(data["urgency"])
            confidence = float(data["confidence"])
            rationale = str(data.get("rationale", ""))
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise RuntimeError(f"LLMClassifier returned unparseable output: {raw_text!r}") from exc

        return ClassificationResult(
            category=category,
            urgency=urgency,
            confidence=max(0.0, min(confidence, 1.0)),
            rationale=rationale,
            classifier_name=self.name,
        )


def get_default_classifier() -> BaseClassifier:
    """Return the LLM classifier if usable, otherwise the rule-based fallback."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return LLMClassifier()
        except RuntimeError:
            pass
    return RuleBasedClassifier()
