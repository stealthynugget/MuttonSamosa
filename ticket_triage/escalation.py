"""Builds a concise, pre-filled escalation packet for human agents."""

from __future__ import annotations

import re

from .models import Category, ClassificationResult, Ticket, Urgency

# Which team a category routes to when a human needs to take over.
_TEAM_ROUTING: dict[Category, str] = {
    Category.BILLING: "Billing Support",
    Category.TECHNICAL_ISSUE: "Technical Support (L2)",
    Category.ACCOUNT_ACCESS: "Account Security",
    Category.BUG_REPORT: "Engineering Triage",
    Category.FEATURE_REQUEST: "Product Team",
    Category.CANCELLATION: "Retention Team",
    Category.COMPLAINT: "Customer Success Lead",
    Category.SECURITY: "Security / Trust & Safety",
    Category.GENERAL_INQUIRY: "General Support",
    Category.UNKNOWN: "General Support",
}

# SLA targets (minutes) by urgency, used to communicate response deadlines
# to the human queue.
_SLA_MINUTES: dict[Urgency, int] = {
    Urgency.CRITICAL: 15,
    Urgency.HIGH: 60,
    Urgency.MEDIUM: 60 * 8,
    Urgency.LOW: 60 * 24,
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Cheap extractive summary: first N sentences, cleaned of whitespace.

    Kept intentionally simple/deterministic (no LLM call required) so that
    escalation always works even if an LLM provider is down or unset. Swap
    in an LLM-based summarizer here for higher-quality summaries when an API
    key is available -- the calling code (pipeline.py) is agnostic to how
    the summary was produced.
    """
    cleaned = " ".join(text.split())
    sentences = _SENTENCE_SPLIT.split(cleaned)
    summary = " ".join(sentences[:max_sentences])
    if len(sentences) > max_sentences:
        summary += " [...]"
    return summary


def assigned_team(category: Category) -> str:
    return _TEAM_ROUTING.get(category, "General Support")


def sla_minutes(urgency: Urgency) -> int:
    return _SLA_MINUTES.get(urgency, _SLA_MINUTES[Urgency.MEDIUM])


def build_escalation_summary(ticket: Ticket, classification: ClassificationResult) -> str:
    body_summary = _extractive_summary(ticket.body)
    team = assigned_team(classification.category)
    sla = sla_minutes(classification.urgency)

    return (
        f"[{classification.urgency.value.upper()}] {ticket.subject.strip()}\n"
        f"Ticket: {ticket.ticket_id} | Customer: {ticket.customer_id} | Channel: {ticket.channel}\n"
        f"Category: {classification.category.value} "
        f"(confidence {classification.confidence:.0%}, via {classification.classifier_name})\n"
        f"Route to: {team} | SLA: respond within {sla} min\n"
        f"Summary: {body_summary}\n"
        f"Classifier notes: {classification.rationale}"
    )
