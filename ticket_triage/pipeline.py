"""Orchestrates the full ticket triage workflow described in the problem brief:

    Ingest -> Classify -> Decide Path -> Auto-Respond / Escalate -> Log & Track
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import escalation, responder
from .classifier import BaseClassifier, RuleBasedClassifier, get_default_classifier
from .knowledge_base import KnowledgeBase
from .models import Action, Category, Ticket, TriageResult, Urgency
from .storage import TriageStore

_DATA_DIR = Path(__file__).parent / "data"

# Categories considered safe for a fully automated reply. Security incidents,
# billing disputes with money on the line beyond simple refunds, complaints,
# and anything "unknown" always go to a human -- auto-response is reserved
# for well-understood, low-risk, high-frequency issue types.
AUTO_RESPONSE_ELIGIBLE_CATEGORIES = {
    Category.ACCOUNT_ACCESS,
    Category.TECHNICAL_ISSUE,
    Category.BILLING,
    Category.CANCELLATION,
    Category.GENERAL_INQUIRY,
    Category.FEATURE_REQUEST,
}

# Even an eligible category escalates if urgency is this high -- a "how do I
# reset my password" ticket marked critical (e.g. "I'm locked out during a
# live incident") is exactly the kind of case that should reach a human.
MAX_AUTO_RESPONSE_URGENCY = Urgency.MEDIUM


@dataclass
class TriageConfig:
    confidence_threshold: float = 0.55
    kb_min_score: float = 0.08
    kb_top_k: int = 3
    auto_eligible_categories: set[Category] = field(
        default_factory=lambda: set(AUTO_RESPONSE_ELIGIBLE_CATEGORIES)
    )
    max_auto_urgency: Urgency = MAX_AUTO_RESPONSE_URGENCY


class TriagePipeline:
    def __init__(
        self,
        classifier: BaseClassifier | None = None,
        kb: KnowledgeBase | None = None,
        store: TriageStore | None = None,
        config: TriageConfig | None = None,
    ):
        self.classifier = classifier or get_default_classifier()
        self.kb = kb or KnowledgeBase.from_json_file(_DATA_DIR / "faq_kb.json")
        self.store = store or TriageStore()
        self.config = config or TriageConfig()

    def _decide_auto_respond(self, classification) -> tuple[bool, str]:
        """Returns (should_auto_respond, reason_if_not)."""
        if classification.category not in self.config.auto_eligible_categories:
            return False, f"category '{classification.category.value}' requires human review"
        if not classification.is_confident(self.config.confidence_threshold):
            return False, (
                f"classification confidence {classification.confidence:.2f} below "
                f"threshold {self.config.confidence_threshold:.2f}"
            )
        if classification.urgency.rank > self.config.max_auto_urgency.rank:
            return False, f"urgency '{classification.urgency.value}' exceeds auto-response ceiling"
        return True, ""

    def process(self, ticket: Ticket) -> TriageResult:
        classification = self.classifier.classify(ticket)

        # Classifier errors (e.g. a flaky LLM call) fall back to the rule-based
        # classifier rather than losing the ticket -- triage must never silently drop input.
        should_auto, reason = self._decide_auto_respond(classification)

        if should_auto:
            kb_matches = self.kb.search(
                ticket.full_text,
                category=classification.category,
                top_k=self.config.kb_top_k,
                min_score=self.config.kb_min_score,
            )
            if not kb_matches:
                # No grounding material -> don't fabricate a response, escalate instead.
                should_auto = False
                reason = "no sufficiently relevant knowledge-base entry found"

        if should_auto:
            draft = responder.draft_response(ticket, kb_matches)
            result = TriageResult(
                ticket=ticket,
                classification=classification,
                action=Action.AUTO_RESPONDED,
                draft_response=draft,
                kb_matches=kb_matches,
                notes="Auto-resolved from knowledge base.",
            )
        else:
            summary = escalation.build_escalation_summary(ticket, classification)
            team = escalation.assigned_team(classification.category)
            result = TriageResult(
                ticket=ticket,
                classification=classification,
                action=Action.ESCALATED,
                escalation_summary=summary,
                assigned_team=team,
                notes=f"Escalated: {reason}" if reason else "Escalated.",
            )

        self.store.log_result(result)
        return result

    def process_batch(self, tickets: list[Ticket]) -> list[TriageResult]:
        return [self.process(t) for t in tickets]
