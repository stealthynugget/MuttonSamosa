"""Core data models used across the ticket triage pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL_ISSUE = "technical_issue"
    ACCOUNT_ACCESS = "account_access"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    CANCELLATION = "cancellation"
    COMPLAINT = "complaint"
    GENERAL_INQUIRY = "general_inquiry"
    SECURITY = "security"
    UNKNOWN = "unknown"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2, "critical": 3}[self.value]


class Action(str, Enum):
    AUTO_RESPONDED = "auto_responded"
    ESCALATED = "escalated"


@dataclass
class Ticket:
    """A single incoming support ticket."""

    subject: str
    body: str
    customer_id: str = "unknown"
    channel: str = "email"  # email, form, api, chat
    ticket_id: str = field(default_factory=lambda: f"TCK-{uuid.uuid4().hex[:8].upper()}")
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def full_text(self) -> str:
        return f"{self.subject}\n{self.body}"


@dataclass
class ClassificationResult:
    """Output of the classification stage."""

    category: Category
    urgency: Urgency
    confidence: float  # 0.0 - 1.0
    rationale: str = ""
    classifier_name: str = "unknown"

    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold


@dataclass
class KBMatch:
    kb_id: str
    question: str
    resolution: str
    score: float


@dataclass
class TriageResult:
    """Final outcome of running a ticket through the full pipeline."""

    ticket: Ticket
    classification: ClassificationResult
    action: Action
    draft_response: Optional[str] = None
    kb_matches: list[KBMatch] = field(default_factory=list)
    escalation_summary: Optional[str] = None
    assigned_team: Optional[str] = None
    processed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""
