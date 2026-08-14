"""Generates draft auto-responses grounded in knowledge-base matches.

Deliberately template-based rather than free-generation: every sentence in
the draft traces back to a specific KB entry, which keeps auto-responses
auditable and prevents the classic "confident but wrong" LLM hallucination
problem for a channel that goes straight to the customer without review.
"""

from __future__ import annotations

from .models import KBMatch, Ticket

_GREETING = "Hi there,"

_SIGNOFF = (
    "If this doesn't fully resolve things, just reply to this email and it'll "
    "go straight to a member of our team.\n\nBest,\nSupport Team"
)


def draft_response(ticket: Ticket, kb_matches: list[KBMatch]) -> str:
    if not kb_matches:
        raise ValueError("draft_response requires at least one KB match")

    top = kb_matches[0]
    lines = [
        _GREETING,
        "",
        f"Thanks for reaching out about \"{ticket.subject.strip()}\". "
        f"Here's how to resolve this:",
        "",
        top.resolution,
    ]

    # If there's a strong second match on a related question, offer it too
    # rather than padding the response with a marginal one.
    if len(kb_matches) > 1 and kb_matches[1].score >= 0.2:
        lines += [
            "",
            f"You might also find this useful: {kb_matches[1].resolution}",
        ]

    lines += ["", _SIGNOFF]
    return "\n".join(lines)
