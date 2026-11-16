"""Conservative first-pass checks for urgent and personal medical requests."""

from __future__ import annotations

import re
import unicodedata


# Educational prompts about a symptom should not be treated as a live emergency.
# Direct first-person/second-person reports and unqualified acute phrases should.
EDUCATIONAL_PREFIX = re.compile(
    r"^\s*(?:what|which|why|how does|how do|explain|define|describe|"
    r"tell me about|learn about|meaning of|signs? of|symptoms? of|"
    r"causes? of|associated with|related to|risk factors? for)\b",
    re.IGNORECASE,
)

ACUTE_BREATHING = re.compile(
    r"(?:"
    r"can(?:not|'?t|\s+not)\s+breath(?:e|ing)?|"
    r"could\s+not\s+breath(?:e|ing)?|"
    r"unable\s+to\s+(?:get|catch)?\s*(?:my\s+)?breath(?:e|ing)?|"
    r"(?:having|have|experiencing|with)\s+(?:severe\s+)?(?:difficulty|trouble|problems?)\s+"
    r"(?:to\s+)?breath(?:e|ing)|"
    r"(?:difficulty|trouble|problems?)\s+breath(?:e|ing)|"
    r"can(?:not|'?t)\s+catch\s+my\s+breath|"
    r"breathing\s+(?:is|feels?)\s+(?:very\s+)?difficult|"
    r"short(?:ness)?\s+of\s+breath|breathlessness|"
    r"struggling\s+to\s+breath(?:e|ing)|hard\s+to\s+breath(?:e|ing)|"
    r"gasping\s+for\s+air|not\s+(?:getting|able\s+to\s+get)\s+(?:enough\s+)?air|"
    r"air\s+hunger|suffocating|choking|"
    r"breathing\s+(?:very\s+)?(?:badly|poorly)|"
    r"(?:have|having)\s+(?:severe\s+)?breathing\s+problems?"
    r")",
    re.IGNORECASE,
)

PERSONAL_OR_LIVE_CONTEXT = re.compile(
    r"\b(?:i|i'm|im|my|me|you|you're|youre|someone|somebody|they|he|she|"
    r"right now|currently|at the moment)\b",
    re.IGNORECASE,
)

URGENT_OTHER_PATTERN = re.compile(
    r"\b(?:unconscious|not breathing|severe bleeding|overdosed?|stroke(?: symptoms?)?|"
    r"face drooping|sudden weakness on one side)\b",
    re.IGNORECASE,
)

PERSONAL_DECISION_PATTERNS = (
    re.compile(r"\b(?:diagnose me|do i have|what's wrong with me|what is wrong with me)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:should|can|could|may)\s+i\b.{0,70}\b(?:take|stop|change|increase|decrease|wait|go)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bwhat (?:dose|dosage) should i\b", re.IGNORECASE),
    re.compile(r"\bmy (?:symptoms|medication|prescription|pain|breathing)\b", re.IGNORECASE),
)


def _normalize(question: str) -> str:
    """Normalize punctuation/apostrophes and common breath spelling variants."""
    text = unicodedata.normalize("NFKC", question).casefold()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    # These are spelling variants, not individual question exceptions.
    text = re.sub(r"\bbreth(?:e|ing)?\b", "breathe", text)
    text = re.sub(r"\bbreathng\b", "breathing", text)
    text = re.sub(r"\bbrething\b", "breathing", text)
    return text


def _urgent_breathing_match(question: str) -> bool:
    if not ACUTE_BREATHING.search(question):
        return False
    # "What are symptoms of difficulty breathing?" is educational. A direct
    # report such as "I can't breathe" or "someone is gasping" is urgent.
    if EDUCATIONAL_PREFIX.search(question) and not PERSONAL_OR_LIVE_CONTEXT.search(question):
        return False
    return True


def _urgent_other_match(question: str) -> bool:
    if not URGENT_OTHER_PATTERN.search(question):
        return False
    if EDUCATIONAL_PREFIX.search(question) and not PERSONAL_OR_LIVE_CONTEXT.search(question):
        return False
    return True


def safety_message(question: str) -> str | None:
    """Return a safety boundary for obvious high-risk wording, otherwise None."""
    normalized = _normalize(question)
    if _urgent_breathing_match(normalized) or _urgent_other_match(normalized):
        return (
            "This may describe an urgent situation. MediScope cannot assess emergencies; "
            "contact your local emergency service or a medical professional now."
        )
    if any(pattern.search(normalized) for pattern in PERSONAL_DECISION_PATTERNS):
        return (
            "MediScope cannot diagnose you or recommend personal treatment or medication "
            "changes. Please contact a qualified healthcare professional."
        )
    return None
