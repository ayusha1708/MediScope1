"""Safety-first, explainable query routing around the trained classifier.

The statistical model handles wording variation. Small explicit rules protect the
approved operational boundaries: safety always wins, explicit research-analysis
requests route to literature, medically detailed questions use evidence retrieval,
and uncertain questions avoid unsupported general answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.classification.data import LABELS
from app.safety.screening import safety_message


LITERATURE_CUES = re.compile(
    r"\b(?:research paper(?:s)?|published paper(?:s)?|study|studies|trial(?:s)?|"
    r"systematic review(?:s)?|literature|meta-analysis|research methods?|sample size|"
    r"study findings?|study limitations?|compare (?:the )?(?:papers?|studies?|trials?)|"
    r"conclusions? of (?:the )?(?:papers?|studies?|reviews?))\b",
    re.IGNORECASE,
)
GENERAL_DEFINITION_CUES = re.compile(
    r"\b(?:what does .+ mean|what is .+|define .+|"
    r"meaning of .+|what does the abbreviation .+ stand for|"
    r"explain .+ in (?:simple|plain|basic) terms?)\b",
    re.IGNORECASE,
)
MEDICAL_DETAIL_CUES = re.compile(
    r"\b(?:symptoms?|signs?|serious|severe|causes?|risk factors?|"
    r"complications?|stages?|screening|prevention|treatment|prognosis|"
    r"mechanisms?|physiology|pathology|guidelines?|recommendations?|"
    r"how does .+ affect|how can .+ cause|associated with|related to|"
    r"medical evidence|clinical guidance|source-supported)\b",
    re.IGNORECASE,
)
DETAILED_DEFINITION_CUES = re.compile(
    r"\b(?:mechanism|cause|causes|risk factors?|complications?|treatment|"
    r"screening|prevention|stages?|physiology|pathology|effects?|affect|"
    r"associated|related)\b",
    re.IGNORECASE,
)

# These are routing thresholds, not calibrated medical probabilities.
MIN_CONFIDENCE = 0.55
MIN_MARGIN = 0.10


@dataclass(frozen=True)
class ClassificationResult:
    route: str
    confidence: float
    confidence_margin: float
    model_route: str
    reason: str
    safety_response: str | None = None


def _model_scores(model: Any, question: str) -> tuple[str, float, float]:
    """Return route, top score, and separation from the runner-up.

    Scores are uncertainty signals only. They are not calibrated probabilities
    and must never take precedence over the safety screen.
    """
    predicted = str(model.predict([question])[0])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([question])[0]
        ordered = sorted((float(value) for value in probabilities), reverse=True)
        return predicted, ordered[0], ordered[0] - ordered[1]

    if hasattr(model, "decision_function"):
        import numpy as np

        scores = np.asarray(model.decision_function([question])).reshape(-1)
        scores -= scores.max()
        probabilities = np.exp(scores) / np.exp(scores).sum()
        ordered = sorted((float(value) for value in probabilities), reverse=True)
        return predicted, ordered[0], ordered[0] - ordered[1]
    return predicted, 1.0, 1.0


def classify_question(model: Any, question: str) -> ClassificationResult:
    """Apply safety and explainable routing policy before model-only routing."""
    boundary = safety_message(question)
    if boundary:
        return ClassificationResult(
            route="SAFETY",
            confidence=1.0,
            confidence_margin=1.0,
            model_route="SAFETY",
            reason="safety_screen_matched",
            safety_response=boundary,
        )

    model_route, model_confidence, confidence_margin = _model_scores(model, question)
    if model_route not in LABELS:
        raise ValueError(f"Classifier returned an unknown route: {model_route}")

    if LITERATURE_CUES.search(question):
        return ClassificationResult(
            route="LITERATURE_CONTEXT",
            confidence=max(model_confidence, 0.90),
            confidence_margin=confidence_margin,
            model_route=model_route,
            reason="explicit_research_or_study_request",
        )

    uncertain = (
        model_confidence < MIN_CONFIDENCE or confidence_margin < MIN_MARGIN
    )
    is_simple_definition = (
        GENERAL_DEFINITION_CUES.search(question)
        and not DETAILED_DEFINITION_CUES.search(question)
    )
    if is_simple_definition and not uncertain:
        return ClassificationResult(
            route="GENERAL_INFORMATION",
            confidence=max(model_confidence, 0.75),
            confidence_margin=confidence_margin,
            model_route=model_route,
            reason="explicit_low_risk_definition_request",
        )

    # Medical-detail questions need evidence even when phrased as "what are"
    # questions. This prevents broad definition matching from routing symptom,
    # cause, risk, and complication questions to general chat.
    if MEDICAL_DETAIL_CUES.search(question):
        return ClassificationResult(
            route="KNOWLEDGE_RAG",
            confidence=model_confidence,
            confidence_margin=confidence_margin,
            model_route=model_route,
            reason="medical_detail_requires_evidence_retrieval",
        )

    if uncertain:
        return ClassificationResult(
            route="KNOWLEDGE_RAG",
            confidence=model_confidence,
            confidence_margin=confidence_margin,
            model_route=model_route,
            reason="low_confidence_safe_evidence_fallback",
        )

    return ClassificationResult(
        route=model_route,
        confidence=model_confidence,
        confidence_margin=confidence_margin,
        model_route=model_route,
        reason="trained_model",
    )
