"""Classify one question with the saved MediScope baseline model."""

import argparse
from pathlib import Path

import joblib

from app.classification.router import classify_question

MODEL_FILE = Path(__file__).resolve().parent / "models" / "query_classifier.joblib"
ROUTE_DESCRIPTIONS = {
    "KNOWLEDGE_RAG": "Search approved medical documents for supporting evidence.",
    "LITERATURE_CONTEXT": "Search research papers and summarize study-level evidence.",
    "GENERAL_INFORMATION": "Give a brief, low-risk educational explanation.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a MediScope question.")
    parser.add_argument("question", help="A question to classify")
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        raise SystemExit(
            "Trained model not found. Run `py train_query_classifier.py` first."
        )

    model = joblib.load(MODEL_FILE)
    result = classify_question(model, args.question)
    print(f"Question: {args.question}")
    print(f"Route: {result.route}")
    if result.safety_response:
        print(f"Safety response: {result.safety_response}")
        return
    print(f"Confidence signal: {result.confidence:.3f}")
    print(f"Runner-up margin: {result.confidence_margin:.3f}")
    print("Confidence note: heuristic routing signal, not guaranteed accuracy.")
    print(f"Routing reason: {result.reason}")
    print(f"Model route before policy: {result.model_route}")
    print(f"Next action: {ROUTE_DESCRIPTIONS[result.route]}")


if __name__ == "__main__":
    main()
