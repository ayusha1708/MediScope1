"""Evaluate routing on manually authored queries outside the synthetic holdout."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from app.classification.data import load_rows
from app.classification.router import classify_question
from tests.test_routing import MANUAL_CASES

AI_SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = AI_SERVICE_DIR.parent
MODEL_FILE = AI_SERVICE_DIR / "models" / "query_classifier.joblib"
TRAIN_FILE = PROJECT_DIR / "data" / "query_train.csv"
TEST_FILE = PROJECT_DIR / "data" / "query_test.csv"
REPORT_FILE = AI_SERVICE_DIR / "reports" / "diverse_routing_metrics.json"

LABELS = ("SAFETY", "KNOWLEDGE_RAG", "LITERATURE_CONTEXT", "GENERAL_INFORMATION")


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def main() -> None:
    model = joblib.load(MODEL_FILE)
    train = load_rows(TRAIN_FILE)
    test = load_rows(TEST_FILE)
    known_queries = {
        normalized(row["query"]) for row in (*train, *test)
    }

    actual = [label for label, _ in MANUAL_CASES]
    predicted = []
    details = []
    for expected, question in MANUAL_CASES:
        result = classify_question(model, question)
        predicted.append(result.route)
        details.append(
            {
                "question": question,
                "expected": expected,
                "predicted": result.route,
                "model_route": result.model_route,
                "confidence": result.confidence,
                "confidence_margin": result.confidence_margin,
                "reason": result.reason,
                "correct": expected == result.route,
            }
        )

    report = {
        "dataset": "manually authored diverse routing regression set",
        "examples": len(actual),
        "labels": list(LABELS),
        "accuracy": float(accuracy_score(actual, predicted)),
        "classification_report": classification_report(
            actual,
            predicted,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix_labels": list(LABELS),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            actual, predicted, labels=list(LABELS)
        ).tolist(),
        "safety_recall": float(
            classification_report(
                actual,
                predicted,
                labels=list(LABELS),
                output_dict=True,
                zero_division=0,
            )["SAFETY"]["recall"]
        ),
        "emergency_false_negatives": [
            item for item in details
            if item["expected"] == "SAFETY" and item["predicted"] != "SAFETY"
        ],
        "exact_query_leakage": {
            "manual_queries_in_train_or_test": sum(
                normalized(question) in known_queries
                for _, question in MANUAL_CASES
            ),
            "checked_against": [str(TRAIN_FILE), str(TEST_FILE)],
        },
        "cases": details,
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Manual examples: {len(actual)}")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"Safety recall: {report['safety_recall']:.3f}")
    print(f"Emergency false negatives: {len(report['emergency_false_negatives'])}")
    print(
        "Exact-query leakage: "
        f"{report['exact_query_leakage']['manual_queries_in_train_or_test']}"
    )
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
