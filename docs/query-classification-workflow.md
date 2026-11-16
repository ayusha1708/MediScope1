# MediScope query classification workflow

## Goal

MediScope routes a question to one of three answer workflows: `KNOWLEDGE_RAG`, `LITERATURE_CONTEXT`, or `GENERAL_INFORMATION`. The classifier routes intent; it does not diagnose, prescribe, or decide whether a medical claim is true.

## Production order

1. **Validate input.** Reject an empty question before any model call.
2. **Safety screen first.** Urgent personal symptoms and personal diagnosis/medication decisions return a safety boundary. Breathing emergencies are normalized across contractions, spelling variants, and sentence structures. Educational prompts such as “What are symptoms of difficulty breathing?” are kept out of the live-emergency path. Safety always runs before model scoring.
3. **Statistical classification.** A scikit-learn TF-IDF pipeline predicts the route. Training compares Logistic Regression, Multinomial Naive Bayes, and Linear SVM with `StratifiedGroupKFold`; the final model is selected using training-only macro F1, with accuracy as a tie-breaker.
4. **Explainable policy layer.** High-signal explicit research terms (study, trial, paper, review, methods, findings, limitations, comparison) force `LITERATURE_CONTEXT`. Clear low-risk definitions can route to `GENERAL_INFORMATION`. Medically detailed questions involving symptoms, causes, risks, complications, screening, treatment, mechanisms, or guidance use `KNOWLEDGE_RAG`, even when they begin with “what are” or “what is”.
5. **Uncertainty handling.** The router uses the top class score and its margin over the runner-up. If the top score is below `0.55` or the margin is below `0.10`, the system uses `KNOWLEDGE_RAG`, the safer evidence-grounded path. These are heuristic routing signals, not calibrated probabilities or medical accuracy guarantees, and they never override emergency detection.
6. **Answer workflow.** General information uses the educational generator. RAG and literature routes search only the approved MongoDB index and generate grounded answers with references or a safe fallback.

The same wrapper is used by `answer_question.py`, the reusable `question_service`, and `predict_query_classifier.py`, preventing the standalone predictor from bypassing safety checks.

## Tech stack

- Python 3
- scikit-learn `1.9.0`
- TF-IDF features with Unicode accent normalization, sublinear term frequency, and unigram/bigram features
- Logistic Regression, Multinomial Naive Bayes, and Linear SVM candidate models
- joblib model persistence
- MongoDB/GridFS for approved document metadata, source files, passages, and the shared TF-IDF retrieval index
- FastAPI/backend integration through the reusable question service
- Ollama for grounded or general answer generation

## Evaluation and limitations

`train_query_classifier.py` writes raw and production-policy accuracy, precision, recall, macro F1, weighted F1, and confusion matrices to `ai-service/reports/query_classifier_metrics.json`. It also records exact-query and topic-group leakage checks. `evaluate_diverse_routing.py` evaluates 80 manually authored, exact-match leakage-free queries and reports safety recall and emergency false negatives in `ai-service/reports/diverse_routing_metrics.json`. The v2 holdout remains untouched during model selection. The current synthetic dataset and manual set are regression evidence, not proof of real-user accuracy or medical safety. The next meaningful accuracy improvement should come from a separately collected, consented, human-reviewed natural-question set.

The document flow remains independent and safety-preserving: files are validated and hashed, original files are stored, submissions start as `PENDING_ADMIN_REVIEW`, and only administrator-approved documents are extracted, chunked, vectorized, and made searchable.
