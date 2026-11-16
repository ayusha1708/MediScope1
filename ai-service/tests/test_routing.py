import unittest

from app.classification.router import classify_question
from app.safety.screening import safety_message


class FakeModel:
    def __init__(self, route, confidence=0.85, runner_up=0.05):
        self.route = route
        self.confidence = confidence
        self.runner_up = runner_up
        self.classes_ = ["GENERAL_INFORMATION", "KNOWLEDGE_RAG", "LITERATURE_CONTEXT"]

    def predict(self, questions):
        return [self.route]

    def predict_proba(self, questions):
        remaining = max(0.0, 1.0 - self.confidence - self.runner_up)
        values = {label: remaining for label in self.classes_}
        values[self.route] = self.confidence
        other = [label for label in self.classes_ if label != self.route]
        values[other[0]] = self.runner_up
        return [[values[label] for label in self.classes_]]


MANUAL_CASES = [
    # Live emergency reports and natural-language/spelling variations.
    ("SAFETY", "I can't breathe"),
    ("SAFETY", "I cannot breathe right now"),
    ("SAFETY", "I cant breath"),
    ("SAFETY", "I am having difficulty breathing"),
    ("SAFETY", "My breathing is very difficult"),
    ("SAFETY", "I am struggling to breathe"),
    ("SAFETY", "I am gasping for air"),
    ("SAFETY", "I feel like I am suffocating"),
    ("SAFETY", "Someone is having trouble breathing"),
    ("SAFETY", "My chest hurts and I am short of breath"),
    ("SAFETY", "I am not getting enough air"),
    ("SAFETY", "I can't catch my breath"),
    ("SAFETY", "I am breathng very badly"),
    ("SAFETY", "I have severe breathing problems right now"),
    ("SAFETY", "What should I do if I can't breathe right now?"),
    ("SAFETY", "I am unconscious and someone needs help"),
    ("SAFETY", "My friend is not breathing"),
    ("SAFETY", "I have severe bleeding right now"),
    ("SAFETY", "Someone may have overdosed"),
    ("SAFETY", "I think this may be a stroke and it is happening now"),
    # Educational emergency-symptom questions must not be falsely escalated.
    ("KNOWLEDGE_RAG", "What are symptoms of difficulty breathing?"),
    ("KNOWLEDGE_RAG", "What serious signs are associated with respiratory distress?"),
    ("KNOWLEDGE_RAG", "How does shortness of breath happen in asthma?"),
    ("KNOWLEDGE_RAG", "Explain why someone may feel breathless during exercise."),
    ("KNOWLEDGE_RAG", "What are the causes of gasping for air?"),
    # Medical detail and symptom policy overrides.
    ("KNOWLEDGE_RAG", "What are some serious symptoms associated with type 2 diabetes?"),
    ("KNOWLEDGE_RAG", "What signs are commonly associated with anemia?"),
    ("KNOWLEDGE_RAG", "What causes persistent headaches?"),
    ("KNOWLEDGE_RAG", "What complications can untreated hypertension cause?"),
    ("KNOWLEDGE_RAG", "What are the risk factors for kidney disease?"),
    ("KNOWLEDGE_RAG", "How is asthma severity classified?"),
    ("KNOWLEDGE_RAG", "What screening is recommended for colorectal cancer?"),
    ("KNOWLEDGE_RAG", "What treatment guidance exists for iron deficiency?"),
    ("KNOWLEDGE_RAG", "How does inflammation affect blood vessels?"),
    ("KNOWLEDGE_RAG", "What are warning signs of dehydration?"),
    ("KNOWLEDGE_RAG", "Can poor sleep affect blood pressure?"),
    ("KNOWLEDGE_RAG", "What is the mechanism behind insulin resistance?"),
    ("KNOWLEDGE_RAG", "Give a source-supported explanation of vaccine safety."),
    ("KNOWLEDGE_RAG", "What medical guidance explains migraine triggers?"),
    ("KNOWLEDGE_RAG", "What stages are used to describe chronic kidney disease?"),
    # Clear low-risk definitions.
    ("GENERAL_INFORMATION", "Could you define inflammation for a beginner?"),
    ("GENERAL_INFORMATION", "Define a chromosome in simple terms."),
    ("GENERAL_INFORMATION", "Define metabolism in simple terms."),
    ("GENERAL_INFORMATION", "What does prognosis mean in medical language?"),
    ("GENERAL_INFORMATION", "Explain what a vaccine is in plain language."),
    ("GENERAL_INFORMATION", "What does MRI stand for?"),
    ("GENERAL_INFORMATION", "What is the immune system?"),
    ("GENERAL_INFORMATION", "Explain the difference between bacteria and viruses simply."),
    ("GENERAL_INFORMATION", "What does benign mean?"),
    ("GENERAL_INFORMATION", "What is a red blood cell?"),
    ("GENERAL_INFORMATION", "Define the term antibody."),
    ("GENERAL_INFORMATION", "What does the abbreviation ECG stand for?"),
    ("GENERAL_INFORMATION", "Explain metabolism for a beginner."),
    ("GENERAL_INFORMATION", "What is DNA?"),
    ("GENERAL_INFORMATION", "What does the word symptom mean?"),
    # Explicit research/literature requests.
    ("LITERATURE_CONTEXT", "Compare the findings of two diabetes studies."),
    ("LITERATURE_CONTEXT", "Which research papers examine sleep and blood pressure?"),
    ("LITERATURE_CONTEXT", "What were the methods used in the asthma trial?"),
    ("LITERATURE_CONTEXT", "Summarize published trials on asthma treatments."),
    ("LITERATURE_CONTEXT", "What limitations did the systematic review report?"),
    ("LITERATURE_CONTEXT", "How do the conclusions of these two papers differ?"),
    ("LITERATURE_CONTEXT", "What was the sample size in the study?"),
    ("LITERATURE_CONTEXT", "Does the medical literature agree about vitamin D?"),
    ("LITERATURE_CONTEXT", "Find studies reporting results about diabetes prevention."),
    ("LITERATURE_CONTEXT", "Compare the research methods used by the trials."),
    ("LITERATURE_CONTEXT", "What did the published review find?"),
    ("LITERATURE_CONTEXT", "Summarize the limitations of the research papers."),
    ("LITERATURE_CONTEXT", "Which clinical trials investigated this treatment?"),
    ("LITERATURE_CONTEXT", "What outcomes did the study report?"),
    ("LITERATURE_CONTEXT", "Review the evidence across published studies."),
    # Ambiguous or low-confidence routing should prefer evidence retrieval.
    ("KNOWLEDGE_RAG", "Tell me about blood pressure."),
    ("KNOWLEDGE_RAG", "Can you discuss diabetes?"),
    ("KNOWLEDGE_RAG", "I need information about anemia."),
    ("KNOWLEDGE_RAG", "Please explain this health condition."),
    ("KNOWLEDGE_RAG", "What should a learner know about this medical topic?"),
    ("KNOWLEDGE_RAG", "Can you give me medical background on asthma?"),
    ("KNOWLEDGE_RAG", "Tell me about the effects of poor sleep."),
    ("KNOWLEDGE_RAG", "I want evidence about a health topic."),
    ("KNOWLEDGE_RAG", "Explain this condition medically."),
    ("KNOWLEDGE_RAG", "What does the evidence say about this condition?"),
]


class RoutingTests(unittest.TestCase):
    def test_manual_regression_suite_has_at_least_50_cases(self):
        self.assertGreaterEqual(len(MANUAL_CASES), 50)

    def test_manual_regression_cases(self):
        for expected, question in MANUAL_CASES:
            with self.subTest(expected=expected, question=question):
                model_route = "KNOWLEDGE_RAG" if expected == "SAFETY" else expected
                result = classify_question(FakeModel(model_route), question)
                self.assertEqual(result.route, expected)

    def test_safety_always_precedes_low_confidence_or_policy(self):
        for question in (
            "I can't breathe",
            "I am having difficulty breathing",
            "I cant breath right now",
        ):
            with self.subTest(question=question):
                result = classify_question(FakeModel("GENERAL_INFORMATION", 0.99), question)
                self.assertEqual(result.route, "SAFETY")
                self.assertEqual(result.reason, "safety_screen_matched")

    def test_educational_breathing_question_is_not_live_emergency(self):
        result = classify_question(
            FakeModel("GENERAL_INFORMATION"),
            "What are symptoms of difficulty breathing?",
        )
        self.assertEqual(result.route, "KNOWLEDGE_RAG")
        self.assertIsNone(safety_message("What are symptoms of difficulty breathing?"))

    def test_medical_detail_overrides_broad_what_are_pattern(self):
        result = classify_question(
            FakeModel("GENERAL_INFORMATION"),
            "What are some serious symptoms associated with type 2 diabetes?",
        )
        self.assertEqual(result.route, "KNOWLEDGE_RAG")
        self.assertEqual(result.reason, "medical_detail_requires_evidence_retrieval")

    def test_explicit_study_request_routes_to_literature(self):
        result = classify_question(
            FakeModel("KNOWLEDGE_RAG", 0.60),
            "Compare the methods and findings of these two studies.",
        )
        self.assertEqual(result.route, "LITERATURE_CONTEXT")
        self.assertEqual(result.reason, "explicit_research_or_study_request")

    def test_simple_definition_can_route_to_general(self):
        result = classify_question(
            FakeModel("KNOWLEDGE_RAG", 0.80),
            "What does inflammation mean?",
        )
        self.assertEqual(result.route, "GENERAL_INFORMATION")

    def test_low_probability_falls_back_to_evidence(self):
        result = classify_question(
            FakeModel("LITERATURE_CONTEXT", 0.40),
            "Tell me about blood pressure.",
        )
        self.assertEqual(result.route, "KNOWLEDGE_RAG")
        self.assertEqual(result.reason, "low_confidence_safe_evidence_fallback")

    def test_small_probability_margin_also_falls_back(self):
        result = classify_question(
            FakeModel("GENERAL_INFORMATION", 0.50, runner_up=0.45),
            "Tell me about a medical topic.",
        )
        self.assertEqual(result.route, "KNOWLEDGE_RAG")
        self.assertEqual(result.reason, "low_confidence_safe_evidence_fallback")


if __name__ == "__main__":
    unittest.main()
