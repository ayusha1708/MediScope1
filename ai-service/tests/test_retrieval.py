import unittest

from sklearn.feature_extraction.text import TfidfVectorizer

from app.retrieval.search import SearchIndex, find_relevant_passages


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            {"index": 0, "text": "Asthma affects the airways and can cause wheezing."},
            {"index": 1, "text": "Iron deficiency can contribute to anemia."},
            {"index": 2, "text": "This passage discusses unrelated library policies."},
        ]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        vectors = vectorizer.fit_transform([chunk["text"] for chunk in chunks]).tocsr()
        self.index = SearchIndex(
            vectorizer=vectorizer,
            vectors=vectors,
            chunks=chunks,
            document={"document_id": "test"},
        )

    def test_unrelated_zero_score_passages_are_not_returned(self):
        results = find_relevant_passages("What causes wheezing?", self.index, limit=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(result.score > 0 for result in results))
        self.assertNotIn("unrelated library policies", results[-1].chunk["text"])

    def test_results_are_descending_by_cosine_similarity(self):
        results = find_relevant_passages("medical information", self.index, limit=3)
        scores = [result.score for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_overlap_returns_no_evidence_candidates(self):
        results = find_relevant_passages("quantum mechanics", self.index, limit=3)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
