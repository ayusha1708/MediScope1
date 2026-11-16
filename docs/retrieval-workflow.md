# MediScope document matching and answer workflow

## Matching workflow

1. The user question is classified by the safety-first route layer.
2. `KNOWLEDGE_RAG` and `LITERATURE_CONTEXT` open the active MongoDB TF-IDF index.
3. The saved vocabulary and IDF weights are reconstructed exactly from `tfidf_indexes`.
4. The question is transformed with the same lowercase tokenization, unigram/bigram features, sublinear-term settings, and L2 normalization used for the stored passage vectors.
5. Stored sparse passage vectors are rebuilt from MongoDB.
6. Cosine similarity is calculated between the question vector and every eligible approved passage:

   `score = (question_vector · passage_vector) / (||question_vector|| × ||passage_vector||)`

   With L2-normalized TF-IDF vectors, this is the normalized weighted word/phrase overlap score, normally between 0 and 1.

7. Results are sorted from highest to lowest score. Only passages with a score greater than zero are returned; unrelated zero-overlap passages are not sent to Ollama.
8. The top three matching passages are passed to Ollama with document title, source, page, passage ID, and retrieval score.
9. Ollama must return JSON where every claim cites one or more supplied passage IDs. Invalid citations, unsupported evidence, empty claims, no matches, or an unavailable Ollama service produce the safe fallback instead of an uncited answer.

## Accuracy assessment

The implementation is internally consistent: the fitted vocabulary, IDF weights, token pattern, unigram/bigram configuration, lowercase handling, TF settings, and L2 normalization are saved with the active index and reused at query time. This is appropriate for the project's simple explainable retrieval requirement.

TF-IDF is lexical retrieval. It is strong when the question and source use related words or short phrases, but it does not understand synonyms as well as neural embeddings. The targeted change does not replace TF-IDF or add a new service; it prevents irrelevant zero-score passages from being treated as evidence and makes ties deterministic.

When documents change, rerun the existing indexing sequence so the active vocabulary and all passage vectors are rebuilt together:

```powershell
python store_document_chunks.py "path\to\document.pdf" "path\to\review.json"
python build_mongodb_tfidf_index.py
```

The retrieval unit tests verify descending score order, exclusion of zero-overlap passages, and safe no-match behavior.
