r"""
### Rank the real documents

[Sentence Transformers semantic search](https://sbert.net/examples/sentence_transformer/applications/semantic-search/README.html)
provides cosine scores. Translate positions back to dataset IDs and order ties
by ascending original ID, including ties at the cutoff. We request every
candidate before truncating, so this exact demonstration costs memory linear
in query count times corpus size; use an index with deterministic boundary-tie
handling for a much larger collection. Which papers does the first query find?
"""

from sentence_transformers import util


def rank_dense(query_vectors, document_vectors, qids=query_ids, dids=corpus_ids):
    # guard: vectors and IDs must line up one-to-one
    if len(query_vectors) != len(qids) or len(document_vectors) != len(dids):
        raise ValueError("Vector and ID counts do not match")
    if len(set(qids)) != len(qids) or len(set(dids)) != len(dids):
        raise ValueError("IDs must be unique")

    # score every (query, document) pair, then map positions back to dataset IDs
    hits = util.semantic_search(query_vectors, document_vectors, top_k=len(dids))
    rankings = {}
    for qid, row in zip(qids, hits):
        ordered = [
            {"corpus_id": dids[hit["corpus_id"]], "score": float(hit["score"])} for hit in row
        ]
        # sort by descending score, break ties by ascending original ID, keep top MAX_K
        rankings[qid] = sorted(ordered, key=lambda h: (-h["score"], h["corpus_id"]))[:MAX_K]
    return rankings


# rank the naive baseline and inspect the first query's top hits
naive_rankings = rank_dense(naive_query_embeddings, naive_corpus_embeddings)
first_hits = naive_rankings[query_ids[0]]
display(
    [
        {
            "id": h["corpus_id"],
            "score": h["score"],
            "labelled_relevant": h["corpus_id"] in relevant_docs[query_ids[0]],
            "text": corpus[h["corpus_id"]][:300],
        }
        for h in first_hits[:3]
    ]
)
