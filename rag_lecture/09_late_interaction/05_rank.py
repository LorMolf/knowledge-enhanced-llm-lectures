r"""
### Rank with native MaxSim

Score every document for every test question, in document chunks to limit peak memory; document vectors stay on CPU and each score block returns there. This is exhaustive on our small closed corpus, not a production approximate index. The native scorer also chunks its intermediates, though a single long pair can exceed its budget. Ties are ordered by original document ID before the cutoff. The helper is reused for validation and the next checkpoint.
"""

def rank_multivector(model, query_vectors, document_vectors, qids=query_ids):
    # sanity check: one query vector per qid, one doc vector per corpus id
    assert len(qids) == len(query_vectors) and len(document_vectors) == len(corpus_ids)
    rankings = {}
    for qid, query_vector in zip(qids, query_vectors):
        # score in 64-doc chunks so peak memory stays bounded
        chunks = [
            model.similarity([query_vector], document_vectors[start : start + 64], device=DEVICE)
            .flatten()
            .cpu()
            for start in range(0, len(document_vectors), 64)
        ]
        scores = torch.cat(chunks).tolist()
        # build hits and sort by score desc, then corpus_id asc for stable ties
        hits = [{"corpus_id": did, "score": float(score)} for did, score in zip(corpus_ids, scores)]
        rankings[qid] = sorted(hits, key=lambda hit: (-hit["score"], hit["corpus_id"]))[:MAX_K]
    return rankings


# run the exhaustive MaxSim ranking over the closed corpus
colbert_rankings = rank_multivector(colbert_model, colbert_queries, colbert_docs)

# show the metrics so the ranking is measured, not just printed
print(evaluate_run("ColBERT", colbert_rankings))
