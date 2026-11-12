r"""
### Score BGE's top ten with a cross-encoder

Fusion combines two rankings computed independently. A cross-encoder reads the question and one paper *together* and scores the pair, so nothing can be precomputed and every candidate costs a forward pass. We reorder only BGE's top ten and keep the same ten papers, so recall@10 cannot change. Which metrics can?
"""

from sentence_transformers import CrossEncoder

# 1. load the small cross-encoder that scores (question, paper) pairs
RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL_ID, device=DEVICE)

# 2. collect the ten BGE candidates per question (same set, so recall@10 is fixed)
bge_top_ids = {qid: [hit["corpus_id"] for hit in all_rankings["BGE"][qid]] for qid in query_ids}

# 3. score every (question, candidate) pair in one batched forward pass
pair_scores = reranker.predict(
    [(queries[qid], corpus[doc_id]) for qid in query_ids for doc_id in bge_top_ids[qid]],
    batch_size=64,
)

# 4. reorder each question's candidates by cross-encoder score (ties broken by id)
reranked_rankings = {}
for i, qid in enumerate(query_ids):
    scores = pair_scores[i * MAX_K : (i + 1) * MAX_K]
    order = sorted(
        zip(bge_top_ids[qid], scores), key=lambda pair: (-float(pair[1]), pair[0])
    )
    reranked_rankings[qid] = [{"corpus_id": doc_id, "score": float(score)} for doc_id, score in order]

print(f"reranked {len(reranked_rankings)} questions with {RERANK_MODEL_ID}")
