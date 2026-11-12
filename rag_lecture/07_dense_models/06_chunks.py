r"""
### Do we retrieve a passage or its sentences?

Everything so far treats a passage as the unit to encode. Real RAG has to choose
that unit: a long passage diluted by one relevant sentence, or a short chunk that
loses the context the question needs. We split the same papers into three-sentence
and one-sentence chunks, score every chunk with BGE, keep each paper's best chunk,
and rank papers as before. Only the unit changes: questions, weights, candidates
and evaluator stay the same, and a paper's score is the maximum over its chunks.
Does the smaller unit find the sentence, and what does it cost at rank one?
"""

import re

# 1. BGE was parked on the CPU in section 03; the GPU is free again after the Qwen cells.
bge_model.to(DEVICE)

# 2. Split every paper into 3-sentence and 1-sentence chunks, tagged by (window, doc_id).
chunk_docs, chunk_texts = [], []
for doc_id, text in corpus.items():
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for window in (3, 1):
        for start in range(0, len(sentences), window):
            chunk_docs.append((window, doc_id))
            chunk_texts.append(" ".join(sentences[start : start + window]))

# 3. Encode every chunk once with BGE.
chunk_vectors = bge_model.encode(
    chunk_texts, batch_size=128, convert_to_tensor=True
).cpu()

# 4. Score each query against every chunk.
chunk_scores = util.cos_sim(bge_query_embeddings.cpu(), chunk_vectors)

# 5. For each chunk size, collapse chunks to papers by taking the max chunk score.
for window in (3, 1):
    positions = [i for i, (size, _) in enumerate(chunk_docs) if size == window]
    best = chunk_scores[:, positions]
    rankings = {}
    for i, qid in enumerate(query_ids):
        per_document = {}
        for j, chunk_index in enumerate(positions):
            doc_id = chunk_docs[chunk_index][1]
            per_document[doc_id] = max(per_document.get(doc_id, float("-inf")), float(best[i, j]))
        ordered = sorted(per_document.items(), key=lambda item: (-item[1], item[0]))[:MAX_K]
        rankings[qid] = [{"corpus_id": doc_id, "score": score} for doc_id, score in ordered]
    # 6. Score this chunk size with the same evaluator as the passage runs.
    evaluate_run(f"BGE {window}-sentence chunks", rankings)

# 7. Compare chunk sizes against whichever passage baselines already ran.
frame = pd.DataFrame(evaluation_rows).set_index("model")
wanted = [name for name in ("BGE", "BGE in-batch", "BGE 3-sentence chunks", "BGE 1-sentence chunks") if name in frame.index]
display(frame.reindex(wanted)[["accuracy@1", "recall@10", "mrr@10", "ndcg@10"]])
