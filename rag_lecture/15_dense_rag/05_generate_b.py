r"""
### The loop, and the cache

Each question is asked three times with the same system prompt, so a difference between
conditions is the evidence and nothing else. Whether retrieval found a labelled paper is
a property of the question, not of the answer, so it is attached to all three rows here.
Writing the frame to parquet is what makes a kernel restart cheap.

"""

# 1. one row per (question, condition), 96 short calls
if not RUN_LLM and RAG_CACHE.exists():
    rag_frame = pd.read_parquet(RAG_CACHE)
    print("reused", len(rag_frame), "cached answers from", RAG_CACHE)
else:
    rag_rows = []
    for number, qid in enumerate(query_ids, 1):
        top_three = [
            {"id": hit["corpus_id"], "text": corpus[hit["corpus_id"]]}
            for hit in all_rankings["BGE"][qid][:3]
        ]
        labelled = [{"id": doc_id, "text": corpus[doc_id]} for doc_id in sorted(relevant_docs[qid])[:3]]
        found_labelled = any(entry["id"] in relevant_docs[qid] for entry in top_three)

        for condition, evidence in zip(CONDITIONS, ([], top_three, labelled)):
            row = answer_row(qid, condition, evidence)
            row["top-3 holds a labelled paper"] = found_labelled
            rag_rows.append(row)
        print(f"{number}/{len(query_ids)} {qid} done")

    # 2. write the cache so a kernel restart does not pay for this twice
    rag_frame = pd.DataFrame(rag_rows)
    RAG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    rag_frame.to_parquet(RAG_CACHE)
    print("generated", len(rag_frame), "answers; cached to", RAG_CACHE)
