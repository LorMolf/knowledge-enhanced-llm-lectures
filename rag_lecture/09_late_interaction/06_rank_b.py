r"""
The top-3 ColBERT hits sit next to BGE's for question 0, each with its labelled-positive flag and a 400-character excerpt. Compare the first hit with BGE's: what evidence does each surface? The ColBERT model and its vectors are freed at the end so the next checkpoint starts clean.
"""

# 6. show the top-3 ColBERT vs BGE hits side by side for question 0
display(
    {
        "question": query_texts[0],
        "ColBERT": [
            {
                **hit,
                "labelled_positive": hit["corpus_id"] in relevant_docs[query_ids[0]],
                "excerpt": corpus[hit["corpus_id"]][:400],
            }
            for hit in colbert_rankings[query_ids[0]][:3]
        ],
        "BGE": [
            {
                **hit,
                "labelled_positive": hit["corpus_id"] in relevant_docs[query_ids[0]],
                "excerpt": corpus[hit["corpus_id"]][:400],
            }
            for hit in all_rankings["BGE"][query_ids[0]][:3]
        ],
    }
)

# 7. free the ColBERT model and its vectors before the next checkpoint
del colbert_model, colbert_queries, colbert_docs, similarities
print("freed ColBERT model and vectors")
