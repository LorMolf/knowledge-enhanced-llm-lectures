r"""
### A second late-interaction checkpoint

[Reason-ModernColBERT](https://huggingface.co/lightonai/Reason-ModernColBERT)
uses a different backbone and training recipe. Load it only after releasing the
first model. Its own role prompts and length settings are applied by the same
native API; both its representations and its ranking are recomputed over the
same question and document collections. How do the first rankings differ?
The model card lists a non-commercial license.
"""

# 1. load the second checkpoint; the first model was already freed above
reason_model = MultiVectorEncoder("lightonai/Reason-ModernColBERT", device=DEVICE)
print(reason_model)
print("Role prompts:", reason_model.prompts)

# 2. encode the same queries and corpus with this model's own prompts
reason_queries = reason_model.encode_query(query_texts, show_progress_bar=True)
reason_docs = reason_model.encode_document(
    corpus_texts, convert_to_numpy=True, show_progress_bar=True
)
print("First token-vector shapes:", reason_queries[0].shape, reason_docs[0].shape)

# 3. score and rank with the shared late-interaction helper
reason_rankings = rank_multivector(reason_model, reason_queries, reason_docs)

# 4. show metrics and the top-3 side by side with the first model
display(
    {
        "metrics": evaluate_run("Reason-ModernColBERT", reason_rankings),
        "question": query_texts[0],
        "Reason": [
            {**hit, "excerpt": corpus[hit["corpus_id"]][:400]}
            for hit in reason_rankings[query_ids[0]][:3]
        ],
        "ColBERT": [
            {**hit, "excerpt": corpus[hit["corpus_id"]][:400]}
            for hit in all_rankings["ColBERT"][query_ids[0]][:3]
        ],
    }
)

# 5. release this model before the next section
del reason_model, reason_queries, reason_docs
