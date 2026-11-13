r"""
### CLS versus mean, same weights

Re-encode the same questions and documents with the native masked mean
pooling module and rank by cosine. Only pooling changes: encoder weights,
token budget and candidates are the same. Which metrics move, and which
papers appear in both top tens? No improvement is assumed.
"""

# re-encode queries and corpus with mean pooling (only pooling changes)
mean_module = pool_modules["mean"]
mean_query_embeddings = encode_biomed(query_texts, naive_model, naive_tokenizer, mean_module)
mean_corpus_embeddings = encode_biomed(corpus_texts, naive_model, naive_tokenizer, mean_module)

# rank by cosine and score the run, appending to the shared evaluation rows
mean_rankings = rank_dense(mean_query_embeddings, mean_corpus_embeddings)
mean_metrics = evaluate_run("PubMedBERT mean", mean_rankings)

# show CLS vs mean side by side on the same metrics
display(
    pd.DataFrame(evaluation_rows)
    .set_index("model")
    .loc[:, ["accuracy@1", "recall@10", "mrr@10", "ndcg@10"]]
)

# how many of the first query's top-10 papers survive the pooling swap
print(
    "first query top-10 overlap:",
    len(
        {h["corpus_id"] for h in naive_rankings[query_ids[0]]}
        & {h["corpus_id"] for h in mean_rankings[query_ids[0]]}
    ),
)

# encode the semantic probe set with mean pooling for the next comparison
semantic_mean = encode_biomed(semantic_texts, naive_model, naive_tokenizer, mean_module)

# free the previous model and scratch tensors before the next section
del naive_model, naive_tokenizer, sample_output, sample_hidden, sample_inputs, sample_pooled
del pool_modules, mean_module, normalize_module, pooled, normalized, pooler_output
del sample_docs, unit_docs, raw_scores, unit_scores
