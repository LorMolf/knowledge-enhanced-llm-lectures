r"""
### Remove role prefixes, keep the retrieval task

Now keep the retrieval adapter fixed and explicitly request an empty prompt for both sides. This is a separate ablation from changing the adapter. The next cell encodes the corpus and queries with no role prefix.
"""

# Encode both sides with the retrieval adapter but an empty prompt, isolating the prefix effect
jina_no_prefix_docs = jina_model.encode(
    corpus_texts,
    task="retrieval",
    prompt="",
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).cpu()

# Use the same empty prompt on the query side so both sides share one setting
jina_no_prefix_queries = jina_model.encode(
    query_texts,
    task="retrieval",
    prompt="",
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).cpu()

print("encoded", len(jina_no_prefix_docs), "docs and", len(jina_no_prefix_queries), "queries with no prefix")
