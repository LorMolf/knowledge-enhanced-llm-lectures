r"""
Retrieval mode encodes the closed corpus with the document role and every test question with the query role. This cell produces both embedding matrices and reports their shapes so we can confirm one vector per input.
"""

# encode the corpus with the document role
jina_retrieval_docs = jina_model.encode(
    corpus_texts,
    task="retrieval",
    prompt_name="document",
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).cpu()

# encode the questions with the query role
jina_retrieval_queries = jina_model.encode(
    query_texts,
    task="retrieval",
    prompt_name="query",
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).cpu()

# report the shapes so we can check one vector per input
print("queries", tuple(jina_retrieval_queries.shape), "| docs", tuple(jina_retrieval_docs.shape))
