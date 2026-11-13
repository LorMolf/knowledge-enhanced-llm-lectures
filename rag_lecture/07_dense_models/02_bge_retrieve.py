r"""
### Encode and retrieve

Encode documents once and questions with the published instruction. The
shared ranker maps vector positions back to real document IDs; the evaluator
uses the same held-out questions and partial relevance labels as before. What
does the first retrieved paper discuss, and is it a labelled positive?
"""

# encode the corpus once, normalized so dot product is cosine similarity
bge_corpus_embeddings = bge_model.encode(
    corpus_texts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True
).cpu()

# encode the questions with the published BGE query instruction
bge_query_embeddings = bge_model.encode(
    query_texts,
    prompt=BGE_QUERY_PROMPT,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).cpu()

# rank with the shared dense ranker, mapping positions back to document IDs
bge_rankings = rank_dense(bge_query_embeddings, bge_corpus_embeddings, query_ids)

# score against the same held-out questions and partial labels as before
bge_metrics = evaluate_run("BGE", bge_rankings)

# inspect the top hit for the first question
first_bge_hit = bge_rankings[query_ids[0]][0]
display(
    {
        "question": query_texts[0],
        "hit": first_bge_hit,
        "labelled_positive": first_bge_hit["corpus_id"] in relevant_docs[query_ids[0]],
        "document": corpus[first_bge_hit["corpus_id"]],
        "metrics": bge_metrics,
    }
)
