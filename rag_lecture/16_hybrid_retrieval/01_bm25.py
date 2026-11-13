r"""
<a name="section-16"></a>

# Lexical, hybrid and reranked retrieval

BM25 rewards matching terms while accounting for term frequency and document
length. Unlike dense embeddings, this simple baseline does not understand
synonyms. We lowercase and split into alphanumeric tokens, including digits;
this deliberately simple tokenizer may split biomedical punctuation.
[rank_bm25](https://github.com/dorianbrown/rank_bm25). We reuse section 12's
`lexical_tokens` and `lexical_index`. Equal scores use ascending original paper
ID, including ties at the cutoff. What does the first hit actually discuss?
"""

# Score every query with BM25 and keep the top MAX_K hits per query.
bm25_rankings = {}
for qid, question in queries.items():
    # score each corpus doc against the tokenized question
    scores = lexical_index.get_scores(lexical_tokens(question))
    # rank by descending score, breaking ties by ascending paper ID
    order = sorted(range(len(corpus_ids)), key=lambda i: (-float(scores[i]), corpus_ids[i]))[:MAX_K]
    # store the ranked hits with their scores
    bm25_rankings[qid] = [{"corpus_id": corpus_ids[i], "score": float(scores[i])} for i in order]

# Inspect the top hit for the example query and check whether it is labelled relevant.
first_bm25_id = bm25_rankings[example_qid][0]["corpus_id"]
display(
    {
        "question": example_question,
        "hits": bm25_rankings[example_qid],
        "first_paper_excerpt": corpus[first_bm25_id][:600],
        "first_is_labelled_relevant": first_bm25_id in relevant_docs[example_qid],
    }
)

# Report retrieval metrics for the BM25 run.
evaluate_run("BM25", bm25_rankings)
