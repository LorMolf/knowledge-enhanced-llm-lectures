r"""
With both embedding matrices in hand, we rank the corpus for every question and score the run. The displayed hits show which top result is chosen for the first question and whether it is a labelled positive.
"""

# rank the corpus for every query
jina_retrieval_rankings = rank_dense(jina_retrieval_queries, jina_retrieval_docs, query_ids)

# show metrics and the top hits for the first question
display(
    {
        "metrics": evaluate_run("Jina retrieval", jina_retrieval_rankings),
        "question": query_texts[0],
        "hits": [
            {
                **hit,
                "labelled_positive": hit["corpus_id"] in relevant_docs[query_ids[0]],
                "excerpt": corpus[hit["corpus_id"]][:400],
            }
            for hit in jina_retrieval_rankings[query_ids[0]][:3]
        ],
    }
)
