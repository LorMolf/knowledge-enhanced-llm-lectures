r"""
### The questions nobody retrieves

Five rows where every run missed. Read the labelled paper next to what BGE put first:
either the question needs vocabulary the paper does not use, or the labelled paper
answers a different question than the one that was asked. The second case is a label
problem, and no retriever can fix it.

"""

# For the misses, show the question, the labelled paper, and BGE's first choice.
missed_qids = first_rank.index[first_rank.isna().all(axis=1)][:5]
missed_rows = [
    {
        "question": queries[qid],
        "first labelled paper": corpus[sorted(relevant_docs[qid])[0]][:300],
        "BGE rank-1 paper": corpus[all_rankings["BGE"][qid][0]["corpus_id"]][:300],
    }
    for qid in missed_qids
]
display(pd.DataFrame(missed_rows))
print("questions every run misses:", len(first_rank.index[first_rank.isna().all(axis=1)]))
