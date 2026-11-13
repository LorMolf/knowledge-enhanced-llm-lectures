r"""
### One grid: where the first labelled paper sits

Each cell is the rank of the first labelled paper for that question and that run, and
blank means no labelled paper made the top ten. Columns are sorted by nDCG@10 so the
best run is on the left, and rows are sorted by how many runs missed the question, so
the failures gather at the top instead of hiding in a wall of numbers.

"""

# rank of the first labelled hit per (question, run); NaN when none made the top-K
def first_hit_rank(run, qid):
    for rank, hit in enumerate(run[qid], 1):
        if hit["corpus_id"] in relevant_docs[qid]:
            return rank
    return np.nan


first_rank = pd.DataFrame(
    {name: [first_hit_rank(run, qid) for qid in query_ids] for name, run in all_rankings.items()},
    index=query_ids,
)

# best run on the left, questions every run misses at the top
run_order = comparison.set_index("model")["ndcg@10"].sort_values(ascending=False)
first_rank = first_rank[[name for name in run_order.index if name in first_rank.columns]]
first_rank = first_rank.loc[first_rank.isna().sum(axis=1).sort_values().index]
print("grid:", first_rank.shape, "| runs:", list(first_rank.columns))
