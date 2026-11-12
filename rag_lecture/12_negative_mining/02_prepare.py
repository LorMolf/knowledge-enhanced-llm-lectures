r"""
### Select and inspect negatives

For each retained question, sample one random candidate or take the best BM25
or BGE candidate from the **same globally filtered pool**. Which candidate might
still be an unjudged positive? The native
[`mine_hard_negatives`](https://sbert.net/docs/package_reference/util/hard_negatives.html)
adds supplied positives to its text corpus and can regroup examples by query;
our explicit ID-based selection keeps this matched cohort and its global filter
visible without an adapter.
"""

# 1. pick one negative per question under each strategy
negative_ids = {"random": [], "bm25": [], "dense": []}
for i, qid in enumerate(train_qids):
    bm25_scores = lexical_index.get_scores(lexical_tokens(train_queries[qid]))
    choices = {
        "random": rng.choice(eligible),
        "bm25": min(eligible, key=lambda j: (-float(bm25_scores[j]), corpus_ids[j])),
        "dense": min(eligible, key=lambda j: (-float(dense_mining_scores[i, j]), corpus_ids[j])),
    }
    for method, index in choices.items():
        negative_ids[method].append(corpus_ids[index])

# 2. guard: no mined negative may be a known training positive
assert all(
    len(ids) == len(train_qids) and set(ids).isdisjoint(all_training_positive_ids)
    for ids in negative_ids.values()
), "A mined negative is a known training positive."

# 3. attach the chosen negatives as a column on the training pairs
negative_datasets = {
    method: train_pairs.add_column("negative", [corpus[doc_id] for doc_id in ids])
    for method, ids in negative_ids.items()
}

# 4. show one question with its positive and each strategy's negative
display(
    {
        "question": train_queries[train_qids[0]],
        "positive": corpus[selected_positive_ids[train_qids[0]]],
        "unique_negatives_per_source": {
            method: len(set(ids)) for method, ids in negative_ids.items()
        },
        "negatives": {
            method: {"id": ids[0], "text": corpus[ids[0]]} for method, ids in negative_ids.items()
        },
    }
)
