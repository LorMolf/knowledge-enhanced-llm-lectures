r"""
### Add distractors, then freeze the corpus

The corpus needs papers that are *not* labelled for anything, or every retriever looks
good: without them the top ten is all relevant by construction. Distractors are sampled
once from the unused documents and topped up to a floor, and the three assertions check
that the result covers every positive and has no empty or duplicated text.

"""

# every labelled document across the three splits, as strings
positive_ids = {
    str(doc_id)
    for split in (qa_train, qa_valid, qa_test)
    for row in split
    for doc_id in row["relevant_passage_ids"]
}

# distractors: never labelled, shuffled with the same seed, topped up to the floor
distractors = sorted(available - positive_ids)
random.Random(SEED).shuffle(distractors)
n_distractors = max(0, CORPUS_FLOOR - len(positive_ids))
selected_ids = positive_ids | set(distractors[:n_distractors])
corpus = {doc_id: text for doc_id, text in all_docs.items() if doc_id in selected_ids}

# guards: coverage, no empty text, no two papers with identical text
assert positive_ids <= set(corpus) and len(corpus) >= min(CORPUS_FLOOR, len(all_docs))
assert all(isinstance(text, str) and text.strip() for text in corpus.values()), "Empty document text"
normalized = {" ".join(text.split()).casefold() for text in corpus.values()}
assert len(normalized) == len(corpus), "Duplicate document text"
print("positive documents:", len(positive_ids), "| distractors:", n_distractors, "| corpus:", len(corpus))
