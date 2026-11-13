r"""
### Index the documents, then keep the questions that can be answered

Two checks before any split: every document has a unique non-empty ID, and every
question whose labelled passages are missing from the corpus is dropped rather than
scored as a miss. The shuffle is seeded, so the split below is the same on every run.

"""

# one ID -> text map, with the two guards that make the rest safe
assert all(row["id"] is not None and str(row["id"]).strip() for row in doc_source), "Empty document ID"
all_docs = {str(row["id"]): row["passage"] for row in doc_source}
assert len(all_docs) == len(doc_source), "Duplicate document IDs"
available = set(all_docs)

# a question is eligible only when every labelled passage exists, then shuffle
def usable(row):
    ids = row["relevant_passage_ids"]
    return bool(ids) and all(str(doc_id) in available for doc_id in ids)

eligible_train = qa_source["train"].filter(usable).shuffle(seed=SEED)
eligible_test = qa_source["test"].filter(usable).shuffle(seed=SEED)
assert len(eligible_train) >= N_TRAIN + N_VALID and len(eligible_test) >= N_TEST

# carve the frozen splits out of the shuffled pools
qa_train = eligible_train.select(range(N_TRAIN))
qa_valid = eligible_train.select(range(N_TRAIN, N_TRAIN + N_VALID))
qa_test = eligible_test.select(range(N_TEST))
print("documents:", len(all_docs), "| eligible:", len(eligible_train), "| kept:", len(qa_train), len(qa_valid), len(qa_test))
