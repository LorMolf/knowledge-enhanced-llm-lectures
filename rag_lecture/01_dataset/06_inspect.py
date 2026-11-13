r"""
### Build the question and document maps

We key every question by its dataset ID and collect the relevant document IDs per split. A quick length check confirms no ID collapsed during dict construction. The next cell audits the flattened IDs and text.
"""

# every question must carry a non-empty ID before we key anything by it
assert all(
    r["id"] is not None and str(r["id"]).strip()
    for split in (qa_train, qa_valid, qa_test)
    for r in split
), "Empty question ID"

# build id -> question maps, one per split
train_queries = {str(r["id"]): r["question"] for r in qa_train}
val_queries = {str(r["id"]): r["question"] for r in qa_valid}
queries = {str(r["id"]): r["question"] for r in qa_test}

# build id -> relevant document ids, one per split
train_relevant_docs = {str(r["id"]): {str(i) for i in r["relevant_passage_ids"]} for r in qa_train}
val_relevant_docs = {str(r["id"]): {str(i) for i in r["relevant_passage_ids"]} for r in qa_valid}
relevant_docs = {str(r["id"]): {str(i) for i in r["relevant_passage_ids"]} for r in qa_test}

# no question ID may have collapsed during dict construction
assert (
    len(train_queries) == len(qa_train)
    and len(val_queries) == len(qa_valid)
    and len(queries) == len(qa_test)
)

# report the map sizes so the reader sees the split counts
print(
    f"train {len(train_queries)} | val {len(val_queries)} | test {len(queries)}"
)
