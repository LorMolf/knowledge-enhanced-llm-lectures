r"""
### Keep only the questions that can be trained on

The conservative choice: keep a question only when at least one of its positives is
labelled positive for no other question. That paper can then be a true positive for
its row and a true negative for every other row, so the in-batch loss never sees a
contradiction. It throws rows away, and the print says how many. A relevance-aware
sampler could keep more; this one is auditable.

`train_qids` and `selected_positive_ids` become the cohort for every run in sections
11 to 13, so the three methods are compared on exactly the same rows.
"""

# 1. one exclusive positive per question, chosen as the smallest ID for determinism
selected_positive_ids = {
    qid: min(exclusive)
    for qid in all_train_qids
    if (exclusive := {doc_id for doc_id in train_positive_ids[qid] if positive_counts[doc_id] == 1})
}
train_qids = [qid for qid in all_train_qids if qid in selected_positive_ids]

# 2. the batch size is a property of the cohort, so it lives with it
TRAIN_BATCH_SIZE = 8

# 3. guards: enough rows to fill a batch, and no known positive shared across rows
assert len(train_qids) >= TRAIN_BATCH_SIZE, (
    f"Need at least {TRAIN_BATCH_SIZE} exclusive-positive training questions, found {len(train_qids)}"
)
assert all(
    selected_positive_ids[other] not in train_positive_ids[qid]
    for qid in train_qids
    for other in train_qids
    if other != qid
), "Cross-row known-positive collision survived the filter"

print(
    "retained:",
    len(train_qids),
    "| excluded:",
    len(all_train_qids) - len(train_qids),
    "of",
    len(all_train_qids),
    "| batch size:",
    TRAIN_BATCH_SIZE,
)
