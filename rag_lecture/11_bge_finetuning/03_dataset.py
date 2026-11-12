r"""
### The dataset the trainer receives

Two text columns and nothing else: the question, and the one paper the loss should
score highest. The in-batch negatives are implicit, which is why the ID appears in
the printout: if you disagree with this pairing for the first row, the whole run is
suspect, and the labelled set is printed next to it so you can check.
"""

from datasets import Dataset

# 1. materialise the (query, positive) pairs in cohort order
train_pairs = Dataset.from_dict(
    {
        "query": [train_queries[qid] for qid in train_qids],
        "positive": [corpus[selected_positive_ids[qid]] for qid in train_qids],
    }
)

# 2. show the first row together with every labelled positive of that question
display(
    train_pairs[0],
    {"qid": train_qids[0], "all_known_positive_ids": sorted(train_positive_ids[train_qids[0]])},
)
print("training rows:", len(train_pairs), "| columns:", train_pairs.column_names)
