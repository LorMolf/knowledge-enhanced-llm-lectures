r"""
<a name="section-11"></a>

# Fine-tune BGE

### Audit the training labels before training on them

Two questions decide whether a matched training pair is even possible. First, how
many training positive documents are shared between questions: a paper that is
positive for two rows becomes a false negative for both when the loss treats the
other row's column as negative. Second, does any question share a positive with
another question's first positive, which is what a naive "take the first one"
sampler would pick.

This cell only counts and prints. The next cell filters on the count.
"""

from collections import Counter

# per-question positive sets, with a guard that they exist and live in the corpus
all_train_qids = list(train_queries)
train_positive_ids = {qid: set(train_relevant_docs[qid]) for qid in all_train_qids}
assert all(
    train_positive_ids[qid] and train_positive_ids[qid] <= corpus.keys() for qid in all_train_qids
), "Every training question needs at least one labelled paper inside the corpus"

# how often each paper is labelled positive, across all training questions
all_training_positive_ids = set().union(*train_positive_ids.values())
positive_counts = Counter(doc_id for ids in train_positive_ids.values() for doc_id in ids)

# the collision audit: what a naive sampler would get wrong
first_positives = {qid: sorted(train_positive_ids[qid])[0] for qid in all_train_qids}
collisions = [
    (qid, other, first_positives[other])
    for qid in all_train_qids
    for other in all_train_qids
    if other != qid and first_positives[other] in train_positive_ids[qid]
]

print("training questions:", len(all_train_qids))
print("distinct positive papers:", len(all_training_positive_ids))
print("papers shared by more than one question:", sum(1 for c in positive_counts.values() if c > 1))
print("naive-sampler collisions:", len(collisions), "| examples:", collisions[:5])
