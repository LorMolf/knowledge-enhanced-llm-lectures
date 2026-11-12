r"""
### Audit IDs, text and labels

We flatten all splits to check unique IDs, non-empty text, and no duplicate normalized question text across splits. Every labelled document must exist in the corpus. The final display shows one concrete test question and its first relevant passage.
"""

# 1. flatten all splits to audit IDs and text together
question_splits = (train_queries, val_queries, queries)
all_question_ids = [qid for split in question_splits for qid in split]
all_question_texts = [text for split in question_splits for text in split.values()]

# 2. audit: unique IDs, non-empty text, no duplicate normalized text across splits
assert len(all_question_ids) == len(set(all_question_ids)), "Question ID repeated across splits"
assert all(isinstance(text, str) and text.strip() for text in all_question_texts), "Empty question"
assert len({" ".join(text.split()).casefold() for text in all_question_texts}) == len(
    all_question_texts
), "Duplicate question text"

# 3. every labelled document must exist in the corpus
assert all(
    ids <= set(corpus) and ids
    for rel in (train_relevant_docs, val_relevant_docs, relevant_docs)
    for ids in rel.values()
)

# 4. show one concrete question -> document link from the test split
corpus_ids, corpus_texts = list(corpus), list(corpus.values())
query_ids, query_texts = list(queries), list(queries.values())
example_id = query_ids[0]
example_doc = sorted(relevant_docs[example_id])[0]
display(
    {
        "question_id": example_id,
        "question": queries[example_id],
        "document_id": example_doc,
        "document": corpus[example_doc],
    }
)
