r"""
### Re-index and score with MaxSim

Re-encode every paper with the fine-tuned model, keep the token index on CPU, and
reuse section 09's bounded native scoring helper. Validation is logged to W&B and
the test ranking enters the shared ledger, so the row lands next to section 09's
ColBERT. Which metric and which paper changed, and does the changed paper's text
support the question? The whole cell is skipped when the training run was skipped.
"""

if CAN_TRAIN:
    # 1. re-encode the corpus with the fine-tuned model (CPU token index)
    colbert_docs = colbert_train_model.encode_document(
        corpus_texts, batch_size=8, convert_to_numpy=True, show_progress_bar=True
    )

    # 2. rank the validation questions against it with MaxSim
    colbert_val = colbert_train_model.encode_query(list(val_queries.values()), batch_size=8)
    val_hits = rank_multivector(
        colbert_train_model, colbert_val, colbert_docs, qids=list(val_queries)
    )
    val_metrics = validation_evaluator.compute_metrics(
        [val_hits[qid] for qid in validation_evaluator.queries_ids]
    )

    # 3. the same index, now for the test questions, into the shared ledger
    colbert_test = colbert_train_model.encode_query(query_texts, batch_size=8)
    test_hits = rank_multivector(colbert_train_model, colbert_test, colbert_docs, qids=query_ids)
    display(val_metrics, evaluate_run("ColBERT in-batch", test_hits))
    show_training_comparison("ColBERT", "ColBERT in-batch")

    # 4. free the fine-tuned model
    del colbert_trainer, colbert_train_model, colbert_docs, colbert_val, colbert_test
else:
    print("ColBERT fine-tuning was skipped, so there is no new row to score.")
