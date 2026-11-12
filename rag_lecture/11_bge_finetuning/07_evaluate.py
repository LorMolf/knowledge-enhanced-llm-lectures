r"""
### Rebuild the dense index

After an update, old document vectors no longer represent the same model. Let's
encode the *whole* common corpus again, then compare validation and held-out
test retrieval. The held-out test is descriptive, not a model-selection signal.
Which metric moved, and by how much? The comparison printed below names the papers
that changed place, and the next cell shows how the score distributions shifted.
"""

# The trainer opens the W&B run; section 12 closes it after each method.
import wandb


# 1. re-encode the whole corpus with the fine-tuned model (old vectors are stale)
bge_ft_docs = bge_train_model.encode(corpus_texts, convert_to_tensor=True, device="cuda").cpu()

# 2. encode validation queries with the query prompt
bge_ft_val = bge_train_model.encode(
    list(val_queries.values()), prompt=BGE_QUERY_PROMPT, convert_to_tensor=True, device="cuda"
).cpu()

# 3. rank validation queries against the fresh document index
val_hits = rank_dense(bge_ft_val, bge_ft_docs, qids=list(val_queries))

# 4. score validation retrieval
val_metrics = validation_evaluator.compute_metrics(
    [val_hits[qid] for qid in validation_evaluator.queries_ids]
)

# 5. encode held-out test queries (descriptive only, not for model selection)
bge_ft_test = bge_train_model.encode(
    query_texts, prompt=BGE_QUERY_PROMPT, convert_to_tensor=True, device="cuda"
).cpu()

# 6. show validation metrics next to held-out test retrieval
before = {row["model"]: row for row in evaluation_rows}["BGE"]
after = evaluate_run("BGE in-batch", rank_dense(bge_ft_test, bge_ft_docs, qids=query_ids))
display(val_metrics, after)
print("change on the held-out test:", {k: round(after[k] - before[k], 3)
                                       for k in ("accuracy@1", "recall@10", "mrr@10", "ndcg@10")})

# 7. compare fine-tuned BGE against untouched BGE
show_training_comparison("BGE", "BGE in-batch")
