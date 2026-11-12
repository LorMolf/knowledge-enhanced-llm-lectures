r"""
### Three runs, one loop

The loop is now four lines of work per method, which is the point: the experiment
is the loop over negative sources, not the training code. `loss_histories` and
`val_scores` keep one entry per method and feed the figure and the table after this
cell. Freeing the model between runs matters on a 16 GB T4: two BGE models plus two
corpus matrices do not fit.
"""

loss_histories, val_scores = {}, {}
for method, dataset in negative_datasets.items():
    print(f"--- {method}: {len(dataset)} training rows")

    # 1. train, then score validation + test
    trainer, model = train_one(method, dataset)
    val_metrics, test_metrics = score_one(method, model)

    # 2. keep what the comparison needs, and log the validation curve
    loss_histories[method] = trainer.state.log_history
    val_scores[method] = {"ndcg@10": val_metrics["ndcg@10"], "mrr@10": val_metrics["mrr@10"]}

    # 3. the row and the loss curve, side by side for this method
    display({"method": method, "validation": val_metrics, "test": test_metrics})
    show_training_comparison("BGE", f"BGE {method} negative")

    # 4. free the GPU before the next method
    del trainer, model
    torch.cuda.empty_cache()

print("validation scores by method:", val_scores)
