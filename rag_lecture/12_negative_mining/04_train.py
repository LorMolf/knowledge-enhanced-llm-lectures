r"""
### One run, written as a function

Every comparison in this section needs exactly the same training setup, so the
setup becomes a function and the differences live in its argument. A fresh BGE
checkpoint, the same step budget, the same batch size, the same seed, the same
query prompt: only `dataset` changes, and the dataset is what carries the mined
negative. The function returns the trainer (for its loss log) and the trained
model (for the next cell to score).
"""

def train_one(method, dataset):
    """Train BGE once on one negative-mining dataset and return trainer + model."""
    run_name = f"bge-{method}-negative"
    model = SentenceTransformer(BGE_MODEL_ID, device="cuda")
    model.max_seq_length = MAX_DOC_TOKENS

    # The same knobs for every method: nothing else may explain a difference.
    args = SentenceTransformerTrainingArguments(
        output_dir=f"models/{run_name}",
        max_steps=TRAIN_STEPS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=2e-5,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        dataloader_drop_last=True,
        seed=SEED,
        logging_steps=2,
        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        prompts={"query": BGE_QUERY_PROMPT},
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    wandb.finish()  # this cell opened the run, so it closes it
    return trainer, model


print("train_one defined: it trains one matched run and changes nothing else.")
