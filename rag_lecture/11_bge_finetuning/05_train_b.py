r"""
### The trainer, then the forty steps

`BATCH_SAMPLER` is what makes the loss work: every row contributes its positive and the
rest of the batch as negatives, so batch size *is* the number of negatives. `drop_last`
keeps that count constant. `logging_steps=2` gives the loss curve twenty points instead
of four.

"""

from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

# the same budget for every method compared in this section
bge_args = SentenceTransformerTrainingArguments(
    output_dir="models/bge-in-batch",
    max_steps=TRAIN_STEPS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    batch_sampler=BatchSamplers.BATCH_SAMPLER,
    dataloader_drop_last=True,
    seed=SEED,
    logging_steps=2,
    save_strategy="no",
    report_to="wandb",
    run_name="bge-in-batch",
    prompts={"query": BGE_QUERY_PROMPT},
)

# in-batch negatives, then the run
bge_trainer = SentenceTransformerTrainer(
    model=bge_train_model,
    args=bge_args,
    train_dataset=train_pairs,
    loss=MultipleNegativesRankingLoss(bge_train_model),
)
bge_trainer.train()
losses = [row for row in bge_trainer.state.log_history if "loss" in row]
print("steps logged:", len(losses), "| last loss:", round(losses[-1]["loss"], 4) if losses else "n/a")
