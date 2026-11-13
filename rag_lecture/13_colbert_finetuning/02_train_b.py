r"""
### The training function

Same budget as every other run in this lecture, same batch sampler, and the two prompt
roles passed explicitly so the loss sees query/document pairs the way the checkpoint
expects. The loss is MaxSim inside the in-batch contrastive objective: each query's
positive is its own paper, every other paper in the batch is a negative.

"""

from sentence_transformers import MultiVectorEncoderTrainer, MultiVectorEncoderTrainingArguments
from sentence_transformers.multi_vector_encoder.losses import MultiVectorMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers


def train_colbert(model=colbert_model):
    """Fine-tune ColBERTv2 on the section-11 pairs and return the trainer."""
    # the same budget as every other training run in this lecture
    args = MultiVectorEncoderTrainingArguments(
        output_dir="models/colbert-in-batch",
        max_steps=TRAIN_STEPS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=2e-5,
        batch_sampler=BatchSamplers.BATCH_SAMPLER,
        dataloader_drop_last=True,
        seed=SEED,
        logging_steps=2,
        save_strategy="no",
        report_to="wandb",
        run_name="colbert-in-batch",
        prompts={"query": model.prompts["query"], "positive": model.prompts["document"]},
    )

    # MaxSim inside the in-batch contrastive loss, then the run
    trainer = MultiVectorEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_pairs,
        loss=MultiVectorMultipleNegativesRankingLoss(model),
    )
    trainer.train()
    return trainer, model


print("train_colbert defined; the next cell runs it only if a GPU is there.")
