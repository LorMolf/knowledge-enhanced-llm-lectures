r"""
### Set up the run

Four decisions before a single gradient step, and each one is a number you can change:
the step budget, the batch size that decides how many in-batch negatives exist, the
learning rate, and the seed. The previous BGE goes to CPU first, because a fresh copy on
the GPU next to the old one does not fit a T4.

"""

import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.training_args import BatchSamplers

# 1. W&B offline by default: the notebook must run without an account
os.environ["WANDB_PROJECT"] = "rag-lecture-retrievers"
os.environ.setdefault("WANDB_MODE", "offline")
os.environ["WANDB_CONSOLE"] = "off"
TRAIN_STEPS = 40
LEARNING_RATE = 2e-5

# 2. free the previous model, then require CUDA for training
bge_model.to("cpu")
assert torch.cuda.is_available(), "Run fine-tuning on a CUDA machine."

# 3. a fresh BGE from the original checkpoint, capped at the document budget
bge_train_model = SentenceTransformer(BGE_MODEL_ID, device="cuda")
bge_train_model.max_seq_length = MAX_DOC_TOKENS
print("trainable:", BGE_MODEL_ID, "| steps:", TRAIN_STEPS, "| batch:", TRAIN_BATCH_SIZE)
