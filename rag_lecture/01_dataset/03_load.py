r"""
### The dataset

A retriever maps a question to documents; the answer is reserved for later
qualitative discussion. We use [mini BioASQ](https://huggingface.co/datasets/enelpol/rag-mini-bioasq),
whose QA train/test splits are distinct from the document collection's `test`
split. Validation comes only from QA training. Let's inspect the available
records before sampling.

The dataset card lists CC BY 2.5: credit the dataset and BioASQ when reusing
its question/answer annotations. Documents are paper texts collected from
PubMed; check the rights of each underlying source separately rather than
assuming the dataset label covers every document.

**Question:** what splits and how many records does each source expose?
"""

# Load the two sources once; every later cell reuses these handles.
from datasets import load_dataset
from IPython.display import display
import random
import torch

# Section constants: retrieval depth, token budget, device, split sizes.
KS = [1, 3, 5, 10]
MAX_K = 10
MAX_DOC_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TRAIN, N_VALID, N_TEST = 128, 32, 32
CORPUS_FLOOR = 2000
DATASET_ID = "enelpol/rag-mini-bioasq"
DATASET_REVISION = "8a845907dc1cff31d42fa6f7bb9c6eef5f3ae6f6"

# Pull QA pairs (train/test) and the document corpus (test split only).
qa_source = load_dataset(DATASET_ID, "question-answer-passages", revision=DATASET_REVISION)
doc_source = load_dataset(DATASET_ID, "text-corpus", split="test", revision=DATASET_REVISION)

# Show what we got so the split sizes are visible before sampling.
print("splits:", {name: len(split) for name, split in qa_source.items()})
print("train:", len(qa_source["train"]), "| test:", len(qa_source["test"]),
      "| documents:", len(doc_source), "| device:", DEVICE)
