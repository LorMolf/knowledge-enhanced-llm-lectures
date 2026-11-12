r"""
### Setup: seeds, interpreter and the versions that loaded

This cell sets every seed the lecture relies on and prints the versions that are
actually loaded, next to the versions the notebook was checked against. If the
right-hand column of the last line is not empty, the kernel is still the old one:
run `Runtime → Restart session`, then run this cell again. Every later comparison
assumes the same interpreter, the same seeds and the same weights.

**Question:** does the loaded version of each package match the one the notebook
was checked against?
"""

import os

# Training cells report to W&B; offline mode needs no account and keeps a local run.
os.environ.setdefault("WANDB_MODE", "offline")
import random
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

# 1. One seed for every random source the notebook uses.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

# 2. What is running: interpreter, CUDA build, and the GPU we actually got.
print("Python:", sys.executable)
print("CUDA build of torch:", torch.version.cuda, "| GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 3. What was asked for, compared with what was loaded.
EXPECTED = {"sentence-transformers": "6.2.0", "transformers": "5.22.2", "datasets": "4.8.5"}
for package, wanted in EXPECTED.items():
    try:
        print(f"{package}: loaded {version(package)} | expected {wanted}")
    except PackageNotFoundError:
        print(f"{package}: MISSING — run the install cell, then restart the runtime")
