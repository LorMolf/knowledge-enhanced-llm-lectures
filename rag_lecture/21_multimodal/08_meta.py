r"""
### MetaEncoder v1

The check runs while section 14's `llama-server` still holds the GPU, so on a 14 GB
T4 it reports less free memory than MetaEncoder needs; that is the honest answer, not a bug.

[MetaEncoder v1](https://huggingface.co/facebook/meta-encoder-v1) is Meta's
embedding model for text, images and video: 28.4B parameters (57 GB in bfloat16)
and 6,656-dimensional vectors instead of 768. Its card says one 80 GB GPU holds it
in bfloat16. The vector depends on a free-text instruction such as "Find a
document image that matches the given query.": the same input embeds differently
under different instructions, and queries and documents usually take different
ones. The earlier gated checkpoint `facebook/meta-encoder` is described in the
[MetaEncoder paper](https://arxiv.org/abs/2610.11316). The v1 card cites no paper
and declares no license, so check its terms before using it outside a classroom.

The wrapper class ships inside the model repository, so this is remote code again,
as for Jina in section 08. We pin the revision, download that snapshot and read
`modeling_metaencoder.py` first: it builds the prompt, runs `AutoModel`, projects
the last token with the stored embedding head and L2-normalises. Its image
processor needs `torchvision`, which `install.sh` provides. The class also has
`retrieve` and `decide` methods; we only call `encode`, so the evaluation path
stays the same as for every other model.

The cell first frees the EmbeddingGemma weights and checks that enough GPU memory
is free, and only then downloads 57 GB. A `llama-server` from section 14 on the
same GPU counts against the free memory. When the check fails the cell prints why
and does not raise, so "Run all" reaches the last cell of the section, which still
plots the EmbeddingGemma rows. This pair and the next one are the optional
big-GPU tail.

> Does this machine have enough free GPU memory to hold MetaEncoder v1?
"""

# No GPU means no memory to measure; the flag below then reads False.
if not torch.cuda.is_available():
    print("no GPU visible: the big-encoder tail is skipped")

import gc
import sys

from huggingface_hub import snapshot_download

# 1. constants for this section: model id, pinned revision, memory floor
META_MODEL_ID = "facebook/meta-encoder-v1"
META_REVISION = "ed04ebd4defb8e837b5eaea164143177272696ac"
META_MIN_FREE_GB = 64  # 57 GB of bfloat16 weights plus room for activations

# 2. free the previous model so its weights do not count against the check
if "gemma_model" in globals():
    del gemma_model
gc.collect()
torch.cuda.empty_cache()

# 3. measure free GPU memory across all devices
free_gb = (sum(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())) / 1e9
           if torch.cuda.is_available() else 0.0)
META_AVAILABLE = free_gb >= META_MIN_FREE_GB

# 4. only download 57 GB if the memory check passed
if META_AVAILABLE:
    meta_path = snapshot_download(META_MODEL_ID, revision=META_REVISION)
    sys.path.insert(0, meta_path)
    from modeling_metaencoder import MetaEncoder

    meta_model = MetaEncoder.from_pretrained(meta_path)
    print(
        "Embedding size:",
        meta_model.embedding_dim,
        "| free GPU memory before loading (GB):",
        round(free_gb),
    )
else:
    # 5. explicit skip: print the reason, do not raise, keep the rest runnable
    print(
        f"MetaEncoder needs about {META_MIN_FREE_GB} GB of free GPU memory, found {free_gb:.0f} GB. "
        "Skipping the MetaEncoder cell; the last cell of section 21 still plots the "
        "EmbeddingGemma rows."
    )
