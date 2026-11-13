r"""
<a name="section-13"></a>

# Fine-tune ColBERT

### Two things a ColBERT run needs that BGE did not

The checkpoint, and its role prefixes. `colbertv2.0` ships a query prefix and a document
prefix, and training with them mismatched is the quiet way to get a model that scores
worse than the one it started from. The model is loaded here, once, so the next cell can
show the prefixes before training touches anything.

"""

from sentence_transformers import MultiVectorEncoder

# the checkpoint, on the GPU, with whatever prompts it carries
colbert_model = MultiVectorEncoder("colbert-ir/colbertv2.0", device=DEVICE)
print("role prompts:", list(colbert_model.prompts))
