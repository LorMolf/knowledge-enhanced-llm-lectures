r"""
<a name="section-09"></a>

# ColBERT and Reason-ModernColBERT

A single dense vector must represent an entire document. ColBERT instead keeps
one contextual vector per token. For each query token, find the most similar
document token, then sum those maxima:

$$\operatorname{MaxSim}(q,d)=\sum_i \max_j q_i^\top d_j.$$

![Late interaction and MaxSim](assets/late-interaction.png)

*Figure: ColBERT's late interaction and MaxSim, from [Khattab & Zaharia, 2020](https://arxiv.org/abs/2004.12832).*

The token vectors can be stored before a question arrives, but take more space
than a single vector per document. Let's load the original
[ColBERTv2 checkpoint](https://huggingface.co/colbert-ir/colbertv2.0) through
[Sentence Transformers' native multi-vector API](https://sbert.net/docs/multi_vector_encoder/usage/usage.html).
The native loader supports legacy checkpoint formats. Inspect its resulting
prompts and modules before comparing token-vector shapes.

**Question:** what role prompts and modules does the native loader attach to this legacy checkpoint?
"""

# Load the original ColBERTv2 checkpoint through the native multi-vector API.
from sentence_transformers import MultiVectorEncoder

# build the encoder on the shared device (legacy checkpoint format is supported)
colbert_model = MultiVectorEncoder("colbert-ir/colbertv2.0", device=DEVICE)

# inspect what the loader attached: the module tree and the role prompts
print(colbert_model)
print("Role prompts:", colbert_model.prompts)
