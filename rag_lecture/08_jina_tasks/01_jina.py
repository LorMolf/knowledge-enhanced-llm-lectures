r"""
<a name="section-08"></a>

# 08. Jina task and role ablations

[Jina embeddings v5](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)
exposes two separate choices: `task` selects a task-specific adapter, while
`prompt_name` identifies a query or a document. These are not interchangeable.
Retrieval is asymmetric: a short question seeks a longer relevant paper. Start
with the model card's retrieval configuration on both sides. The model card
lists CC BY-NC 4.0, a non-commercial license.

This checkpoint uses custom remote code. The pinned loader does not forward its
revision to every weight/adapter download, so first resolve a local snapshot of
that exact revision. Review its source before executing it. Its last-token pooling
assumes right padding. We use `encode` explicitly rather than `encode_query`,
which can supply an incompatible task to this model's custom implementation.
What task and role does each input receive?
"""

# 1. pin the checkpoint and the exact revision we reviewed
JINA_MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
JINA_REVISION = "dd76d535f5447ca3897a9c893fb1e612ead98192"
from huggingface_hub import snapshot_download

# 2. resolve a local snapshot so every weight/adapter file comes from that revision
jina_path = snapshot_download(
    JINA_MODEL_ID,
    revision=JINA_REVISION,
    allow_patterns=["*.py", "*.json", "*.safetensors", "*.txt", "*.model"],
)

# 3. load with remote code; last-token pooling needs right padding
jina_model = SentenceTransformer(jina_path, trust_remote_code=True, device=DEVICE)
jina_model.tokenizer.padding_side = "right"
jina_model.max_seq_length = MAX_DOC_TOKENS

# 4. show the registered prompts and the two input roles we will encode
print("Prompts:", jina_model.prompts)
print("Question:", query_texts[0])
print("Document:", corpus_texts[0][:400])
