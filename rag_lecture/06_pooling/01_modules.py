r"""
<a name="section-06"></a>

# Pooling and normalization

We encode two queries once and reuse the same hidden states for every pooling mode. This cell builds one Pooling module per mode and applies each to a copy of the token vectors, so the comparison is fair.
"""

from sentence_transformers import models

# encode two queries once; every pooling mode reuses these hidden states
sample_inputs = naive_tokenizer(
    query_texts[:2], padding=True, truncation=True, max_length=MAX_DOC_TOKENS, return_tensors="pt"
).to(DEVICE)
with torch.inference_mode():
    sample_output = naive_model(**sample_inputs)
    sample_hidden = sample_output.last_hidden_state

    # build one Pooling module per mode, all on the same hidden size
    pool_modules = {
        mode: models.Pooling(sample_hidden.shape[-1], pooling_mode=mode).to(DEVICE)
        for mode in ("cls", "mean", "max", "lasttoken")
    }

    # apply each pooler to a copy of the same token vectors
    sample_pooled = {
        mode: module(
            {
                "token_embeddings": sample_hidden.clone(),
                "attention_mask": sample_inputs["attention_mask"],
            }
        )["sentence_embedding"]
        for mode, module in pool_modules.items()
    }

print("pooled modes:", list(sample_pooled), "| hidden:", tuple(sample_hidden.shape))
