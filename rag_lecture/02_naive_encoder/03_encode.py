r"""
### Encode questions and documents

For each input, take the final hidden state at token position zero. Section 06
will pass a native pooling module to the same encoder loop. Both representations
use the same weights, truncation and data. What are the resulting tensor shapes?
"""

def encode_biomed(texts, encoder, tokenizer, pooling=None, batch_size=16):
    # collect one CPU tensor per batch, then concatenate at the end
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            # tokenize this slice with the shared truncation budget
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=MAX_DOC_TOKENS,
                return_tensors="pt",
            ).to(next(encoder.parameters()).device)
            # run the encoder once and keep the last hidden state
            hidden = encoder(**inputs).last_hidden_state
            # pool: token-zero by default, or the native pooling module
            vectors = (
                hidden[:, 0]
                if pooling is None
                else pooling(
                    {"token_embeddings": hidden, "attention_mask": inputs["attention_mask"]}
                )["sentence_embedding"]
            )
            batches.append(vectors.cpu())
    return torch.cat(batches)


# encode both sides with the same weights so shapes are comparable
naive_query_embeddings = encode_biomed(query_texts, naive_model, naive_tokenizer)
naive_corpus_embeddings = encode_biomed(corpus_texts, naive_model, naive_tokenizer)

# the shapes answer the question: (n_texts, hidden_size) on CPU
print("questions:", naive_query_embeddings.shape, "documents:", naive_corpus_embeddings.shape)
assert naive_query_embeddings.device.type == naive_corpus_embeddings.device.type == "cpu"
