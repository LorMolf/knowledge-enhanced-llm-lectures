r"""
### Encode token vectors

Unlike the dense models, each entry is a variable-length matrix, not a single
vector. Native role-specific methods apply the checkpoint's formatting and
scoring masks. Query expansion can retain additional query vectors, while
masked document tokens are filtered from the output. The returned row numbers
are positions among retained vectors, not guaranteed tokenizer positions. We
use the checkpoint's configured lengths, which may differ from the dense
models' 256-token limit. How many vectors represent the first inputs?
"""

# encode queries with the role-specific method (applies query formatting + expansion)
colbert_queries = colbert_model.encode_query(query_texts, show_progress_bar=True)

# encode documents with the role-specific method (applies doc formatting + masking)
colbert_docs = colbert_model.encode_document(
    corpus_texts, convert_to_numpy=True, show_progress_bar=True
)

# show the first inputs and the resulting per-input vector counts
print("First question:", query_texts[0])
print("First document:", corpus_texts[0][:400])
print("Token-vector shapes:", colbert_queries[0].shape, colbert_docs[0].shape)
