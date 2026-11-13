r"""
### Qwen: does the instruction change retrieval?

[Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
formats the query as `Instruct: …\nQuery:…` and leaves documents unprompted.
Let's compare that task instruction with an empty query prompt, without
changing documents or the test questions. These are experiments, not a promise
that either setting wins. The query instruction below specifies biomedical
literature retrieval rather than general web search.

> We load the model, encode the corpus once with an empty prompt, and print the
> query format and the resulting document matrix shape.
"""

# Load Qwen and encode the corpus once; queries get the instruction later.
QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_QUERY_PROMPT = "Instruct: Given a biomedical question, retrieve relevant scientific papers that answer the question\nQuery:"

# load the embedding model onto the notebook's device
qwen_model = SentenceTransformer(QWEN_MODEL_ID, device=DEVICE)
qwen_model.max_seq_length = MAX_DOC_TOKENS

# encode documents with an empty prompt (Qwen leaves docs unprompted)
qwen_corpus_embeddings = qwen_model.encode(
    corpus_texts,
    prompt="",
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).cpu()

# show the query format and the document matrix shape
print("Query input:", QWEN_QUERY_PROMPT + query_texts[0])
print("Document vector shape:", qwen_corpus_embeddings.shape)
