r"""
<a name="section-02"></a>

# 02. A naïve AutoModel encoder

A transformer produces a contextual vector for **each token**. Let's start with
its first token, `[CLS]`, as a whole-text representation. This checkpoint was
pretrained on biomedical text; that alone does not train its CLS vectors for
question–document retrieval. Later we will keep the same weights and change
pooling. See the [BiomedBERT model card](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext).

![Encoder and pooling](assets/encoder-pooling.png)

*Figure: BERT encoder plus pooling, from the [Sentence-BERT paper](https://arxiv.org/abs/1908.10084).*

**Question:** what shape does the encoder return for one question, and what do
the first few coordinates of its first tokens look like?
"""

from transformers import AutoModel, AutoTokenizer

# 1. Load the biomedical checkpoint and its matching tokenizer
PUBMED_MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
naive_tokenizer = AutoTokenizer.from_pretrained(PUBMED_MODEL_ID)
naive_model = AutoModel.from_pretrained(PUBMED_MODEL_ID).to(DEVICE).eval()

# 2. Tokenize the first question, truncated to the document budget
tokens = naive_tokenizer(
    query_texts[0], return_tensors="pt", truncation=True, max_length=MAX_DOC_TOKENS
)
print("question:", query_texts[0])
print("tokens:", naive_tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))
print("input shape:", tokens["input_ids"].shape)

# 3. Run the encoder once; no gradients needed for a forward pass
with torch.inference_mode():
    raw_output = naive_model(**tokens.to(DEVICE))
print("token-state shape:", raw_output.last_hidden_state.shape)
print("First three tokens, first four coordinates:", raw_output.last_hidden_state[0, :3, :4].cpu())

# 4. Free the output tensor before the next cell
del raw_output
