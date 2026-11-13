r"""
### How much of the document does the encoder see?

For the passage shown in section 01, compare original and retained token counts,
including special tokens; the model cannot use a truncated tail. Then look at
every passage in the corpus at once. The dashed line is the 256-token limit used
from here on: passages to its right keep only their beginning. What share of the
corpus is truncated, and does that change what a "document" can contain?
"""

import matplotlib.pyplot as plt

# tokenize the example passage twice: once whole, once cut at the limit
full_document_tokens = naive_tokenizer(corpus[example_doc], truncation=False)["input_ids"]
retained_document_tokens = naive_tokenizer(
    corpus[example_doc], truncation=True, max_length=MAX_DOC_TOKENS
)["input_ids"]

# measure every passage in the corpus, then count how many exceed the limit
document_lengths = [len(naive_tokenizer(text, truncation=False)["input_ids"]) for text in corpus_texts]
truncated = sum(length > MAX_DOC_TOKENS for length in document_lengths)

# show the example side by side so the loss is visible
print(
    "example document:",
    example_doc,
    "original tokens:",
    len(full_document_tokens),
    "retained:",
    len(retained_document_tokens),
)

# histogram of all passage lengths with the truncation limit marked
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(document_lengths, bins=60, color="steelblue")
print("papers longer than", MAX_DOC_TOKENS, "tokens:", truncated, "of", len(document_lengths), f"({truncated/len(document_lengths):.1%})")
print("the tail past", MAX_DOC_TOKENS, "tokens never reaches the encoder: read the truncated papers as their first", MAX_DOC_TOKENS, "tokens")
ax.axvline(MAX_DOC_TOKENS, color="crimson", linestyle="--", label=f"limit {MAX_DOC_TOKENS}")
ax.set(
    xlabel="tokens per passage, special tokens included",
    ylabel="passages",
    title=f"{truncated} of {len(document_lengths)} passages are truncated ({truncated / len(document_lengths):.0%})",
)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
