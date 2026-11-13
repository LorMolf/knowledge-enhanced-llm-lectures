r"""
CLS pooling must reproduce the raw CLS token vector exactly, which validates the wiring. We then print the first five dimensions of each mode side by side and compare the learned `pooler_output` against raw CLS.
"""

# sanity check: CLS pooling must equal the raw CLS token vector
assert torch.allclose(sample_pooled["cls"], sample_hidden[:, 0], atol=1e-6)

# compare the learned pooler output against raw CLS pooling
pooler_output = getattr(sample_output, "pooler_output", None)
display(
    pd.DataFrame({mode: vectors[0, :5].cpu().tolist() for mode, vectors in sample_pooled.items()})
)
print(
    "token vectors:",
    tuple(sample_hidden.shape),
    "pooler output:",
    None if pooler_output is None else pooler_output[0, :5].cpu(),
)
