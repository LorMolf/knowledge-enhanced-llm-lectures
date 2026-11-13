r"""
### Normalize without changing cosine

Normalize rescales nonzero vectors to unit length. Because cosine already
normalizes both operands, its scores should stay the same. Let's check that on
the two actual mean-pooled questions and inspect their norms; this is not a
claim that normalization leaves raw dot products unchanged.

**Question:** do the norms change while the cosine scores stay fixed?
"""

# build the normalizer and take the two mean-pooled questions
normalize_module = models.Normalize().to(DEVICE)
pooled = sample_pooled["mean"]
assert torch.all(pooled.norm(dim=1) > 0), "Cosine requires nonzero vectors"

# normalize a copy so the originals stay untouched
normalized = normalize_module({"sentence_embedding": pooled.clone()})["sentence_embedding"]

# cosine between the two questions, before and after normalizing
cosine_before = F.cosine_similarity(pooled[0:1], pooled[1:2])
cosine_after = F.cosine_similarity(normalized[0:1], normalized[1:2])
assert torch.allclose(cosine_before, cosine_after, atol=1e-6)

# same check against five real documents from the corpus
sample_docs = naive_corpus_embeddings[:5].to(DEVICE)
assert torch.all(sample_docs.norm(dim=1) > 0)
raw_scores = util.cos_sim(pooled[:1], sample_docs)
unit_docs = normalize_module({"sentence_embedding": sample_docs.clone()})["sentence_embedding"]
unit_scores = util.cos_sim(normalized[:1], unit_docs)
assert torch.allclose(raw_scores, unit_scores, atol=1e-6)

# show the norms shrinking to 1 while the cosines hold
print("norms:", pooled.norm(dim=1).cpu(), "→", normalized.norm(dim=1).cpu())
print("cosine before/after:", cosine_before.cpu(), cosine_after.cpu())
print("sample document cosine scores:", raw_scores.cpu(), unit_scores.cpu())
