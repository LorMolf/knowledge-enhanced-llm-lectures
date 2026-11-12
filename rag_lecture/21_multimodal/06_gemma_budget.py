r"""
The 280-token row was already measured, so we reuse it as the baseline of the sweep. We then re-encode the same pages at the low and high budgets, timing each pass so cost is visible next to score.
"""

import time

# 1. baseline: reuse the 280-token row already measured earlier
default_row = next(
    row for row in mm_rows if row["model"] == "EmbeddingGemma 2, 280 image tokens"
)
budget_rows = [{"vision tokens": 280, "encode seconds": gemma_encode_seconds, **default_row}]

# 2. re-encode pages at the low and high budgets, timing each pass
GEMMA_BUDGET_BATCH = 4  # the budget sweep encodes this many pages per call
for budget in (70, 1120):
    started = time.perf_counter()
    page_vectors = (
        gemma_model.encode(
            [{"image": page} for page in mm_pages.values()],
            batch_size=GEMMA_BUDGET_BATCH,
            processing_kwargs={"image": {"max_soft_tokens": budget}},
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        .cpu()
        .float()
    )
    seconds = time.perf_counter() - started

    # 3. score the new page vectors against the same queries
    budget_rankings = rank_dense(gemma_query_embeddings, page_vectors, mm_query_ids, mm_page_ids)
    row = evaluate_mm(f"EmbeddingGemma 2, {budget} image tokens", budget_rankings)
    budget_rows.append({"vision tokens": budget, "encode seconds": seconds, **row})

print(f"measured {len(budget_rows)} budgets: {[r['vision tokens'] for r in budget_rows]}")
