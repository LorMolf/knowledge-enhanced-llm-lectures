r"""
### Questions to pages

Encode every page once, as an image, and every question with the `SearchQuery` prompt. The default budget is 280 vision tokens per image, and the timing of the page pass is recorded for later comparison.
"""

import time

# encode every page image once, timing the pass
started = time.perf_counter()
gemma_page_embeddings = (
    gemma_model.encode(
        [{"image": page} for page in mm_pages.values()],
        batch_size=8,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    .cpu()
    .float()
)
gemma_encode_seconds = time.perf_counter() - started

# encode every question with the SearchQuery prompt
gemma_query_embeddings = (
    gemma_model.encode(
        list(mm_queries.values()),
        prompt_name="SearchQuery",
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    .cpu()
    .float()
)

print(f"encoded {len(mm_pages)} pages and {len(mm_queries)} queries")
