r"""
### The text pipeline it replaces

Every multimodal claim needs the baseline it beats. A text pipeline is OCR and then a text
encoder, and here it is the same model: `pytesseract` reads each page, the text tower
embeds it with the `Document` prompt, and the pages are ranked exactly as before. Same
questions, same corpus, same evaluator, one difference — the pixels went through a character
recogniser first. What does the page look like to the text tower where the OCR failed, and
is the nDCG gap bigger or smaller than the gap between two vision budgets?
"""

import importlib.util
import json
import shutil
from pathlib import Path

# OCR needs a binary and a wrapper; a missing one skips with a reason, never an error.
OCR_CACHE = Path("page_ocr.json")
OCR_AVAILABLE = shutil.which("tesseract") is not None and importlib.util.find_spec("pytesseract") is not None

if not OCR_AVAILABLE:
    print("skipped: tesseract is not installed, so there is no text pipeline to compare against.")
else:
    # Cache the recognised text: OCR is minutes, and a rerun should not pay for it.
    import pytesseract

    if OCR_CACHE.exists():
        ocr_texts = json.loads(OCR_CACHE.read_text())
    else:
        ocr_texts = {page_id: pytesseract.image_to_string(page) for page_id, page in mm_pages.items()}
        OCR_CACHE.write_text(json.dumps(ocr_texts))
    mean_chars = sum(len(text) for text in ocr_texts.values()) / len(ocr_texts)
    print(f"OCR read {len(ocr_texts)} pages, {mean_chars:.0f} characters per page")

    # The same text tower the questions use, pointed at recognised page text.
    ocr_embeddings = (
        gemma_model.encode(
            [ocr_texts[page_id][:2000] for page_id in mm_page_ids],
            prompt_name="Document",
            convert_to_tensor=True,
            show_progress_bar=True,
        )
        .cpu()
        .float()
    )
    ocr_rankings = rank_dense(gemma_query_embeddings, ocr_embeddings, mm_query_ids, mm_page_ids)
    display(evaluate_mm("EmbeddingGemma 2, OCR text", ocr_rankings))
