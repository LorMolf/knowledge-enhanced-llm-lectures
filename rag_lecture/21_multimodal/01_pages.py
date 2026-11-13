r"""
<a name="section-21"></a>

# Beyond text: retrieving page images

So far a paper was a string. Real documents also hold tables, charts, scanned
pages and slides, and an OCR step before the encoder throws that layout away. A
multimodal embedding model reads the page image itself and places it in the same
vector space as a text question, so retrieval stays the cosine ranking of
sections 03 and 07. Only the document side changes, from text to pixels — and section 08 measures what that
buys over the text pipeline it replaces, on the same pages and the same questions.

We use [ViDoRe healthcare](https://huggingface.co/datasets/vidore/syntheticDocQA_healthcare_industry_test)
(MIT license): 1,000 page images from PDFs that a crawler found with the query
"healthcare industry", so some are only loosely medical. An LLM wrote 100
questions while looking at the pages; they point to 65 distinct pages, and the
other pages are distractors. These are reports and slides, not PubMed papers, and
the questions are synthetic. Only the annotated page is labelled relevant, so
another page may answer a question too, as in section 01.

We freeze all 100 questions and a closed corpus of 300 distinct pages: every
annotated page plus random distractors. The decoded images stay in memory
(about 3 GB). This corpus is easier than the full benchmark, so do not compare
its scores with published ViDoRe numbers. Nothing in this section is trained or
tuned on the questions. Let's build the corpus and look at one page. What would
a text-only pipeline need to answer this question?
"""

import random

# constants for this section: dataset pin and closed-corpus size
MM_DATASET_ID = "vidore/syntheticDocQA_healthcare_industry_test"
MM_DATASET_REVISION = "d973f3da3e60c8eaf566efa2f7d2a1515cba40ed"
MM_CORPUS_SIZE = 300

# load the benchmark split and split rows into queries vs. page rows
mm_source = load_dataset(MM_DATASET_ID, split="test", revision=MM_DATASET_REVISION)
mm_texts = list(mm_source["query"])  # None for pages without a question
mm_files = list(mm_source["image_filename"])  # the real page ID
mm_query_rows = [i for i, text in enumerate(mm_texts) if text]
mm_queries = {f"q{n:03d}": mm_texts[i] for n, i in enumerate(mm_query_rows)}
mm_relevant = {f"q{n:03d}": {mm_files[i]} for n, i in enumerate(mm_query_rows)}
annotated_pages = {mm_files[i] for i in mm_query_rows}

# keep one row per distinct page, then sample distractors with a fixed seed
first_row = {}
for i, file in enumerate(mm_files):
    first_row.setdefault(file, i)  # one row per distinct page
page_distractors = sorted(set(first_row) - annotated_pages)
random.Random(SEED).shuffle(page_distractors)
mm_page_files = sorted(annotated_pages) + page_distractors[: MM_CORPUS_SIZE - len(annotated_pages)]

# decode only the chosen pages into memory and check the corpus is closed
mm_pages = {file: mm_source[first_row[file]]["image"] for file in mm_page_files}
assert len(mm_pages) == MM_CORPUS_SIZE and annotated_pages <= set(mm_pages)
print("questions:", len(mm_queries), "annotated pages:", len(annotated_pages), "corpus:", len(mm_pages))

# show the first question next to its annotated page image
first_qid = next(iter(mm_queries))
first_page = mm_pages[next(iter(mm_relevant[first_qid]))]
print(first_qid, mm_queries[first_qid], first_page.size)
display(first_page.resize((first_page.width // 3, first_page.height // 3)))
