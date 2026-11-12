# RAG lecture: authoring protocol

This folder is the editable source of the new lecture, separate from the older `rag_chatbot` notebook. The `.ipynb` is a generated teaching artifact, not the editing source.

## One explanation–code pair per Python file

Sections and pairs have zero-padded numeric prefixes. Read and assemble them in lexical order:

```text
rag_lecture/
  README.md
  pyproject.toml
  uv.lock
  install.sh
  build_notebook.py
  check_sources.py
  assets/
  01_dataset/
    01_load.py
    02_sample.py
    03_inspect.py
  02_naive_encoder/
    01_load.py
    03_encode.py
  ...
  20_comparison/
  21_multimodal/
```

Every pair is ordinary Python with one leading raw, triple-double-quoted module docstring. That string is the Markdown cell; the rest is the code cell. Use `r"""..."""` so LaTeX backslashes are preserved. Example:

```python
r"""
### Inspect one question

Let's look at its text and the IDs of its supporting papers. The answer is a
reference for discussion; it must not be inserted into the retrieval corpus.
"""
example = qa_train[0]
display({key: example[key] for key in ("question", "relevant_passage_ids")})
```

- One concept or experiment per pair. Split long code rather than hide it in a new abstraction.
- Pair files share one notebook/kernel namespace. They are **not standalone programs**. Their Markdown names any required earlier section.
- Keep imports, model IDs, prompts, parameters and intermediate values visible. Use functions only for operations genuinely reused later.
- Use native library APIs for metrics, mining, training, fusion, plotting and interactive exploration. Never write another retrieval framework.
- Do not use IPython shell magics in `.py` sources. Environment/server commands belong in fenced Markdown instructions. Never start a server, train, install packages or download models as a side effect of importing a helper or building the notebook.
- Author the narration in the style of *To Prompt or to Finetune?*: brief transitions, concrete examples, theory when needed, then “What do you notice?”. Do not state an expected improvement as an observed result.
- Include useful figures or meaningful plots, not model logos as decoration. Attribute external diagrams and links. Local image paths must work relative to this folder and the generated notebook.

## Experimental contract

- Dataset: `enelpol/rag-mini-bioasq`, pinned to the explicit revision in section 01. Its QA `train`/`test` splits are distinct from the corpus configuration's split named `test`. Derive validation from QA training data.
- Use a small, frozen **closed teaching corpus**, with all required relevant documents retained and explicit distractors. Keep the same corpus and held-out questions for every retriever. Do not claim these scores measure full-PubMed retrieval.
- Only document text enters the index and graph. Answers and query-specific relevance labels are supervision/evaluation, never graph edges or model inputs.
- Sentence Transformers `InformationRetrievalEvaluator` is the only retrieval-effectiveness calculator. Use its ranked-hit metric path for retrievers that do not share its encoding interface; align hits to `queries_ids`, preserve real document IDs, deduplicate and retrieve enough results for every cutoff.
- Use the same total order for evaluation, fusion and inspection: descending score, then ascending original string document ID. Resolve boundary ties before taking top-k. Rerunning an experiment replaces its ledger row rather than duplicating it.
- TrecTools: one overlap table between selected runs, with no plot; the informative figure is the per-question rank heatmap beside it. ranx: rank fusion, not a second evaluation engine. Embedding Atlas: required early interactive exploration using the actual experiment embeddings. Seaborn/Matplotlib: small explanatory figures, including score distributions before and after training, chunking, and per-question wins and losses.
- Keep the naïve encoder before the explicit Sentence Transformers module/pooler/pooling/normalization lesson. Hold weights fixed when comparing pooling.
- For Jina v5, vary native adapter `task` separately from query/document `prompt_name`; re-encode both sides. Do not confuse hosted task names with the local model API.
- Log training and validation with W&B. Test questions are not used for negative mining or hyperparameter selection. Exclude every known positive from negatives and inspect possible false negatives.
- For the small training demonstration, retain questions with a globally exclusive positive and exclude all training-positive documents from explicit-negative pools. Display the retained cohort size and use that same cohort for every matched run. This conservative restriction prevents known cross-row false negatives without introducing a custom batch sampler; it does not remove unjudged positives.
- Use Gemma 4 12B (QAT dynamic Q4) through `llama-server`. Tool calling is a small schema → requested call → allowlisted function → tool-result example, not an agent framework.
- Show graph construction from corpus text with provenance. A NetworkX graph is not automatically a HippoRAG2 index; use the native HippoRAG2 indexing path unless an extractor adapter has been separately verified.
- Section 21 is the only multimodal section and has its own closed corpus: ViDoRe healthcare at a pinned revision, all 100 questions, 300 distinct pages, scored with the same evaluator into a separate `mm_rows` ledger (the BioASQ ledger is not touched). EmbeddingGemma 2 loads through Sentence Transformers. MetaEncoder v1 is remote code from a pinned snapshot, needs about 64 GB of free GPU memory and `torchvision`, and its two pairs print a skip message and leave the rest of the section runnable when memory is missing. Nothing is trained or tuned on these questions.

## Creation and review boundary

The user has authorized source creation and a local uv environment, **not execution of lecture code on this laptop**. Do not execute pair files, build/run the notebook, load models, download model weights or run training. Static parsing, source/API review and dependency resolution are allowed. GPU/runtime and clean-kernel execution remain explicit future checks.

Terra review must cover three distinct questions: does the implementation match the documented APIs and data flow; can unnecessary code/dependencies be removed; and can a student understand the explanation, comments, figures, links and observed outputs from scratch? Reviewers exchange their concrete findings before final reconciliation. Static approval is not a claim of zero runtime bugs.

## Environment and later commands

The main project uses modern Sentence Transformers. HippoRAG2's upstream package pins older Torch/Transformers versions, so `hippo_backend/` has a separate uv environment and lockfile. Do not override those pins to force both stacks into one kernel. Section 19 exchanges a small JSON input/result file with that backend; the main notebook still owns the corpus IDs and computes every IR metric.

`install.sh` prepares both locked environments and is independent of the caller's working directory. It does not build a notebook or import lecture models. `uv` must already be installed; caches default to this project's ignored `.cache/uv` directory.

On the intended execution machine, from this folder after reviewing the sources:

```bash
bash install.sh
uv run python check_sources.py       # parser checks, compilation, lint and SVG checks only
uv run python build_notebook.py      # assembles cells; does not execute them
uv run jupyter lab
```

The setup script and static checks are allowed during source creation. Notebook assembly and execution, model downloads and training remain deferred. `REVIEW.md` records the acceptance boundary and the Terra review outcomes; detailed findings live in `reviews/`.

`build_notebook.py` inlines every `assets/` figure as a base64 data URI, so the generated `RAG_Lecture.ipynb` is self-contained: it can be uploaded to Colab on its own, without the repository folder. The first cell carries the same front matter as the previous lecture notebook (title, author block, course, group, index with anchors) and sets the seeds; environment and `llama-server` commands stay in fenced Markdown because a pair cannot begin a cell with an IPython `%%shell` magic.

## Sections

01 Dataset; 02 Naïve encoder; 03 Similarity; 04 Embedding Atlas; 05 Retrieval evaluation; 06 Pooling; 07 Dense models; 08 Jina tasks; 09 Late interaction; 10 Contrastive learning; 11 BGE fine-tuning; 12 Negative mining; 13 ColBERT fine-tuning; 14 Gemma 4 12B server; 15 Dense RAG; 16 Hybrid retrieval; 17 Retrieval tools; 18 Graph construction; 19 HippoRAG2; 20 Comparison; 21 Multimodal page retrieval.
