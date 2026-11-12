# Source-only acceptance

**Accepted as an unexecuted lecture source draft.** The project contains 20 numbered sections, 65 Markdown–code pairs and six static SVG figures. It is not an executed notebook and contains no invented experiment results.

## Verified by the controller

- The project-local main and isolated HippoRAG environments are installed and locked. `install.sh` succeeded from outside the project, checking 215 main-environment packages and 80 backend packages. It installs dependencies only.
- `check_sources.py` passed its parser rejection fixtures, Python compilation, combined-namespace undefined-name/error lint, section and unique-pair coverage, outline-anchor checks and SVG/image checks. It does not evaluate cells or import the backend.
- All 25 referenced primary URLs returned HTTP 200 or 206. This checks reachability, not legal applicability or the correctness of every linked statement.
- All six SVGs were rendered and inspected. Final proofs in `reviews/figures/` preserve arrows, dashes, labels and bounds. Earlier clipped Quick Look thumbnails and marker-incomplete PyMuPDF proofs were not accepted; fitting only the render-copy viewport resolved that preview issue without altering the authored coordinate system.
- The current sources match the source edition reviewed in round 2. File hashes, installed versions, commands, actual output, reference checks and reviewer receipts are in `reviews/verification.json`.
- No generated notebook or lecture output/model directories were created. The older notebooks and unrelated tracked files were left untouched. No commit or push was made.

The final static check returned:

```text
All checks passed!
All checks passed!
Static checks passed: 20 sections, 65 pairs; lecture code NOT executed.
```

## Terra review and cross-examination

Four Terra authors created the initial sections. Three separate `gpt-6.1-terra` reviewer sessions assessed correctness, simplicity and teaching clarity. Authors responded and repaired their sections; the controller integrated remaining fixes. In round 2, each reviewer read the other reviews, the author replies and the current sources. All three accepted the **source draft with runtime checks deferred**.

- [Correctness](reviews/01_correctness.md): original C01–C05 closed at source level.
- [Simplicity](reviews/02_simplicity.md): original SIM-01–SIM-11 closed at source level.
- [Teaching and figures](reviews/03_teaching.md): original T01–T09 closed at the stated source/figure level; all six final figure previews inspected.

The reports retain their historical first-round findings. Their **ROUND 2 / review-v3** sections contain the current assessments. These are separate same-model reviews, not human validation or proof of zero bugs.

## Decisions resolved in review

- **Training negatives:** select a conservative, globally exclusive-positive cohort and exclude all training-positive IDs from explicit negatives. Use native ordinary full batches and drop a partial tail; a duplicate-filtering sampler could otherwise change the effective batch size. Repeated negative columns retain repeated loss weight, which the lesson explains. No validation/test labels enter mining.
- **Ranking and memory:** make ties deterministic before truncation and reuse the same ranking helpers. Store multi-vector document indexes on CPU and score bounded blocks. Full-corpus scoring remains a classroom-scale simplification.
- **Jina:** inspect the actual pinned custom source, resolve a local pinned snapshot to prevent nested downloads drifting to another revision, and separate adapter tasks from role prefixes. Snapshot-file selection was checked against the repository tree without downloading weights.
- **HippoRAG:** retain the separate environment required by upstream dependency pins. Use native ingestion, a native no-prefix dense control, original document IDs, explicit fallback modes and seed provenance. The GLiNER/NetworkX demonstration is not passed off as a Hippo index.
- **W&B:** the controller accepts offline logging by default for this source draft. Online logging is an explicit option after secure pre-login on the execution machine. No W&B run was launched here.

The initial unsupported CLI/preflight and resume-argument attempts were corrected before successful workers ran. The dependency conflict was resolved by isolation rather than forcing incompatible versions. These recoveries do not convert failed attempts into successful checks.

## Still deliberately unverified

With separate execution permission on the intended machine: run the assembled notebook from a clean kernel, verify the pinned data and retained cohort, load all model/adapters, inspect actual Atlas/plot outputs, measure CPU/GPU memory, exercise W&B and LFM3 tools, and run Hippo extraction/indexing/retrieval with cache, fallback and provenance checks. Real retrieval scores, training outcomes and answer quality must come from those runs.

**No lecture cells, model inference, fine-tuning, server or Hippo backend were run on the laptop.**

# Added after the acceptance above

Section 21 (`21_multimodal/`, 9 pairs) was added on 12 November 2026, together with `torchvision` in `pyproject.toml`/`uv.lock` and small edits to `README.md`, `01_dataset/00_overview.py` (outline, GPU note) and `check_sources.py` (21 sections). These files differ from the hashes in `reviews/verification.json`; the Terra reviews and the 25-URL check do not cover section 21. Statically checked only; its APIs were read against the installed Sentence Transformers 6.2.0 / Transformers 5.22.2 sources and the pinned model cards. Still unverified: a clean-kernel run, the real scores, GPU memory and speed of MetaEncoder v1 (28.4B parameters, `batch_size=4` is a guess), and the vision-token ablation outputs. A delegated static reviewer failed on a rate limit; the API checks were done by the author.

## 12 November, final check before building

Changed since the block above: `anywidget` added (the Atlas widget failed to import without it, so sections 04, 06, 07 and 11 would have stopped at `EmbeddingAtlasWidget`); `transformers` floor raised to 5.15 and `pandas` to 3 to match what the code and `embedding-atlas` require; `01_dataset/00_overview.py` gained a Colab install block and the `llama-server` start command; section 16 gained `03_rerank.py` (cross-encoder reranking of BGE's top ten) and its heading now says so.

Checked statically: every third-party import in all 75 pairs resolves in the synced environment (`anywidget` was the only failure, now installed and re-checked); `from embedding_atlas.widget import EmbeddingAtlasWidget` works; `CrossEncoder` and its `predict` signature match the installed 6.2.0; `torchvision` 0.29.2 imports beside torch 2.15.

Still unverified: no cell has been executed. Colab behaviour, real scores, training outcomes, the Atlas render, the llama-server sections and section 21's GPU memory remain runtime checks.

## Design review round, 12 November

Two independent reviews (Opus 5.5, Fable 5.2) plus a cross-critique round judged the notebook for teaching value, not style. Applied from their agreement:

- Cut `06_pooling/05_atlas.py` (third Atlas of the same 40 points) and `10_contrastive_learning/02_triplets.py` (a loss the lecture never trains with); its idea moved into 10/01 as two sentences.
- `20_comparison/03_agreement.py` keeps the TrecTools overlap computation as a table only; the misleading zero-width "system rank" plot is gone. New `20/04` is the per-question heatmap of the rank of the first labelled paper.
- `11_bge_finetuning/05` is now "What did forty steps change?": score densities before and after training plus the training loss, at `logging_steps=2`. `12/04` plots the three mining runs' losses and validations.
- New `15_dense_rag/03`: 32 questions × 3 evidence conditions, with citation counts and a labelled BGE-cosine proxy. New `07_dense_models/06`: chunking at 3 and 1 sentences. New `18_graph_construction/03`: papers linked by shared entities for one question.
- `02/02` shows the token-length histogram of the whole corpus; `03/02` puts each question against its own labelled paper; `05/02` adds the ideal-ranking ceiling; `12/02` adds a negative-hardness strip plot; `10/01` draws the temperature heatmaps; `21/04` shows the labelled page as a fourth panel; `21/05` plots score and measured encoding time against vision tokens.
- Section 14 no longer says "in another terminal" only; it points at the Colab start command.

Not done, and why: one sorted dot plot of every ledger row (keeps 20/02's six metric panels instead); UMAP on the full corpus for Atlas (keeps the 40-point sample fast in Colab); token-level labels on the MaxSim heatmap; DataFrame instead of `display(dict)` in sections 14–17. Author overrides of both reviewers: MetaEncoder stays (two cells plus the ablation), 09/05 stays, and the comparison cell stays last so it includes every multimodal row.

## Closing pass on the review list, 12 November

Implemented after the two-model debate, including the items the first pass had left out:

- Cuts: `21_multimodal/08_meta_instruction.py` (a T4 cannot run it; instruction conditioning is measured on Jina in 08/04–05) and `08_jina_tasks/03_view.py` (the same 40 points a fourth time). 76 pairs now.
- `21/06` and `21/07` no longer raise: the memory check sets `META_AVAILABLE` and prints the skip message, so "Run all" reaches `21/09` and still plots the EmbeddingGemma rows. This replaces the reviewers' proposal to move `21/09` forward, which would have dropped the Meta rows from the comparison.
- `20/02` is now one sorted dot plot of every ledger row (recall@10 and nDCG@10) with a rank-1 table; `20/04` adds the questions every run misses, with the two competing papers side by side.
- `15/03` keeps the system prompt and passes an empty evidence list for "no context", counts cited and abstained answers, and prints the two highest and two lowest scoring answers.
- `11/05` adds the untrained-CLS density panel, so anisotropy is visible next to the before/after shift.
- `16/02` draws a bump chart of one question through BM25, RRF and BGE; sections 14–17 display DataFrames instead of dict dumps; `03/04`, `03/05` and `06/04` fit their ranges to the observed scores.
- `04/02` adds one UMAP view of the whole 2,000-document corpus (`umap-learn` is now a direct dependency; it was already installed through `embedding-atlas`).

Left as indices, deliberately: the token labels on the ColBERT MaxSim heatmap (the retained vectors are not tokenizer positions, so labels would be invented), bytes per document in `20/01` (the same lesson is numeric in `21/09`), and BM25 rank columns in `07/03` (BM25 does not exist yet at that point). Opus's Matryoshka addition was rejected in both rounds for being over budget.

Still unverified: nothing here has been executed. The UMAP view, the 96 generated answers, the bump chart and the guard's skip path all remain static-checked only.

## Cross-check against installed sources and published cards, 12 November

Every third-party call compared with the installed package and with the model card at the pinned revision; no lecture cell was executed.

- Embedding Atlas 0.26: `EmbeddingAtlasWidget(frame, **options)` accepts `text`, `x`, `y`, `neighbors` (present in `EmbeddingAtlasOptions`). `compute_projection` also offers `umap_args`, unused.
- UMAP 0.5.12: `UMAP(n_neighbors, min_dist, metric, random_state)` matches; it arrives with `embedding-atlas` and is now also declared directly.
- ranx: `fuse(runs, norm, method, params)` calls `fusion_switch(method)(runs, **params)`, and `rrf(runs, k=60)` takes `k`, so `params={"k": 60}` is honoured; min-max normalisation preserves ranks.
- trectools: `load_run_from_dataframe` requires exactly `query, q0, docid, rank, score, system`, which the section-20 frame supplies.
- gliner2: `extract_entities` lives on `BoundaryExtractor`, the class `AutoExtractor.from_pretrained` returns for `architecture: boundary`; 2.1.0 is the newest release on PyPI, so the pin is current. Signature `(text, entity_types, threshold, format_results, include_confidence, include_spans, …)` matches the call.
- Sentence Transformers 6.2: `InformationRetrievalEvaluator.compute_metrics(list)`; `encode(..., prompt_name, prompt, processing_kwargs via **kwargs)`; `MultiVectorEncoder` with `encode_query`, `encode_document`, `similarity`, instance `prompts`; `MultiVectorEncoderTrainer`, `MultiVectorEncoderTrainingArguments` and `MultiVectorMultipleNegativesRankingLoss` all exist at the imported paths.
- `processing_kwargs={"image": {"max_soft_tokens": N}}`: ST maps the `image` modality into processor kwargs, and `transformers`' `embedding_gemma2` processor reads `max_soft_tokens` from `images_kwargs`.
- EmbeddingGemma 2 card: 768 dimensions, `config_kwargs={"audio_config": None}` for 440M text+image, 280 image tokens by default with 70–1120 available, 8,192-token shared context, prompts for text only — our image calls pass no prompt.
- MetaEncoder v1 card and pinned `modeling_metaencoder.py`: `encode(inputs, instruction=None, batch_size=32)` returns `(n, 6656)` L2-normalised float32; `transformers>=5.15` is the stated floor and matches `pyproject.toml`; one 80 GB GPU. Repository sha unchanged (`ed04ebd4…`) although its metadata was touched today.
- Jina v5: `task="retrieval"` with `prompt_name="query"/"document"`, exactly as used in section 08. Qwen3-Embedding: the card recommends a task-specific `Instruct: …\nQuery:` string, which is what section 07 passes as `prompt=` instead of the shipped generic prompt.
- Reason-ModernColBERT card: `MultiVectorEncoder("lightonai/Reason-ModernColBERT")` then `encode_query`, the same pattern as `09/05`. ColBERTv2 loads through the native multi-vector loader as the section text claims.
- LFM3 GGUF: card requires llama.cpp b11960/v0.8.0; `b11960/llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz` returns 200 and the repo ships `LFM3-1.7B-Q4_K_M.gguf`, so `-hf LiquidAI/LFM3-1.7B-GGUF:Q4_K_M` resolves.
- Links: 34 of 34 external URLs in the pairs return 200 (the two loopback URLs are expected to fail). ViDoRe revision `d973f3da…` still resolves and matches the pin.
- Arithmetic restated in the text: 768 × 4 B × 10⁶ = 3.07 GB and 6,656 × 4 B × 10⁶ = 26.6 GB per million pages; random accuracy@10 = 10/300 ≈ 3%.

One fix from this pass: `07/06` encodes with BGE after section 03 parked it on the CPU, so the cell now moves it back to `DEVICE` first.
