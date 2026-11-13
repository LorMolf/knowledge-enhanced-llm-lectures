r"""
<h1> <img src="assets/title.png" width="110" alt="A detective with a magnifying glass, the lecture title icon"> <strong>Retrieval-Augmented Generation: from Similarity to Multimodal Retrieval</strong></h1>

```
Author: 🎤Lorenzo Molfetta
Professor: Gianluca Moro

{name}.{surname}@unibo.it
```

_Alma Mater Studiorum Università di Bologna_

**Course**: TextMining & LLMs

**Research Group**: <mark><strong>UniboNLP@Cesena</strong></mark>

**Our website**: [link](https://disi-unibo-nlp.github.io)

<img src="https://drive.google.com/uc?export=view&id=1uskMMZ3v4QSDkC-eH5wTiTMmKiVfZFHs" width="300">

Retrieval-Augmented Generation answers a question by first finding evidence and
only then writing an answer. The retrieval step decides what the model can say,
so this notebook spends most of its time there: what an encoder returns, how a
ranking is scored, how a retriever is trained, and how lexical, dense, late
interaction, graph and multimodal retrieval differ on the same questions.

We use one biomedical corpus of paper passages and one frozen set of held-out
questions throughout. A small local language model then answers from the papers
each retriever returned, so every step is measured and every claim can be traced
back to a paper ID and its text.

## Index

1. [Dataset and linked papers](#section-01)
2. [A naïve AutoModel encoder](#section-02)
3. [Similarity, geometry and rankings](#section-03)
4. [Explore actual embeddings with Atlas](#section-04)
5. [Evaluate retrieval](#section-05)
6. [Pooling and normalization](#section-06)
7. [Asymmetric dense retrieval](#section-07)
8. [Jina task and role ablations](#section-08)
9. [ColBERT and Reason-ModernColBERT](#section-09)
10. [Contrastive and metric learning](#section-10)
11. [Fine-tune BGE](#section-11)
12. [Mine and compare negatives](#section-12)
13. [Fine-tune ColBERT](#section-13)
14. [A reasoning model, served from Python](#section-14)
15. [Dense RAG](#section-15)
16. [BM25, hybrid fusion and reranking](#section-16)
17. [Retrieval tools and a second search](#section-17)
18. [Extract a graph from paper text](#section-18)
19. [HippoRAG 2](#section-19)
20. [Compare the observed results](#section-20)
21. [Beyond text: retrieving page images](#section-21)

<a name="section-01"></a>

# Dataset and linked papers

### How this runs

This is the single-author edition of the course's RAG lecture, built on the same front
matter, index and setup pattern as the chatbot lecture. Every image is embedded in the
`.ipynb` itself, so the notebook uploads alone; `assets/` matters only when rebuilding
from these sources.

The next two cells come before the lecture. The first is a shell cell: it installs
the pinned environment into the interpreter this kernel runs, and it is meant to be
run, not copied. The second sets the seeds and prints every version it actually
loaded, so a missing restart shows up immediately as `loaded X | expected Y`.

Prepare both locked environments with `bash install.sh` and use the main
project's Python kernel when you run it on your own machine instead of Colab.
Encoder comparisons may run on CPU, but the training sections explicitly require
CUDA. Gemma 4 12B runs in a separate `llama-server` process, which the notebook starts
for you in section 14; HippoRAG uses its isolated environment and can make many
extraction requests. Section 21 needs a GPU, though EmbeddingGemma 2 is small enough for a free T4.

### Install the pinned stack

**Question:** does this cell leave the runtime with every pinned package importable?
"""

%%shell
# Pins match pyproject.toml and uv.lock; torch stays as Colab ships it.
set -e

command -v uv >/dev/null || pip install -q uv

uv pip install --system \
  "sentence-transformers[train]==6.2.0" "transformers==5.22.2" "datasets==4.8.5" \
  "huggingface-hub" "sentencepiece" "pandas>=3" "scikit-learn>=1.6" \
  "matplotlib" "seaborn" "embedding-atlas>=0.26" "umap-learn>=0.5" "anywidget" \
  "trectools==0.0.50" "ranx>=0.3.20,<0.4" "rank-bm25==0.2.2" "wandb>=0.19" \
  "gliner2[local]==2.1.0" "networkx>=3.4" "requests" "pyarrow" "openai>=1.60"

# llama.cpp's OpenAI-compatible server, built once with CUDA: this is the slow line.
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
  uv pip install --system --reinstall --no-binary llama-cpp-python "llama-cpp-python>=0.3.16"

# The OCR binary for the text-pipeline baseline in section 21; apt, not pip.
apt-get -qq update >/dev/null && apt-get -qq install -y tesseract-ocr >/dev/null
uv pip install --system "pytesseract>=0.3.13"

echo "Installed. Restart the runtime (Runtime > Restart session), then run the next cell."
