r"""
<a name="section-07"></a>

# 07. Asymmetric dense retrieval

A dense retriever compresses each question and each document to one vector. A
similarity score orders the documents; the representation itself depends on
how the model was trained and on the query instruction. We keep the document
collection and evaluation from sections 01–06 fixed.

![Dense retrieval pipeline](assets/dense-retrieval.png)

*Figure: the bi-encoder, one shared encoder over two towers, from the [Sentence-BERT paper](https://arxiv.org/abs/1908.10084).*

Let's load [BGE base v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). Its
model card recommends the instruction below for queries, not passages. We
retain this unchanged baseline for the later RAG and hybrid experiments. Which
parts of the pipeline can be computed before a question arrives?
"""

# Load the BGE encoder once; every later section reuses this same model.
from sentence_transformers import SentenceTransformer

# 1. constants for this section: model id and the query-side instruction
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

# 2. build the encoder on the shared device and cap its input length
bge_model = SentenceTransformer(BGE_MODEL_ID, device=DEVICE)
bge_model.max_seq_length = MAX_DOC_TOKENS

# 3. show the loaded model and one query as the model will actually see it
print(bge_model)
print("Example query input:", BGE_QUERY_PROMPT + query_texts[0])
