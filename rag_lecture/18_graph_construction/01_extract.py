r"""
<a name="section-18"></a>

# 18. From text to a graph

![Graph retrieval overview](assets/graph-retrieval.png)

*Figure: HippoRAG's offline indexing and online retrieval, from [Gutierrez et al., 2024](https://arxiv.org/abs/2405.14831), Fig. 1.*

A graph records entities as nodes and predicted relations as directed edges.
The figure is fictional, not a clinical claim or a computed HippoRAG result.
Let's inspect extraction from one actual paper, including character offsets.
An extracted edge is a model prediction, **not** a verified biomedical fact.
[GLiNER2.5 model card](https://huggingface.co/fastino/gliner2.5-base-v1).
Which predicted spans are informative, and which are wrong?
"""

# 1. load the extractor and pick one document to inspect
from gliner2 import AutoExtractor
import networkx as nx

extractor = AutoExtractor.from_pretrained("fastino/gliner2.5-base-v1")
graph_doc_id = corpus_ids[0]
graph_text = corpus[graph_doc_id]

# 2. predict entity spans (with offsets and confidence) for four entity types
entities = extractor.extract_entities(
    graph_text, ["disease", "drug", "gene", "protein"], include_spans=True, include_confidence=True
)

# 3. predict relation spans (with offsets and confidence) for three relation types
relations = extractor.extract_relations(
    graph_text,
    ["treats", "associated_with", "interacts_with"],
    include_spans=True,
    include_confidence=True,
)

# 4. verify every entity span actually matches the source text at its offsets
for spans in entities["entities"].values():
    for span in spans:
        if (
            not (0 <= span["start"] < span["end"] <= len(graph_text))
            or graph_text[span["start"] : span["end"]] != span["text"]
        ):
            raise ValueError(f"Predicted entity span does not match source: {span}")

# 5. show the document, its predicted entities, and its predicted relations
display(
    {"document_id": graph_doc_id, "text": graph_text, "entities": entities, "relations": relations}
)
