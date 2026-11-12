r"""
The previous graph came from one paper. Graph retrieval starts from a question: its labelled papers and the papers BGE returns for it. This cell collects those papers and builds a bipartite paper-entity graph by extracting entities from each one.
"""

import matplotlib.pyplot as plt

# 1. Collect the papers for this question: labelled ones plus BGE's top hits.
bridge_ids = sorted(relevant_docs[example_qid] | {hit["corpus_id"] for hit in all_rankings["BGE"][example_qid]})

# 2. Build a bipartite paper-entity graph by extracting entities from each paper.
paper_graph = nx.Graph()
for doc_id in bridge_ids:
    found = extractor.extract_entities(
        corpus[doc_id], ["disease", "drug", "gene", "protein"], include_spans=True
    )
    for spans in found["entities"].values():
        for span in spans:
            paper_graph.add_edge(doc_id, span["text"].casefold())

# 3. Free the extractor; we only need the graph from here on.
del extractor

# 4. Keep only entities mentioned by at least two of these papers (the bridges).
shared = [
    node
    for node in paper_graph
    if node not in bridge_ids and paper_graph.degree(node) >= 2
]

print(f"{len(bridge_ids)} papers, {len(shared)} shared entities")
