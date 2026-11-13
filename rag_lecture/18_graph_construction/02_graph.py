r"""
### Inspect provenance

Build a small graph from this paper's predicted relations. Node identity uses the
exact surface text, not aggressive biomedical alias normalization. Each edge
retains its source document and spans; repeated edges remain separate. The graph
is a teaching example, not the HippoRAG index in the next section. Can you trace
each edge back to a substring of the paper?
"""

# Build a provenance-preserving multigraph from the predicted relations.
paper_graph = nx.MultiDiGraph()

# Walk every relation type and every prediction it produced.
for relation_type, predictions in relations["relation_extraction"].items():
    for prediction in predictions:
        head, tail = prediction["head"], prediction["tail"]

        # Verify each span really is a substring of the source text.
        for span in (head, tail):
            if (
                not (0 <= span["start"] < span["end"] <= len(graph_text))
                or graph_text[span["start"] : span["end"]] != span["text"]
            ):
                raise ValueError(f"Predicted span does not match source: {span}")

        # Add the edge, keeping document id and both spans as provenance.
        paper_graph.add_edge(
            head["text"],
            tail["text"],
            relation=relation_type,
            document_id=graph_doc_id,
            head_span=(head["start"], head["end"]),
            tail_span=(tail["start"], tail["end"]),
        )

# Show the raw edge list so every edge can be traced back.
display(list(paper_graph.edges(data=True)))
