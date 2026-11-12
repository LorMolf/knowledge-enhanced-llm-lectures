r"""
### Draw the graph

With the edges verified and stored, we can draw the multigraph. Parallel edges
are collapsed into a single label per node pair so the figure stays readable.
Which relation types connect the same two entities?
"""

# Draw the graph only if it has nodes.
if paper_graph.number_of_nodes():
    import matplotlib.pyplot as plt

    pos = nx.spring_layout(paper_graph, seed=SEED)
    nx.draw_networkx(paper_graph, pos=pos, node_size=1500, font_size=8, arrows=True)

    # 1. Collapse parallel edges into one label per (u, v) pair for drawing.
    edge_labels = {
        (u, v): ", ".join(sorted({edge["relation"] for edge in edges.values()}))
        for u, targets in paper_graph.adjacency()
        for v, edges in targets.items()
    }
    nx.draw_networkx_edge_labels(
        nx.DiGraph(paper_graph), pos=pos, edge_labels=edge_labels, font_size=7
    )
    plt.axis("off")
    plt.show()
