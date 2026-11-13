r"""
### Draw the largest connected piece

A spring layout of every node turns into a hairball the moment the graph has more than a
few dozen, and the labels overlap by construction. So the figure draws one connected
component, with node names trimmed and the edge labels collapsed to one line per pair:
enough to see whether the graph is a chain, a star or a triangle, which is the shape the
retrieval later walks. The counts for the whole graph are printed underneath.
"""

# Pick the largest component: the only part a spring layout can draw legibly.
if paper_graph.number_of_nodes():
    import matplotlib.pyplot as plt

    largest = max(nx.connected_components(nx.Graph(paper_graph)), key=len)
    drawn = paper_graph.subgraph(largest)
    print(f"drawing the largest component: {drawn.number_of_nodes()} nodes, "
          f"{drawn.number_of_edges()} edges of {paper_graph.number_of_nodes()} / {paper_graph.number_of_edges()}")

    # A looser k spreads the component out; the same seed keeps the picture stable.
    pos = nx.spring_layout(drawn, seed=SEED, k=2.2 / max(len(drawn), 2) ** 0.5, iterations=200)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    nx.draw_networkx_nodes(drawn, pos, node_size=320, node_color="#9D2235", ax=ax)
    nx.draw_networkx_edges(drawn, pos, arrows=True, alpha=0.55, ax=ax)
    nx.draw_networkx_labels(drawn, pos, {node: str(node)[:8] for node in drawn.nodes}, font_size=6, ax=ax)

    # One label per node pair, because parallel edges would write on top of each other.
    edge_labels = {
        (u, v): ", ".join(sorted({edge["relation"] for edge in pair.values()}))
        for u, targets in drawn.adjacency()
        for v, pair in targets.items()
    }
    nx.draw_networkx_edge_labels(drawn, pos, edge_labels=edge_labels, font_size=5, label_pos=0.4, ax=ax)
    ax.set_title("one connected component, relation types merged per pair", fontsize=9, loc="left")
    ax.axis("off")
    plt.show()
