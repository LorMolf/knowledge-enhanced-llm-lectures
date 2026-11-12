r"""
Shared entities are the paths a graph retriever would walk, and a missed alias (one protein written two ways) splits a path that a human would follow. Each paper is a square node, green when labelled for this question; round nodes are entities mentioned by at least two of these papers. Which entity links a labelled paper to a retrieved one, and which pair should be the same node but is not?
"""

# 1. Restrict the drawing to papers plus their shared entities.
view = paper_graph.subgraph(bridge_ids + shared)
positions = nx.spring_layout(view, seed=SEED)

# 2. Draw papers as squares, green when labelled for this question.
nx.draw_networkx_nodes(
    view,
    positions,
    nodelist=bridge_ids,
    node_shape="s",
    node_color=["tab:green" if doc_id in relevant_docs[example_qid] else "lightgray" for doc_id in bridge_ids],
    node_size=700,
)

# 3. Draw shared entities as round orange nodes, labelled so we can read them.
nx.draw_networkx_nodes(view, positions, nodelist=shared, node_color="tab:orange", node_size=300)
nx.draw_networkx_edges(view, positions, alpha=0.4)
nx.draw_networkx_labels(view, positions, labels={node: node for node in shared}, font_size=7)

# 4. Show the figure.
plt.axis("off")
plt.title("Papers (squares) and shared entities (round nodes)")
plt.show()
