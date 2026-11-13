r"""
Shared entities are the paths a graph retriever would walk, and a missed alias (one protein written two ways) splits a path that a human would follow. Each paper is a square node, green when labelled for this question; round nodes are entities mentioned by at least two of these papers. Which entity links a labelled paper to a retrieved one, and which pair should be the same node but is not?
"""

# Keep only nodes the graph actually has: a paper with no extracted entity is not a node.
view = paper_graph.subgraph([node for node in bridge_ids + shared if node in paper_graph])
papers_in_view = [doc_id for doc_id in bridge_ids if doc_id in view]
entities_in_view = [node for node in shared if node in view]
positions = nx.spring_layout(view, seed=SEED)

# Papers as squares, green when labelled for this question.
nx.draw_networkx_nodes(
    view, positions, nodelist=papers_in_view, node_shape="s",
    node_color=["tab:green" if doc_id in relevant_docs[example_qid] else "lightgray"
                for doc_id in papers_in_view],
    node_size=700,
)
nx.draw_networkx_labels(view, positions, labels={doc_id: doc_id[:8] for doc_id in papers_in_view}, font_size=6)

# Shared entities as round orange nodes, labelled so we can read them.
nx.draw_networkx_nodes(view, positions, nodelist=entities_in_view, node_color="tab:orange", node_size=300)
nx.draw_networkx_edges(view, positions, alpha=0.4)
nx.draw_networkx_labels(view, positions, labels={node: node[:14] for node in entities_in_view}, font_size=7)

# Show the figure.
plt.axis("off")
plt.title("Papers (squares) and shared entities (round nodes)")
plt.show()
print(f"drawn {len(papers_in_view)} papers and {len(entities_in_view)} shared entities; "
      f"{len(bridge_ids) - len(papers_in_view)} papers had no node to draw")
