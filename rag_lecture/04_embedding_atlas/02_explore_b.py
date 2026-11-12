r"""
### Render it

`EmbeddingAtlasWidget` takes the frame and three column names; everything else — the
scatter, the text panel, the neighbour list — is the widget's. The return value is what
the cell displays, and the click handler needs no code from us.

"""

# The widget is the whole view: scatter, text panel, neighbour list.
from embedding_atlas.widget import EmbeddingAtlasWidget


def show_atlas(texts, vectors, kinds, ids, method="pca"):
    """The widget over the frame built by atlas_frame."""
    frame = atlas_frame(texts, vectors, kinds, ids, method=method)
    return EmbeddingAtlasWidget(frame, text="text", x="x", y="y", neighbors="neighbors")


print("show_atlas defined: call it in the next cell to see the map")
