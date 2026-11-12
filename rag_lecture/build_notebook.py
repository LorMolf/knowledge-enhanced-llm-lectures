"""Assemble text–code pairs into a notebook without executing any cells."""

import ast
import base64
import mimetypes
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent


def inline_assets(markdown):
    """Embed referenced local images so the notebook runs when uploaded alone (Colab)."""

    def data_uri(target):
        if not (ROOT / target).is_file():
            return None
        mime = mimetypes.guess_type(target)[0] or "application/octet-stream"
        encoded = base64.b64encode((ROOT / target).read_bytes()).decode()
        return f"data:{mime};base64,{encoded}"

    # Markdown images: ![alt](assets/x.svg)
    def markdown_image(match):
        uri = data_uri(match.group(1))
        return match.group(0) if uri is None else f"]({uri})"

    # HTML images, used for the title icon: <img src="assets/x.png" width="110">
    def html_image(match):
        uri = data_uri(match.group(1))
        return match.group(0) if uri is None else match.group(0).replace(match.group(1), uri, 1)

    markdown = re.sub(r"\]\((assets/[^)\s]+)\)", markdown_image, markdown)
    return re.sub(r"<img[^>]+src=\"(assets/[^\"]+)\"", html_image, markdown)


def split_pair(source, filename="<pair>"):
    """Use Python's parser, not quote splitting, to preserve Markdown and code."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return split_magic_pair(source, filename)
    if not source.lstrip().startswith('r"""'):
        raise ValueError(f"{filename}: begin with a raw triple-double-quoted Markdown string")
    markdown = ast.get_docstring(tree, clean=False)
    if not markdown or not markdown.strip() or len(tree.body) < 2:
        raise ValueError(f"{filename}: supply both Markdown and executable code")
    end = tree.body[0].end_lineno
    if source.splitlines()[end - 1].strip() != '"""':
        raise ValueError(f"{filename}: close the Markdown string on its own line")
    if tree.body[1].lineno <= end:
        raise ValueError(f"{filename}: begin code on a new line after the Markdown")
    code = "".join(source.splitlines(keepends=True)[end:]).strip("\n")
    return markdown.strip("\n"), code


def split_magic_pair(source, filename="<pair>"):
    """A `%%shell` install cell is bash, not Python: split on the closing fence instead."""
    if not source.lstrip().startswith('r"""'):
        raise ValueError(f"{filename}: begin with a raw triple-double-quoted Markdown string")
    lines = source.splitlines()
    close = next(
        (index for index, line in enumerate(lines[1:], 1) if line.strip() == '"""'), None
    )
    if close is None:
        raise ValueError(f"{filename}: close the Markdown string on its own line")
    markdown = "\n".join(lines[1:close])
    code = "\n".join(lines[close + 1 :]).strip("\n")
    if not code.lstrip().startswith("%%"):
        raise ValueError(f"{filename}: only a `%%magic` cell may hold non-Python code")
    return markdown.strip("\n"), code


def pair_paths():
    return sorted(ROOT.glob("[0-9][0-9]_*/*.py"))


if __name__ == "__main__":
    paths = pair_paths()
    if not paths:
        raise SystemExit("No source pairs found")
    cells = []
    for path in paths:
        markdown, code = split_pair(path.read_text(encoding="utf-8"), str(path))
        markdown = inline_assets(markdown)
        metadata = {"source_file": path.relative_to(ROOT).as_posix()}
        cells.append(nbformat.v4.new_markdown_cell(markdown, metadata=dict(metadata)))
        cells.append(nbformat.v4.new_code_cell(code, metadata=dict(metadata)))
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
    )
    nbformat.validate(notebook)
    output = ROOT / "RAG_Lecture.ipynb"
    nbformat.write(notebook, output)
    print(f"Wrote {output}: {len(paths)} pairs. No cells executed.")
