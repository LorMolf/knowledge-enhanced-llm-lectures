"""Static source/protocol checks only. Never import or execute lecture pairs."""

import ast
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from build_notebook import ROOT, pair_paths, split_pair


def check_parser():
    sample = 'r"""\n# Café\n\\text{query} → document\n"""\nvalue = "三"\n'
    markdown, code = split_pair(sample)
    assert markdown == "# Café\n\\text{query} → document"
    assert code == 'value = "三"'
    ast.parse(code)
    invalid_pairs = (
        "print(1)",
        'r"""\nText only\n"""',
        '"""\nNon-raw text\n"""\nprint(1)',
        'r"""Inline closing"""\nprint(1)',
        'r"""\nText\n"""; print(1)',
        'r"""\n  \n"""\nprint(1)',
    )
    for invalid in invalid_pairs:
        try:
            split_pair(invalid)
        except ValueError:
            continue
        raise AssertionError(f"Parser accepted an invalid pair: {invalid!r}")


if __name__ == "__main__":
    check_parser()
    paths = pair_paths()
    sections = {path.parent.name.split("_", 1)[0] for path in paths}
    assert sections == {f"{i:02d}" for i in range(1, 22)}, f"Section coverage: {sorted(sections)}"
    for directory in {path.parent for path in paths}:
        prefixes = [path.name.split("_", 1)[0] for path in directory.glob("*.py")]
        assert len(prefixes) == len(set(prefixes)), f"Repeated pair number: {directory}"
    code_blocks = []
    python_blocks = []
    markdown_blocks = []
    for path in paths:
        assert re.fullmatch(r"\d{2}_[a-z0-9_]+\.py", path.name), path
        markdown, code = split_pair(path.read_text(encoding="utf-8"), str(path))
        if not code.lstrip().startswith("%%"):  # a `%%shell` cell is bash, not Python
            compile(code, str(path), "exec")  # Compile only: never evaluate a lecture cell.
            python_blocks.append(code)
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown):
            if not target.startswith(("https://", "http://", "data:")):
                assert (ROOT / target).is_file(), f"{path}: missing image {target}"
        code_blocks.append(code)
        markdown_blocks.append(markdown)
    anchors = re.findall(r'<a name="(section-\d{2})">', "\n".join(markdown_blocks))
    assert len(anchors) == 21 and set(anchors) == {f"section-{i:02d}" for i in range(1, 22)}
    # Style ratchet: every cell explains itself and shows something. Report now,
    # enforce with --strict-style once the rewrite is done.
    offences = []
    for path, code in zip(paths, code_blocks):
        if code.lstrip().startswith("%%"):
            continue
        body = [line for line in code.splitlines() if line.strip()]
        if not any(line.lstrip().startswith("#") for line in body):
            offences.append((path, "no comment"))
        if not re.search(r"\b(print|display)\s*\(|\.show\(\)", code):
            offences.append((path, "no visible output"))
    print(f"style: {len(offences)} offences in {len({path for path, _ in offences})} pairs")
    for path, why in offences[:20]:
        print(f"  {path.relative_to(ROOT)}: {why}")
    if "--strict-style" in sys.argv:
        assert not offences, f"{len(offences)} style offences"
    combined_code = "\n\n".join(python_blocks)
    compile(combined_code, "<assembled-lecture>", "exec")
    backend = ROOT / "hippo_backend" / "run.py"
    compile(backend.read_text(encoding="utf-8"), str(backend), "exec")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E9,F63,F7,F821",
            "--stdin-filename",
            "RAG_Lecture.py",
            "-",
        ],
        input=combined_code,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E9,F63,F7,F821",
            str(ROOT / "build_notebook.py"),
            str(ROOT / "check_sources.py"),
            str(backend),
        ],
        check=True,
    )
    for image in sorted((ROOT / "assets").glob("*.svg")):
        ET.parse(image)
    print(
        f"Static checks passed: {len(sections)} sections, {len(paths)} pairs; lecture code NOT executed."
    )
