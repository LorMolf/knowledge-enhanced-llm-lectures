#!/usr/bin/env python3
"""Run RAG_Lecture.ipynb cell by cell in one namespace, headless, and report the first failure.

Used on the GPU box and in Colab: no UI, no editor. `--from N` resumes at cell N (1-based),
`--skip-shell` leaves the `%%shell` install cell to the environment setup.

    MPLBACKEND=Agg python _run_harness.py --skip-shell
"""
import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

import nbformat

try:
    from IPython.display import display as _ipython_display
    def display(*args, **kwargs):          # notebook cells call display() freely
        for arg in args:
            _ipython_display(arg)
except Exception:                           # plain python without IPython
    display = print


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--skip-shell", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="collect every failure instead of stopping")
    ap.add_argument("--notebook", default="RAG_Lecture.ipynb")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="cap the process to this share of total GPU memory, to imitate a smaller card")
    args = ap.parse_args()

    nb = nbformat.reads(Path(args.notebook).read_text(), as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    print(f"{len(cells)} code cells, starting at {args.start}", flush=True)

    if args.mem_fraction:
        import torch
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
        print(f"memory cap: {args.mem_fraction:.0%} of {total:.1f} GB = {total * args.mem_fraction:.1f} GB", flush=True)

    ns = {"__name__": "__main__", "display": display}
    report, failures = [], 0

    for i, cell in enumerate(cells, 1):
        if i < args.start:
            continue
        src = cell.source
        head = next((l for l in src.strip().splitlines() if l.strip()), "(empty)")[:70]

        if src.lstrip().startswith("%%"):
            if args.skip_shell:
                print(f"CELL {i:03} SKIP shell", flush=True)
                report.append({"cell": i, "status": "skip", "head": head})
                continue
            done = subprocess.run(["bash", "-lc", src.split("\n", 1)[1]], capture_output=True, text=True)
            tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
            ok = done.returncode == 0
            print(f"CELL {i:03} {'OK  ' if ok else 'ERR '} shell :: {' | '.join(tail)[:180]}", flush=True)
            report.append({"cell": i, "status": "OK" if ok else "ERR", "kind": "shell", "head": head, "rc": done.returncode})
            if not ok:
                failures += 1
                if not args.keep_going:
                    break
            continue

        try:
            exec(compile(src, f"cell{i}", "exec"), ns)
            try:
                import torch
                if torch.cuda.is_available():
                    head_mem = f" [{torch.cuda.memory_allocated() / 1e9:.1f} GB]"
                else:
                    head_mem = ""
            except Exception:
                head_mem = ""
            print(f"CELL {i:03} OK   {head}{head_mem}", flush=True)
            report.append({"cell": i, "status": "OK", "head": head})
        except Exception:
            tb = traceback.format_exc()
            print(f"CELL {i:03} ERR  {head}", flush=True)
            print(tb[-2500:], flush=True)
            report.append({"cell": i, "status": "ERR", "head": head, "traceback": tb[-2500:]})
            failures += 1
            if not args.keep_going:
                break

    print(f"\n==== SUMMARY: {len(report)} cells attempted, {failures} failed ====")
    Path("harness_report.json").write_text(json.dumps(report, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
