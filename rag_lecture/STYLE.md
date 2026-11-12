# Cell style contract

The notebook is read in a classroom, projected, and run top to bottom in Colab by
students who have never used these libraries. A cell is a slide with an output,
not a script fragment.

## Hard rules

1. **One idea per code cell.** A cell either installs, loads, computes one step, or
   draws one figure. If a cell does two of those, split it.
2. **Length.** Aim for 12 code lines. Over 12, every block carries a numbered
   comment (`# 1. …`). Hard cap 30, and only for one training run, one figure with
   its data prep, or one function definition. Beyond that: split, or extract the
   helper into the previous cell.
3. **Comments.** Every cell has at least one comment naming *why* the step exists,
   and one short comment per block (`# 1. free the previous model`, `# 2. encode`).
   A reader must be able to skim comments only and follow the lecture.
4. **No dense expressions.** Name intermediate values. `scores = q @ d.T` then
   `best = scores.max(dim=1)` beats one nested call. Comprehensions are allowed only
   when the whole expression fits on two lines.
5. **Output first.** Every code cell ends with one visible statement: `print`,
   `display`, or a figure. Setup cells (imports, constants, seeds, `def`) end with
   one `print` naming what they defined.
6. **The install cell is `%%shell` with plain bash** (`set -e`, quoted packages,
   `uv pip install --system`). It runs in place; the student never copies a fenced
   block. The setup cell that follows is the real check: it imports every package and
   prints `loaded X | expected Y`.
7. **No other IPython magics.** `%%shell` exists only in the install cell, and it is
   the one cell `check_sources.py` does not compile. Servers stay Python
   (`Popen` + poll) because later cells call them from Python.
8. **Progression.** Within a section: baseline that runs → one change → the measured
   difference. Never introduce two new ideas in the same step.
9. **Contiguity of comparison.** When two things are compared, their outputs appear
   one after the other in the same cell or in adjacent cells, with the markdown in
   between naming what to look at.
10. **Constants** for one section live in exactly one cell at its top, one per line,
    with a comment. No magic numbers inside loops.
11. **Determinism.** Any random choice takes `SEED`/`random.Random(SEED)`. Any
    threshold used in a claim is a named constant.
12. **Failures are explicit.** A guard that can stop the notebook prints the reason
    and sets a `SKIP_<section>` flag (or `CAN_TRAIN`) that dependent cells check and
    report. Only correctness guards (cohort size, ranking shape, CUDA) may `raise`,
    and the message names the section to re-run.
13. **Expensive cells cache.** Anything over ~30 s writes its result (`parquet` or
    `json`) and reads it back on re-run; a `RUN_LLM`-style flag forces a refresh.
14. **Free the GPU** between model sections: `del model`,
    `torch.cuda.empty_cache()`, and print what is still allocated when it matters.
15. **One kernel: earlier cells define the names later ones use.** A cell that reads
    `corpus`, `evaluate_run` or `query_ids` is not missing an import. `check_sources.py`
    runs ruff `F821` over all cells concatenated in lexical order, so a name that is
    genuinely undefined fails the build. Reviewers must not flag shared-kernel names,
    and cells must not re-define a constant that an earlier cell already owns.

## Structure of a pair

`NN_name.py` = one raw docstring (the markdown cell) + the code cell. The markdown
states what the cell does and asks one question the output answers. The code answers
it. Nothing else: no instructions that only exist in prose.

## Environment

Google Colab, T4 or A100. The first runnable cell installs the locked dependencies
with `uv` and starts `llama-server`; the second prints interpreter and versions and
sets the seeds. Everything after that assumes both succeeded.
