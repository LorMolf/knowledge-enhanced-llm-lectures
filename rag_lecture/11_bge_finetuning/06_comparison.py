r"""
### How will we compare runs?

We will display the existing evaluator rows side by side and inspect the first
test question whose top-paper membership changes, if any. Paper IDs, excerpts
and known labels make the change concrete; they do not create a new metric.
Use validation for model selection, not these descriptive test comparisons.
"""

# Called by the evaluation cell below, once both runs exist in the ledger.
def show_training_comparison(baseline, updated):
    missing = [name for name in (baseline, updated) if name not in all_rankings]
    if missing:
        print("not in the ledger yet:", missing, "- run the cell that scores this model first")
        return

    # show the evaluator rows for both runs side by side
    rows = [row for row in evaluation_rows if row["model"] in (baseline, updated)]
    display(pd.DataFrame(rows))

    # find the first question whose top-paper set differs between runs
    before, after = all_rankings[baseline], all_rankings[updated]
    same_top = lambda q: {h["corpus_id"] for h in before[q]} == {h["corpus_id"] for h in after[q]}
    qid = next((q for q in query_ids if not same_top(q)), query_ids[0])

    # collect the ranked paper IDs for that question in each run
    old_ids = [hit["corpus_id"] for hit in before[qid]]
    new_ids = [hit["corpus_id"] for hit in after[qid]]

    # papers that entered or left the top set
    changed_ids = sorted(set(old_ids) ^ set(new_ids))

    # show the question, both orders, and the changed papers with labels
    display(
        {
            "question_id": qid,
            "question": queries[qid],
            "baseline_order": old_ids,
            "updated_order": new_ids,
            "changed_papers": [
                {
                    "id": doc_id,
                    "known_relevant": doc_id in relevant_docs[qid],
                    "excerpt": corpus[doc_id][:400],
                }
                for doc_id in changed_ids
            ],
        }
    )
