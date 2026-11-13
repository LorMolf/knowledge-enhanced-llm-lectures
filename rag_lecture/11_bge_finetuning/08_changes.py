r"""
The evaluator table says whether the ranking moved. A 2x2 grid says why - three cosine densities and the trainer's loss: for every question and every paper, the cosine to a labelled paper against the cosine to all the others, for the untrained CLS encoder, for BGE before training, and for BGE after. The CLS panel shows anisotropy: everything sits high and close, so a high cosine is not evidence of relevance.
"""

import torch.nn.functional as F

# build the labelled mask: True where a paper is relevant to a question
is_labelled = torch.tensor(
    [[doc_id in relevant_docs[qid] for doc_id in corpus_ids] for qid in query_ids]
)

# three encoders to compare: untrained CLS, BGE before, BGE after 40 steps
views = {
    "PubMedBERT CLS": (naive_query_embeddings, naive_corpus_embeddings),
    "BGE before": (bge_query_embeddings, bge_corpus_embeddings),
    "BGE after 40 steps": (bge_ft_test, bge_ft_docs),
}

# one panel per encoder, plus a fourth for the loss log
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
for ax, (title, (question_vectors, paper_vectors)) in zip(axes.flat, views.items()):
    # 3a. cosine similarity of every question against every paper
    scores = F.normalize(question_vectors.float(), dim=1) @ F.normalize(
        paper_vectors.float(), dim=1
    ).T
    # 3b. density of cosines to non-relevant papers (the background)
    sns.kdeplot(
        scores[~is_labelled].flatten().numpy(), fill=True, color="gray", label="other papers", ax=ax
    )
    # 3c. density of cosines to relevant papers (what training should push right)
    sns.kdeplot(
        scores[is_labelled].flatten().numpy(),
        fill=True,
        color="tab:green",
        label="labelled papers",
        ax=ax,
    )
    ax.set(title=title, xlabel="cosine(question, paper)")
    ax.legend(frameon=False)

loss_log = pd.DataFrame([row for row in bge_trainer.state.log_history if "loss" in row])
axes.flat[3].plot(loss_log["step"], loss_log["loss"], marker="o")
axes.flat[3].set(title="training loss", xlabel="step", ylabel="loss")

plt.tight_layout()
plt.show()

# free the trainer and its tensors now that the figure is drawn
del bge_trainer, bge_train_model, bge_ft_docs, bge_ft_val, bge_ft_test
print("freed the trainer; three densities and the loss are on one grid")
