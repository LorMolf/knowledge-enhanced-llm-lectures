r"""
### Run it, if this machine can

Fine-tuning needs CUDA: the multi-vector model holds one vector per token, and on a
CPU the same forty steps take long enough that students stop. The guard prints the
reason and leaves the notebook runnable; the next cell checks the same flag, so
skipping here does not produce a confusing `NameError` two cells later.
"""

# One flag for the whole section: later cells check the same name.
CAN_TRAIN = torch.cuda.is_available()
if CAN_TRAIN:
    colbert_trainer, colbert_train_model = train_colbert()
    loss_rows = [row for row in colbert_trainer.state.log_history if "loss" in row]
    print("steps logged:", len(loss_rows), "| last loss:", round(loss_rows[-1]["loss"], 4) if loss_rows else "n/a")
else:
    print("No CUDA: skipping the ColBERT fine-tuning run. Sections 14 onward still run.")
