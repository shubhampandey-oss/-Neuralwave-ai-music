# Models Directory

Trained PyTorch checkpoints are saved here:
- `best_model.pt`: Checkpoint with lowest recorded cross-entropy validation loss.
- `latest_model.pt`: Most recent checkpoint from training.
- Checkpoints contain `model_state_dict`, vocabulary dictionaries, architecture hyperparameters, training loss history, and generation metadata.
