"""
Model training pipeline for NeuralWave AI.
Executes PyTorch training loop with checkpointing, validation, and real-time callbacks.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import NeuralWaveLSTM, resolve_device
from .dataset import get_dataloaders


def train_model(
    dataset_dir: str | Path,
    checkpoint_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.002,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
    device_name: str = "auto",
    seed: int = 42,
    val_split: float = 0.1,
    progress_callback: Optional[Callable[[int, int, float, Optional[float], float], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Trains NeuralWaveLSTM on the preprocessed dataset.
    """
    # Seed reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = resolve_device(device_name)

    # Load data
    train_loader, val_loader, context = get_dataloaders(
        dataset_dir=str(dataset_dir),
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
    )

    vocab_size = context["vocab_size"]
    vocab_to_idx = context["vocab_to_idx"]
    idx_to_vocab = context["idx_to_vocab"]
    seq_len = context["sequence_length"]

    model = NeuralWaveLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    ckpt_p = Path(checkpoint_dir)
    ckpt_p.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    best_loss = float("inf")
    start_time = time.time()
    interrupted = False

    for epoch in range(1, epochs + 1):
        if stop_check and stop_check():
            interrupted = True
            break

        model.train()
        running_loss = 0.0
        batch_count = 0

        epoch_start = time.time()
        for batch_x, batch_y in train_loader:
            if stop_check and stop_check():
                interrupted = True
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        if interrupted:
            break

        train_loss = running_loss / max(1, batch_count)

        # Validation step
        val_loss: Optional[float] = None
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_running = 0.0
            val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_logits, _ = model(vx)
                    v_loss = criterion(v_logits, vy)
                    val_running += v_loss.item()
                    val_batches += 1
            if val_batches > 0:
                val_loss = val_running / val_batches
                scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if val_loss is not None else None,
            "elapsed_seconds": round(epoch_elapsed, 2),
        })

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, val_loss, total_elapsed)

        # Checkpoint evaluation (save best model)
        eval_metric = val_loss if val_loss is not None else train_loss
        is_best = eval_metric < best_loss

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab_to_idx": vocab_to_idx,
            "idx_to_vocab": idx_to_vocab,
            "config": {
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "sequence_length": seq_len,
            },
            "training_meta": {
                "epoch": epoch,
                "total_epochs": epochs,
                "loss": eval_metric,
                "best_loss": min(best_loss, eval_metric),
                "device": str(device),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "history": history,
        }

        # Save latest
        torch.save(checkpoint_data, str(ckpt_p / "latest_model.pt"))

        if is_best:
            best_loss = eval_metric
            torch.save(checkpoint_data, str(ckpt_p / "best_model.pt"))

    total_training_time = time.time() - start_time

    # Save summary metadata
    result_meta = {
        "status": "interrupted" if interrupted else "completed",
        "epochs_completed": len(history),
        "total_epochs": epochs,
        "final_loss": history[-1]["train_loss"] if history else None,
        "best_loss": round(best_loss, 4) if best_loss != float("inf") else None,
        "training_time_seconds": round(total_training_time, 2),
        "device": str(device),
        "checkpoint_dir": str(ckpt_p.resolve()),
        "history": history,
    }

    with open(ckpt_p / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(result_meta, f, indent=2)

    return result_meta
