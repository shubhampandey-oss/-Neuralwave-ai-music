"""
Command-Line Interface for Training NeuralWave AI.
Usage:
    python scripts/train.py --data-dir data/midi --epochs 10 --batch-size 32 --lr 0.002
"""

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.preprocessing import preprocess_midi_dataset
from src.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train NeuralWave AI LSTM model on MIDI dataset.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing raw .mid/.midi files")
    parser.add_argument("--preprocessed-dir", type=str, default=None, help="Directory to save/load preprocessed tensors")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store model checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate for AdamW optimizer")
    parser.add_argument("--seq-len", type=int, default=None, help="Input sequence length in notes")
    parser.add_argument("--device", type=str, default=None, help="Target execution device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing and use existing dataset.pt")
    args = parser.parse_args()

    cfg = load_config()

    data_dir = Path(args.data_dir or cfg["dataset"]["data_dir"])
    preprocessed_dir = Path(args.preprocessed_dir or cfg["dataset"]["preprocessed_dir"])
    ckpt_dir = Path(args.checkpoint_dir or cfg["training"]["checkpoint_dir"])
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    seq_len = args.seq_len or cfg["dataset"]["sequence_length"]
    device = args.device or cfg["training"]["device"]
    seed = args.seed or cfg["training"]["seed"]

    print("=" * 60)
    print("  NEURALWAVE AI — MODEL TRAINING ENGINE")
    print("=" * 60)
    print(f"Data directory:         {data_dir}")
    print(f"Preprocessed directory: {preprocessed_dir}")
    print(f"Checkpoint directory:   {ckpt_dir}")
    print(f"Epochs:                 {epochs}")
    print(f"Batch size:             {batch_size}")
    print(f"Learning rate:          {lr}")
    print(f"Sequence length:        {seq_len}")
    print(f"Device:                 {device}")
    print("-" * 60)

    # Preprocessing stage
    if not args.skip_preprocess or not (preprocessed_dir / "dataset.pt").exists():
        print("[1/2] Preprocessing MIDI files...")
        prep_meta = preprocess_midi_dataset(
            data_dir=data_dir,
            output_dir=preprocessed_dir,
            sequence_length=seq_len,
        )
        print(f"      Extracted {prep_meta['total_notes_extracted']} notes across {prep_meta['valid_files']} valid files.")
        print(f"      Built vocabulary with {prep_meta['vocab_size']} unique tokens, {prep_meta['sequence_count']} sequences.")
    else:
        print("[1/2] Using existing preprocessed dataset.pt")

    # Training stage
    print(f"[2/2] Commencing PyTorch LSTM training for {epochs} epochs...")
    results = train_model(
        dataset_dir=preprocessed_dir,
        checkpoint_dir=ckpt_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        device_name=device,
        seed=seed,
    )

    print("-" * 60)
    print(f"[SUCCESS] Training {results['status']}!")
    print(f"Final recorded loss: {results['final_loss']}")
    print(f"Best recorded loss:  {results['best_loss']}")
    print(f"Total time elapsed:  {results['training_time_seconds']} seconds")
    print(f"Best checkpoint:     {ckpt_dir / 'best_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
