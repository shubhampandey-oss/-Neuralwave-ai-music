"""
PyTorch Dataset abstraction for musical note sequences.
"""

from typing import Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from .preprocessing import load_preprocessed_dataset


class MusicSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping note sequence tensors.
    """
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    val_split: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, Any]]:
    """
    Loads preprocessed dataset and returns (train_loader, val_loader, metadata_dict).
    """
    raw_data = load_preprocessed_dataset(dataset_dir)
    inputs = raw_data["inputs"]
    targets = raw_data["targets"]
    vocab_to_idx = raw_data["vocab_to_idx"]
    idx_to_vocab = raw_data["idx_to_vocab"]
    metadata = raw_data["metadata"]

    dataset = MusicSequenceDataset(inputs, targets)

    if val_split > 0 and len(dataset) > 10:
        val_size = max(1, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None

    context = {
        "vocab_to_idx": vocab_to_idx,
        "idx_to_vocab": idx_to_vocab,
        "vocab_size": len(vocab_to_idx),
        "sequence_length": metadata.get("sequence_length", inputs.shape[1]),
        "total_samples": len(dataset),
        "metadata": metadata,
    }

    return train_loader, val_loader, context
