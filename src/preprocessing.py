"""
MIDI dataset preprocessing module for NeuralWave AI.
Discovers, validates, tokenizes MIDI files, constructs training sequences,
and saves vocabulary and sequence tensors.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import torch
from .midi_utils import parse_midi_file


def discover_midi_files(directory: str | Path) -> List[Path]:
    """Recursively finds all .mid and .midi files in the directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    midi_files: List[Path] = []
    for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        midi_files.extend(dir_path.rglob(ext))

    # Deduplicate and sort for deterministic behavior
    return sorted(list(set(midi_files)))


def preprocess_midi_dataset(
    data_dir: str | Path,
    output_dir: str | Path,
    sequence_length: int = 32,
    min_notes_per_file: int = 8,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Parses MIDI files into token sequences without crossing song boundaries,
    builds vocabulary, and saves preprocessed tensors.
    """
    raw_files = discover_midi_files(data_dir)
    if max_files and len(raw_files) > max_files:
        raw_files = raw_files[:max_files]

    if not raw_files:
        raise ValueError(f"No MIDI files found in directory: {data_dir}")

    all_songs_tokens: List[List[str]] = []
    valid_files_meta: List[Dict[str, Any]] = []
    skipped_files: List[Dict[str, str]] = []

    # Parse each file independently
    for file_p in raw_files:
        try:
            tokens, meta = parse_midi_file(file_p, min_notes=min_notes_per_file)
            all_songs_tokens.append(tokens)
            valid_files_meta.append(meta)
        except Exception as e:
            skipped_files.append({"filename": file_p.name, "error": str(e)})

    if not all_songs_tokens:
        raise ValueError(
            f"Failed to extract valid note sequences from {len(raw_files)} discovered files. "
            f"Errors: {[s['error'] for s in skipped_files[:3]]}"
        )

    # Build unique vocabulary
    unique_tokens = set()
    for song_toks in all_songs_tokens:
        unique_tokens.update(song_toks)

    sorted_tokens = sorted(list(unique_tokens))
    # Special tokens: 0: <PAD>, 1: <UNK>
    vocab_to_idx: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for idx, tok in enumerate(sorted_tokens, start=2):
        vocab_to_idx[tok] = idx

    idx_to_vocab: Dict[int, str] = {v: k for k, v in vocab_to_idx.items()}

    # Create sequence windows (X -> seq_len, y -> next_token)
    # Strictly respect song boundaries!
    input_sequences: List[List[int]] = []
    target_tokens: List[int] = []

    for song_toks in all_songs_tokens:
        int_song = [vocab_to_idx.get(t, vocab_to_idx["<UNK>"]) for t in song_toks]
        if len(int_song) <= sequence_length:
            # If song is shorter than or equal to seq_len, pad left
            pad_count = sequence_length - len(int_song) + 1
            padded = [vocab_to_idx["<PAD>"]] * pad_count + int_song
            input_sequences.append(padded[:sequence_length])
            target_tokens.append(padded[sequence_length])
        else:
            for i in range(len(int_song) - sequence_length):
                seq_in = int_song[i : i + sequence_length]
                seq_out = int_song[i + sequence_length]
                input_sequences.append(seq_in)
                target_tokens.append(seq_out)

    if not input_sequences:
        raise ValueError("No training sequences could be generated with the given sequence length.")

    # Convert to PyTorch tensors
    inputs_tensor = torch.tensor(input_sequences, dtype=torch.long)
    targets_tensor = torch.tensor(target_tokens, dtype=torch.long)

    # Save to output directory
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    summary_meta = {
        "discovered_files": len(raw_files),
        "valid_files": len(valid_files_meta),
        "skipped_files": len(skipped_files),
        "total_notes_extracted": sum(m["token_count"] for m in valid_files_meta),
        "vocab_size": len(vocab_to_idx),
        "sequence_count": len(input_sequences),
        "sequence_length": sequence_length,
        "sample_tokens": sorted_tokens[:10],
    }

    payload = {
        "inputs": inputs_tensor,
        "targets": targets_tensor,
        "vocab_to_idx": vocab_to_idx,
        "idx_to_vocab": idx_to_vocab,
        "metadata": summary_meta,
    }

    dataset_path = out_dir_path / "dataset.pt"
    torch.save(payload, str(dataset_path))

    # Save human-readable vocab and metadata
    with open(out_dir_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_to_idx, f, indent=2)

    with open(out_dir_path / "preprocessing_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_meta,
            "valid_files": valid_files_meta,
            "skipped_files": skipped_files,
        }, f, indent=2)

    return summary_meta


def load_preprocessed_dataset(dataset_dir: str | Path) -> Dict[str, Any]:
    """Loads preprocessed dataset.pt from disk."""
    pt_path = Path(dataset_dir) / "dataset.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found at: {pt_path}")

    # Load with PyTorch
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    return data
