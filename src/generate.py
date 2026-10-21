"""
Autoregressive musical sequence generation using trained NeuralWaveLSTM.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import random
import time
import torch
import torch.nn.functional as F

from .model import NeuralWaveLSTM, resolve_device
from .midi_utils import export_midi, validate_midi_file, extract_piano_roll_data


def load_model_checkpoint(checkpoint_path: str | Path, device_name: str = "auto") -> Tuple[Any, Any, Any, Any, Any]:
    """
    Loads model checkpoint and reconstructs architecture.
    """
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")

    device = resolve_device(device_name)
    data = torch.load(str(ckpt_p), map_location=device, weights_only=False)

    cfg = data["config"]
    vocab_to_idx = data["vocab_to_idx"]
    idx_to_vocab = data["idx_to_vocab"]

    model = NeuralWaveLSTM(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(data["model_state_dict"])
    model.eval()

    return model, vocab_to_idx, idx_to_vocab, cfg, device


# Type alias for return
Tuple_Loaded = Any


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.85,
    top_k: int = 40,
) -> int:
    """
    Samples next token from logits with temperature and top-k filtering.
    """
    temp = max(0.05, float(temperature))
    scaled_logits = logits / temp

    if top_k > 0 and top_k < scaled_logits.size(-1):
        top_values, _ = torch.topk(scaled_logits, top_k)
        min_top = top_values[:, -1].unsqueeze(-1)
        scaled_logits = torch.where(scaled_logits < min_top, torch.tensor(-1e9).to(scaled_logits.device), scaled_logits)

    probs = F.softmax(scaled_logits, dim=-1)
    # Multinomial sampling
    sampled_idx = torch.multinomial(probs, num_samples=1).item()
    return int(sampled_idx)


def generate_music_sequence(
    checkpoint_path: str | Path,
    num_notes: int = 64,
    temperature: float = 0.85,
    seed_token: Optional[str] = None,
    random_seed: Optional[int] = None,
    device_name: str = "auto",
) -> List[str]:
    """
    Generates a list of musical tokens autoregressively from a trained checkpoint.
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    model, vocab_to_idx, idx_to_vocab, cfg, device = load_model_checkpoint(checkpoint_path, device_name)
    seq_len = cfg.get("sequence_length", 32)

    # Prepare seed sequence
    # Filter out special tokens for starting seed
    valid_keys = [k for k in vocab_to_idx.keys() if k not in ["<PAD>", "<UNK>", "REST"]]
    if not valid_keys:
        valid_keys = [k for k in vocab_to_idx.keys() if k != "<PAD>"]

    if seed_token and seed_token in vocab_to_idx:
        start_id = vocab_to_idx[seed_token]
    else:
        # Pick a random melodically valid note from the vocabulary
        start_id = vocab_to_idx[random.choice(valid_keys)]

    # Pad seed to seq_len
    current_tokens = [vocab_to_idx.get("<PAD>", 0)] * (seq_len - 1) + [start_id]
    generated_tokens: List[str] = [idx_to_vocab.get(start_id, "C4")]

    with torch.inference_mode():
        for _ in range(num_notes - 1):
            input_tensor = torch.tensor([current_tokens[-seq_len:]], dtype=torch.long, device=device)
            logits, _ = model(input_tensor)

            # Prevent predicting PAD or UNK
            logits[0, vocab_to_idx.get("<PAD>", 0)] = -1e9
            logits[0, vocab_to_idx.get("<UNK>", 1)] = -1e9

            next_id = sample_next_token(logits, temperature=temperature, top_k=25)
            next_token_str = idx_to_vocab.get(next_id, "C4")

            current_tokens.append(next_id)
            generated_tokens.append(next_token_str)

    return generated_tokens


def generate_composition(
    checkpoint_path: str | Path,
    output_path: str | Path,
    num_notes: int = 64,
    temperature: float = 0.85,
    bpm: int = 120,
    instrument: str = "Acoustic Grand Piano",
    quarter_length: float = 0.5,
    seed_token: Optional[str] = None,
    random_seed: Optional[int] = None,
    title: str = "NeuralWave Composition",
    device_name: str = "auto",
) -> Dict[str, Any]:
    """
    End-to-end generation: runs LSTM, produces token sequence, exports MIDI, and validates file.
    """
    tokens = generate_music_sequence(
        checkpoint_path=checkpoint_path,
        num_notes=num_notes,
        temperature=temperature,
        seed_token=seed_token,
        random_seed=random_seed,
        device_name=device_name,
    )

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Export to MIDI
    exported_midi = export_midi(
        tokens=tokens,
        output_path=out_p,
        bpm=bpm,
        quarter_length=quarter_length,
        instrument_name=instrument,
    )

    # Validate output MIDI
    validation = validate_midi_file(exported_midi)
    piano_roll = extract_piano_roll_data(tokens, quarter_length=quarter_length)

    meta = {
        "title": title,
        "filename": exported_midi.name,
        "midi_path": str(exported_midi.resolve()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_notes": len(tokens),
        "bpm": bpm,
        "instrument": instrument,
        "temperature": temperature,
        "seed_token": seed_token,
        "random_seed": random_seed,
        "duration_seconds": validation.get("duration_seconds", 0.0),
        "file_size_bytes": validation.get("size_bytes", 0),
        "valid": validation.get("valid", False),
        "sample_tokens": tokens[:12],
    }

    return {
        "tokens": tokens,
        "midi_path": str(exported_midi.resolve()),
        "metadata": meta,
        "validation": validation,
        "piano_roll": piano_roll,
    }
