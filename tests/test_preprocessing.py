"""
Automated unit tests for MIDI discovery, note parsing, and dataset preprocessing.
"""

import pytest
from pathlib import Path
import tempfile
import torch

from src.midi_utils import create_synthetic_midi, parse_midi_file
from src.preprocessing import discover_midi_files, preprocess_midi_dataset, load_preprocessed_dataset


def test_discover_midi_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        # Create dummy midi files
        f1 = tmp_p / "song1.mid"
        f2 = tmp_p / "sub" / "song2.midi"
        f2.parent.mkdir(parents=True, exist_ok=True)
        txt = tmp_p / "ignore.txt"

        f1.touch()
        f2.touch()
        txt.touch()

        discovered = discover_midi_files(tmp_p)
        assert len(discovered) == 2
        file_names = [f.name for f in discovered]
        assert "song1.mid" in file_names
        assert "song2.midi" in file_names


def test_parse_synthetic_midi():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        synth_file = tmp_p / "synth.mid"
        create_synthetic_midi(synth_file, bpm=120)

        tokens, meta = parse_midi_file(synth_file, min_notes=4)
        assert len(tokens) > 10
        assert meta["token_count"] == len(tokens)
        assert meta["min_pitch"] is not None
        assert meta["max_pitch"] is not None
        assert "C4" in tokens


def test_preprocess_dataset_preserves_song_boundaries():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        data_dir = tmp_p / "raw"
        out_dir = tmp_p / "preprocessed"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create two distinct synthetic files
        create_synthetic_midi(data_dir / "melody1.mid", title="Track 1")
        create_synthetic_midi(data_dir / "melody2.mid", title="Track 2")

        meta = preprocess_midi_dataset(
            data_dir=data_dir,
            output_dir=out_dir,
            sequence_length=8,
            min_notes_per_file=4,
        )

        assert meta["valid_files"] == 2
        assert meta["vocab_size"] >= 4
        assert (out_dir / "dataset.pt").exists()
        assert (out_dir / "vocab.json").exists()

        loaded = load_preprocessed_dataset(out_dir)
        assert "inputs" in loaded
        assert "targets" in loaded
        assert isinstance(loaded["inputs"], torch.Tensor)
        assert loaded["inputs"].shape[1] == 8
