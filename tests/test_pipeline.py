"""
Integration test verifying the end-to-end training and autoregressive generation pipeline.
"""

import pytest
from pathlib import Path
import tempfile
import torch

from src.midi_utils import create_synthetic_midi, validate_midi_file
from src.preprocessing import preprocess_midi_dataset
from src.train import train_model
from src.generate import generate_composition
from src.validation import validate_model_checkpoint


def test_full_miniature_pipeline_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        raw_midi_dir = tmp_p / "midi"
        prep_dir = tmp_p / "preprocessed"
        models_dir = tmp_p / "models"
        outputs_dir = tmp_p / "outputs"

        raw_midi_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create synthetic fixture
        synth_mid = raw_midi_dir / "melody.mid"
        create_synthetic_midi(synth_mid, bpm=120)
        assert synth_mid.exists()

        # 2. Preprocess
        meta = preprocess_midi_dataset(
            data_dir=raw_midi_dir,
            output_dir=prep_dir,
            sequence_length=6,
            min_notes_per_file=4,
        )
        assert meta["sequence_count"] > 0
        assert (prep_dir / "dataset.pt").exists()

        # 3. Train for 2 epochs on CPU
        train_res = train_model(
            dataset_dir=prep_dir,
            checkpoint_dir=models_dir,
            epochs=2,
            batch_size=4,
            learning_rate=0.005,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=1,
            dropout=0.1,
            device_name="cpu",
            val_split=0.0,
        )
        assert train_res["status"] == "completed"
        best_pt = models_dir / "best_model.pt"
        assert best_pt.exists()

        # 4. Checkpoint check
        ckpt_val = validate_model_checkpoint(best_pt)
        assert ckpt_val["valid"] is True

        # 5. Autoregressive generation
        gen_mid = outputs_dir / "test_composition.mid"
        gen_res = generate_composition(
            checkpoint_path=best_pt,
            output_path=gen_mid,
            num_notes=20,
            temperature=0.8,
            bpm=120,
            random_seed=42,
            device_name="cpu",
        )

        assert len(gen_res["tokens"]) == 20
        assert gen_mid.exists()

        # 6. Validate exported MIDI
        val_mid = validate_midi_file(gen_mid)
        assert val_mid["valid"] is True
        assert val_mid["note_count"] > 0
        assert val_mid["duration_seconds"] > 0
