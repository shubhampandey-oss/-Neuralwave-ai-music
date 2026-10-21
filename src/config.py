"""
Configuration management module for NeuralWave AI.
Reads config.yaml and provides typed access and override utilities.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "name": "NeuralWave AI",
        "subtitle": "AI Music Generation Studio",
        "version": "1.0.0",
        "description": "PyTorch-powered LSTM music generation studio for MIDI composition",
    },
    "dataset": {
        "data_dir": "data/midi",
        "preprocessed_dir": "data/preprocessed",
        "sequence_length": 32,
        "min_notes_per_file": 8,
        "max_files": 150,
        "save_format": "pt",
    },
    "model": {
        "name": "NeuralWaveLSTM",
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.2,
    },
    "training": {
        "learning_rate": 0.002,
        "batch_size": 32,
        "epochs": 10,
        "checkpoint_dir": "models",
        "checkpoint_interval": 2,
        "device": "auto",
        "seed": 42,
        "validation_split": 0.1,
    },
    "generation": {
        "num_notes": 64,
        "temperature": 0.85,
        "bpm": 120,
        "instrument": "Acoustic Grand Piano",
        "quarter_length": 0.5,
        "output_dir": "outputs",
        "seed": 42,
    },
    "audio": {
        "soundfont_path": "",
        "fluidsynth_path": "",
        "sample_rate": 44100,
    },
}


def find_project_root() -> Path:
    """Finds the root directory of the project."""
    current = Path(__file__).resolve().parent
    # Check parent dirs for config.yaml or app.py
    for p in [current, current.parent, current.parent.parent]:
        if (p / "config.yaml").exists() or (p / "app.py").exists():
            return p
    return current.parent


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges override dictionary into base."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Loads configuration from YAML file with fallback to defaults."""
    root = find_project_root()
    if config_path is None:
        target_path = root / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    config = dict(DEFAULT_CONFIG)
    if target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
                if isinstance(user_cfg, dict):
                    config = deep_merge(config, user_cfg)
        except Exception as e:
            print(f"[Warning] Failed to parse config file at {target_path}: {e}. Using defaults.")

    # Resolve paths relative to project root
    for key in ["data_dir", "preprocessed_dir"]:
        path_str = config["dataset"][key]
        if not Path(path_str).is_absolute():
            config["dataset"][key] = str((root / path_str).resolve())

    ckpt_dir = config["training"]["checkpoint_dir"]
    if not Path(ckpt_dir).is_absolute():
        config["training"]["checkpoint_dir"] = str((root / ckpt_dir).resolve())

    out_dir = config["generation"]["output_dir"]
    if not Path(out_dir).is_absolute():
        config["generation"]["output_dir"] = str((root / out_dir).resolve())

    return config
