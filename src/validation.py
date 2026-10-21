"""
System environment validation, dependency audits, and checkpoint verification.
"""

from pathlib import Path
from typing import Dict, Any, List
import sys
import os
import torch
import music21

from .audio_utils import get_audio_renderer_status


def get_system_environment_info() -> Dict[str, Any]:
    """
    Returns full diagnostic information about Python, PyTorch, hardware, and audio engines.
    """
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"
    renderer_info = get_audio_renderer_status()

    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "music21_version": music21.__version__,
        "cuda_available": cuda_avail,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "device_name": device_name,
        "os_platform": sys.platform,
        "audio_renderer": renderer_info,
    }


def validate_directories(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Checks that core application directories exist and are writable.
    """
    target_dirs = {
        "data_midi": base_dir / "data" / "midi",
        "data_preprocessed": base_dir / "data" / "preprocessed",
        "models": base_dir / "models",
        "outputs": base_dir / "outputs",
    }

    status: Dict[str, Dict[str, Any]] = {}
    for name, p in target_dirs.items():
        p.mkdir(parents=True, exist_ok=True)
        writable = os.access(p, os.W_OK)
        file_count = len(list(p.glob("*")))
        status[name] = {
            "path": str(p),
            "exists": p.exists(),
            "writable": writable,
            "file_count": file_count,
        }
    return status


def validate_model_checkpoint(checkpoint_path: str | Path) -> Dict[str, Any]:
    """
    Inspects and validates a saved PyTorch checkpoint.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        return {"valid": False, "error": f"Checkpoint not found at: {p}"}

    try:
        data = torch.load(str(p), map_location="cpu", weights_only=False)
        required_keys = ["model_state_dict", "vocab_to_idx", "idx_to_vocab", "config"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            return {
                "valid": False,
                "error": f"Corrupted checkpoint: missing required components {missing_keys}",
            }

        cfg = data["config"]
        meta = data.get("training_meta", {})
        history = data.get("history", [])

        return {
            "valid": True,
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "vocab_size": cfg.get("vocab_size", 0),
            "embedding_dim": cfg.get("embedding_dim", 0),
            "hidden_dim": cfg.get("hidden_dim", 0),
            "num_layers": cfg.get("num_layers", 0),
            "sequence_length": cfg.get("sequence_length", 32),
            "epochs_recorded": meta.get("epoch", len(history)),
            "recorded_loss": meta.get("loss"),
            "device": meta.get("device", "cpu"),
            "timestamp": meta.get("timestamp"),
            "history_length": len(history),
        }
    except Exception as e:
        return {"valid": False, "error": f"Failed to load checkpoint: {str(e)}"}
