"""
Storage and composition library manager for NeuralWave AI.
Manages metadata JSON files, MIDI files, audio renderings, and safety resets.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os
import time
from .midi_utils import validate_midi_file


def get_outputs_dir(custom_path: Optional[str | Path] = None) -> Path:
    """Returns resolved outputs directory."""
    if custom_path:
        p = Path(custom_path)
    else:
        p = Path(__file__).resolve().parent.parent / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_compositions(output_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """
    Returns a sorted list of all compositions stored in the outputs directory.
    Gracefully handles missing or malformed metadata.
    """
    out_p = get_outputs_dir(output_dir)
    compositions: List[Dict[str, Any]] = []

    # Check for metadata JSON files first
    json_files = list(out_p.glob("*.json"))
    seen_bases = set()

    for jf in json_files:
        base_name = jf.stem
        seen_bases.add(base_name)
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify MIDI exists
            midi_path = out_p / f"{base_name}.mid"
            wav_path = out_p / f"{base_name}.wav"
            data["id"] = base_name
            data["midi_exists"] = midi_path.exists()
            data["midi_path"] = str(midi_path) if midi_path.exists() else None
            data["audio_exists"] = wav_path.exists()
            data["audio_path"] = str(wav_path) if wav_path.exists() else None
            compositions.append(data)
        except Exception:
            # Corrupted json, continue
            pass

    # Check for orphan .mid files without JSON
    for mid_file in out_p.glob("*.mid"):
        if mid_file.stem not in seen_bases:
            # Construct synthetic record
            val = validate_midi_file(mid_file)
            wav_path = out_p / f"{mid_file.stem}.wav"
            record = {
                "id": mid_file.stem,
                "title": mid_file.stem.replace("_", " ").title(),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mid_file.stat().st_mtime)),
                "midi_exists": True,
                "midi_path": str(mid_file),
                "audio_exists": wav_path.exists(),
                "audio_path": str(wav_path) if wav_path.exists() else None,
                "num_notes": val.get("note_count", 0),
                "duration_seconds": val.get("duration_seconds", 0.0),
                "bpm": val.get("tempo", 120),
                "instrument": "Standard",
                "valid": val.get("valid", True),
            }
            compositions.append(record)

    # Sort descending by created_at or modified time
    compositions.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
    return compositions


def save_composition_record(
    comp_id: str,
    metadata: Dict[str, Any],
    output_dir: Optional[str | Path] = None,
) -> Path:
    """Saves composition metadata to disk."""
    out_p = get_outputs_dir(output_dir)
    json_path = out_p / f"{comp_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return json_path


def rename_composition(
    comp_id: str,
    new_title: str,
    output_dir: Optional[str | Path] = None,
) -> bool:
    """Updates the title field in a composition's metadata."""
    out_p = get_outputs_dir(output_dir)
    json_path = out_p / f"{comp_id}.json"
    if not json_path.exists():
        # Create minimal json record if missing
        record = {"title": new_title, "id": comp_id, "renamed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        return True

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = new_title
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def delete_composition(
    comp_id: str,
    output_dir: Optional[str | Path] = None,
) -> bool:
    """Deletes MIDI, WAV, and JSON associated with a composition ID."""
    out_p = get_outputs_dir(output_dir)
    deleted_any = False
    for ext in [".json", ".mid", ".wav"]:
        target = out_p / f"{comp_id}{ext}"
        if target.exists():
            try:
                target.unlink()
                deleted_any = True
            except Exception:
                pass
    return deleted_any


def reset_generated_artifacts(output_dir: Optional[str | Path] = None, confirm: bool = False) -> int:
    """
    Safely clears generated compositions without touching raw datasets or model weights.
    Requires explicit confirmation.
    """
    if not confirm:
        return 0

    out_p = get_outputs_dir(output_dir)
    count = 0
    for item in out_p.iterdir():
        if item.name == "README.md":
            continue
        if item.is_file() and item.suffix.lower() in [".mid", ".wav", ".json"]:
            try:
                item.unlink()
                count += 1
            except Exception:
                pass
    return count
