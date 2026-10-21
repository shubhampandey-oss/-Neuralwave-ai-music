"""
Automated unit tests for MIDI stream reconstruction, export, and validation.
"""

import pytest
from pathlib import Path
import tempfile
import music21 as m21

from src.midi_utils import tokens_to_stream, export_midi, validate_midi_file, extract_piano_roll_data


def test_tokens_to_stream_creation():
    tokens = ["C4", "E4", "G4", "C4.E4.G4", "REST", "A4"]
    stream = tokens_to_stream(tokens, bpm=130, quarter_length=0.5, instrument_name="Violin")
    
    assert stream is not None
    flattened = stream.flatten()
    # Check tempo
    metronome = flattened.getElementsByClass(m21.tempo.MetronomeMark)
    assert len(metronome) > 0
    assert metronome[0].number == 130


def test_export_and_reopen_midi():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        out_mid = tmp_p / "test_out.mid"
        tokens = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

        export_midi(tokens, out_mid, bpm=120, quarter_length=0.5)
        assert out_mid.exists()
        assert out_mid.stat().st_size > 0

        val = validate_midi_file(out_mid)
        assert val["valid"] is True
        assert val["note_count"] == 8
        assert val["tempo"] == 120
        assert val["duration_seconds"] > 0


def test_extract_piano_roll_data():
    tokens = ["C4", "E4", "C4.E4.G4", "REST"]
    roll = extract_piano_roll_data(tokens, quarter_length=0.5)

    # C4 (1), E4 (1), chord C4.E4.G4 (3 notes) = 5 note events
    assert len(roll) == 5
    assert all("midi_pitch" in item for item in roll)
    assert all("start_time" in item for item in roll)
