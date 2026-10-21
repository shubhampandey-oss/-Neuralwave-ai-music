"""
MIDI parsing, sequence extraction, validation, and file synthesis utilities.
Powered by music21 with fallback support.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import os
import music21 as m21


# General MIDI Program Mapping for Instrument Selection
INSTRUMENT_MAP: Dict[str, Any] = {
    "Acoustic Grand Piano": m21.instrument.Piano,
    "Electric Piano": m21.instrument.ElectricPiano,
    "Harpsichord": m21.instrument.Harpsichord,
    "Violin": m21.instrument.Violin,
    "Cello": m21.instrument.Violoncello,
    "Acoustic Guitar": m21.instrument.AcousticGuitar,
    "Electric Guitar": m21.instrument.ElectricGuitar,
    "Electric Bass": m21.instrument.ElectricBass,
    "Flute": m21.instrument.Flute,
    "Saxophone": m21.instrument.AltoSaxophone,
    "Trumpet": m21.instrument.Trumpet,
    "Synthesizer": m21.instrument.ElectricBass,  # fallback lead
}


def parse_midi_file(file_path: str | Path, min_notes: int = 4) -> Tuple[List[str], Dict[str, Any]]:
    """
    Parses a MIDI file and extracts sequential note/chord/rest tokens.
    Returns (tokens, metadata).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    try:
        score = m21.converter.parse(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse MIDI file {path.name}: {e}")

    tokens: List[str] = []
    pitches_observed: List[int] = []

    # Flatten score to iterate across all notes and rests
    try:
        elements = score.flatten().notesAndRests
    except Exception:
        # Fallback for simple streams
        elements = score.notesAndRests

    for element in elements:
        if isinstance(element, m21.note.Note):
            tok = str(element.pitch)
            tokens.append(tok)
            pitches_observed.append(element.pitch.midi)
        elif isinstance(element, m21.chord.Chord):
            # Sort pitches to maintain deterministic token naming
            chord_pitches = sorted(element.pitches, key=lambda p: p.midi)
            tok = ".".join(str(p) for p in chord_pitches)
            tokens.append(tok)
            pitches_observed.extend(p.midi for p in chord_pitches)
        elif isinstance(element, m21.note.Rest):
            tokens.append("REST")

    if len(tokens) < min_notes:
        raise ValueError(f"File {path.name} contains too few musical elements ({len(tokens)} < {min_notes})")

    meta = {
        "filename": path.name,
        "token_count": len(tokens),
        "min_pitch": min(pitches_observed) if pitches_observed else None,
        "max_pitch": max(pitches_observed) if pitches_observed else None,
        "unique_tokens": len(set(tokens)),
    }
    return tokens, meta


def tokens_to_stream(
    tokens: List[str],
    bpm: int = 120,
    quarter_length: float = 0.5,
    instrument_name: str = "Acoustic Grand Piano",
) -> m21.stream.Stream:
    """
    Converts a list of symbolic note/chord tokens into a music21 Stream.
    """
    output_stream = m21.stream.Stream()

    # Set tempo
    tempo_mark = m21.tempo.MetronomeMark(number=max(40, min(240, bpm)))
    output_stream.append(tempo_mark)

    # Set instrument
    inst_cls = INSTRUMENT_MAP.get(instrument_name, m21.instrument.Piano)
    try:
        inst = inst_cls()
        output_stream.append(inst)
    except Exception:
        output_stream.append(m21.instrument.Piano())

    current_offset = 0.0

    for tok in tokens:
        tok_str = str(tok).strip()
        if not tok_str or tok_str in ["<PAD>", "<UNK>"]:
            continue

        try:
            if tok_str == "REST":
                rest = m21.note.Rest()
                rest.quarterLength = quarter_length
                rest.offset = current_offset
                output_stream.append(rest)
            elif "." in tok_str:
                # Chord representation
                pitch_names = tok_str.split(".")
                chord_notes = []
                for p_name in pitch_names:
                    try:
                        chord_notes.append(m21.note.Note(p_name))
                    except Exception:
                        pass
                if chord_notes:
                    new_chord = m21.chord.Chord(chord_notes)
                    new_chord.quarterLength = quarter_length
                    new_chord.offset = current_offset
                    output_stream.append(new_chord)
            else:
                # Single note representation
                new_note = m21.note.Note(tok_str)
                new_note.quarterLength = quarter_length
                new_note.offset = current_offset
                output_stream.append(new_note)
        except Exception:
            # Skip invalid token without crashing the stream builder
            continue

        current_offset += quarter_length

    return output_stream


def export_midi(
    tokens: List[str],
    output_path: str | Path,
    bpm: int = 120,
    quarter_length: float = 0.5,
    instrument_name: str = "Acoustic Grand Piano",
) -> Path:
    """
    Exports a token sequence to a valid MIDI file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    stream = tokens_to_stream(tokens, bpm=bpm, quarter_length=quarter_length, instrument_name=instrument_name)
    stream.write("midi", fp=str(out_p))

    # Verification: check file exists and is non-empty
    if not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError(f"Export failed: {out_p} is empty or was not created.")

    return out_p


_MIDI_VALIDATION_CACHE: Dict[str, Dict[str, Any]] = {}


def validate_midi_file(file_path: str | Path) -> Dict[str, Any]:
    """
    Opens and validates a MIDI file, reporting duration, notes, and validity.
    Cached in-memory to prevent repeated music21 parsing on Streamlit reruns.
    """
    path = Path(file_path)
    if not path.exists():
        return {"valid": False, "error": "File does not exist"}
    if path.stat().st_size == 0:
        return {"valid": False, "error": "File is 0 bytes"}

    cache_key = f"{str(path.resolve())}_{path.stat().st_mtime}"
    if cache_key in _MIDI_VALIDATION_CACHE:
        return _MIDI_VALIDATION_CACHE[cache_key]

    try:
        score = m21.converter.parse(str(path))
        notes_and_rests = score.flatten().notesAndRests
        note_count = len([e for e in notes_and_rests if isinstance(e, (m21.note.Note, m21.chord.Chord))])
        rest_count = len([e for e in notes_and_rests if isinstance(e, m21.note.Rest)])
        duration_quarters = float(score.duration.quarterLength)
        
        # Determine tempo
        tempo = 120
        metronome_marks = score.flatten().getElementsByClass(m21.tempo.MetronomeMark)
        if len(metronome_marks) > 0:
            tempo = int(metronome_marks[0].number)

        duration_seconds = (duration_quarters / tempo) * 60.0 if tempo > 0 else 0.0

        res = {
            "valid": True,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "note_count": note_count,
            "rest_count": rest_count,
            "duration_quarters": duration_quarters,
            "tempo": tempo,
            "duration_seconds": round(duration_seconds, 2),
        }
        _MIDI_VALIDATION_CACHE[cache_key] = res
        return res
    except Exception as e:
        return {"valid": False, "error": str(e)}


def create_synthetic_midi(
    output_path: str | Path,
    title: str = "Synthetic Test Melody",
    bpm: int = 120,
) -> Path:
    """
    Creates a small, fully valid synthetic MIDI file for testing purposes.
    Explicitly labeled as synthetic test data.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    stream = m21.stream.Stream()
    stream.append(m21.tempo.MetronomeMark(number=bpm))
    stream.append(m21.instrument.Piano())

    # C Major melodic sequence: C4, D4, E4, F4, G4, A4, B4, C5, followed by chords
    pitches = ["C4", "E4", "G4", "B4", "C5", "G4", "E4", "C4",
               "D4", "F4", "A4", "C5", "D5", "A4", "F4", "D4",
               "E4", "G4", "B4", "D5", "E5", "B4", "G4", "E4",
               "C4.E4.G4", "D4.F4.A4", "E4.G4.B4", "C4.E4.G4.C5"]

    offset = 0.0
    for p in pitches:
        if "." in p:
            ch_notes = [m21.note.Note(n) for n in p.split(".")]
            item = m21.chord.Chord(ch_notes)
        else:
            item = m21.note.Note(p)
        item.quarterLength = 0.5
        item.offset = offset
        stream.append(item)
        offset += 0.5

    stream.write("midi", fp=str(out_p))
    return out_p


def extract_piano_roll_data(tokens: List[str], quarter_length: float = 0.5) -> List[Dict[str, Any]]:
    """
    Extracts pitch and timing coordinates for interactive piano-roll visualization.
    """
    piano_roll_events: List[Dict[str, Any]] = []
    current_time = 0.0

    for idx, tok in enumerate(tokens):
        tok_str = str(tok).strip()
        if not tok_str or tok_str in ["REST", "<PAD>", "<UNK>"]:
            current_time += quarter_length
            continue

        if "." in tok_str:
            # Chord pitches
            p_names = tok_str.split(".")
            for pn in p_names:
                try:
                    n = m21.note.Note(pn)
                    piano_roll_events.append({
                        "step": idx,
                        "pitch_name": pn,
                        "midi_pitch": n.pitch.midi,
                        "start_time": current_time,
                        "end_time": current_time + quarter_length,
                        "duration": quarter_length,
                        "is_accent": (idx % 4 == 0),
                    })
                except Exception:
                    continue
        else:
            try:
                n = m21.note.Note(tok_str)
                piano_roll_events.append({
                    "step": idx,
                    "pitch_name": tok_str,
                    "midi_pitch": n.pitch.midi,
                    "start_time": current_time,
                    "end_time": current_time + quarter_length,
                    "duration": quarter_length,
                    "is_accent": (idx % 4 == 0),
                })
            except Exception:
                pass

        current_time += quarter_length

    return piano_roll_events
