"""
Audio rendering utilities for converting MIDI to WAV.
Detects FluidSynth and SoundFont availability with honest fallback guidance.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import subprocess
import os


def find_soundfont() -> Optional[Path]:
    """Searches common default system directories for a General MIDI SoundFont (.sf2)."""
    search_locations = [
        # Windows common paths
        Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "FluidSynth" / "soundfonts",
        Path("C:/soundfonts"),
        Path("C:/Windows/System32/drivers"),
        # Linux / Unix common paths
        Path("/usr/share/sounds/sf2"),
        Path("/usr/share/soundfonts"),
        Path("/usr/local/share/soundfonts"),
        # Local project path
        Path(__file__).resolve().parent.parent / "soundfonts",
    ]

    known_sf2_names = [
        "default.sf2",
        "FluidR3_GM.sf2",
        "GeneralUser GS.sf2",
        "TimGM6mb.sf2",
        "soundfont.sf2",
    ]

    for loc in search_locations:
        if loc.exists() and loc.is_dir():
            for name in known_sf2_names:
                candidate = loc / name
                if candidate.exists():
                    return candidate
            # Any .sf2 file
            for f in loc.glob("*.sf2"):
                return f

    return None


_GLOBAL_RENDERER_CACHE: Optional[Dict[str, Any]] = None


def get_audio_renderer_status(custom_soundfont: str = "", custom_fluidsynth: str = "") -> Dict[str, Any]:
    """
    Checks if FluidSynth and a compatible SoundFont are available.
    Caches default check to avoid repeated PATH and disk scans.
    """
    global _GLOBAL_RENDERER_CACHE
    if not custom_soundfont and not custom_fluidsynth and _GLOBAL_RENDERER_CACHE is not None:
        return _GLOBAL_RENDERER_CACHE

    fluidsynth_bin = custom_fluidsynth or shutil.which("fluidsynth")
    
    soundfont_path: Optional[Path] = None
    if custom_soundfont and Path(custom_soundfont).exists():
        soundfont_path = Path(custom_soundfont)
    else:
        soundfont_path = find_soundfont()

    available = (fluidsynth_bin is not None) and (soundfont_path is not None)

    res = {
        "available": bool(available),
        "fluidsynth_installed": fluidsynth_bin is not None,
        "fluidsynth_path": str(fluidsynth_bin) if fluidsynth_bin else None,
        "soundfont_found": soundfont_path is not None,
        "soundfont_path": str(soundfont_path) if soundfont_path else None,
        "instructions": (
            "Direct in-browser WAV rendering requires FluidSynth and a General MIDI SoundFont (.sf2).\n"
            "• Windows: install via 'winget install FluidSynth' or 'choco install fluidsynth'\n"
            "• Ubuntu/Debian: 'sudo apt-get install fluidsynth fluid-soundfont-gm'\n"
            "• macOS: 'brew install fluidsynth' and download a SoundFont (e.g., GeneralUser GS)\n"
            "Without FluidSynth, you can always download and play the standard .mid file directly in "
            "MuseScore, GarageBand, FL Studio, VLC, or Windows Media Player."
        ),
    }

    if not custom_soundfont and not custom_fluidsynth:
        _GLOBAL_RENDERER_CACHE = res

    return res


def render_midi_to_wav(
    midi_path: str | Path,
    wav_output_path: str | Path,
    soundfont_path: Optional[str | Path] = None,
    sample_rate: int = 44100,
) -> Path:
    """
    Renders a MIDI file to WAV using FluidSynth if installed.
    """
    m_p = Path(midi_path)
    w_p = Path(wav_output_path)
    w_p.parent.mkdir(parents=True, exist_ok=True)

    if not m_p.exists():
        raise FileNotFoundError(f"Source MIDI not found: {m_p}")

    status = get_audio_renderer_status(custom_soundfont=str(soundfont_path) if soundfont_path else "")
    if not status["available"]:
        raise RuntimeError(
            "FluidSynth or SoundFont not found. Audio rendering unavailable.\n"
            f"{status['instructions']}"
        )

    cmd = [
        status["fluidsynth_path"],
        "-ni",
        status["soundfont_path"],
        str(m_p),
        "-F",
        str(w_p),
        "-r",
        str(sample_rate),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not w_p.exists():
        raise RuntimeError(f"FluidSynth rendering failed: {result.stderr}")

    return w_p
