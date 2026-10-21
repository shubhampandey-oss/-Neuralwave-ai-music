"""
NeuralWave AI — AI Music Generation Studio
Interactive Streamlit Application with Custom Digital Audio Workstation (DAW) Aesthetics.
"""

from pathlib import Path
import json
import os
import sys
import time
import zipfile
import io

import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.midi_utils import (
    create_synthetic_midi,
    validate_midi_file,
    extract_piano_roll_data,
    INSTRUMENT_MAP,
)
from src.preprocessing import (
    discover_midi_files,
    preprocess_midi_dataset,
    load_preprocessed_dataset,
)
from src.train import train_model
from src.generate import generate_composition, load_model_checkpoint
from src.storage import (
    list_compositions,
    save_composition_record,
    rename_composition,
    delete_composition,
    reset_generated_artifacts,
)
from src.validation import (
    get_system_environment_info,
    validate_directories,
    validate_model_checkpoint,
)
from src.audio_utils import get_audio_renderer_status, render_midi_to_wav

# ---------------------------------------------------------
# Page Configuration & Visual Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuralWave AI — AI Music Generation Studio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* NeuralWave AI Custom DAW Aesthetic */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-main: #080B12;
    --bg-sec: #0D111B;
    --bg-card: #111827;
    --bg-elevated: #151D2C;
    --electric-blue: #1677FF;
    --blue-glow: #36A3FF;
    --neon-red: #FF244F;
    --dark-red: #290D18;
    --text-primary: #F4F7FF;
    --text-secondary: #9AA6BA;
    --border-color: #263247;
    --success: #31D6A0;
    --warning: #FFBF5E;
}

/* Global resets & typography */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Main app container */
.stApp {
    background-color: var(--bg-main) !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: var(--bg-sec) !important;
    border-right: 1px solid var(--border-color) !important;
}

/* Header & Navbar styling */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Card & Elevated Containers */
.daw-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.daw-card:hover {
    border-color: rgba(22, 119, 255, 0.4);
}

.daw-card-accent {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--dark-red) 100%);
    border: 1px solid rgba(255, 36, 79, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.daw-stat-val {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}

.daw-stat-label {
    font-size: 0.82rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-top: 4px;
}

/* Custom Badges */
.badge-blue {
    background: rgba(22, 119, 255, 0.15);
    color: var(--blue-glow);
    border: 1px solid rgba(22, 119, 255, 0.4);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    display: inline-block;
}

.badge-red {
    background: rgba(255, 36, 79, 0.15);
    color: var(--neon-red);
    border: 1px solid rgba(255, 36, 79, 0.4);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    display: inline-block;
}

.badge-success {
    background: rgba(49, 214, 160, 0.15);
    color: var(--success);
    border: 1px solid rgba(49, 214, 160, 0.4);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}

/* Custom Buttons */
.stButton > button {
    background: linear-gradient(180deg, #1677FF 0%, #0F5FD6 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #36A3FF !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 0 15px rgba(22, 119, 255, 0.25) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    box-shadow: 0 0 25px rgba(54, 163, 255, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Neon Red Action Button */
.neon-action-btn .stButton > button {
    background: linear-gradient(180deg, #FF244F 0%, #D8133C 100%) !important;
    border: 1px solid #FF5C7E !important;
    box-shadow: 0 0 20px rgba(255, 36, 79, 0.4) !important;
}

.neon-action-btn .stButton > button:hover {
    box-shadow: 0 0 30px rgba(255, 36, 79, 0.7) !important;
}

/* Form input elements */
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
    background-color: var(--bg-sec) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    color: var(--electric-blue) !important;
}

/* Logo brand banner */
.brand-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 0 24px 0;
}
.brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #FFFFFF 0%, #9AA6BA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-sub {
    font-size: 0.75rem;
    color: var(--neon-red);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.1em;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Web Audio Synthesizer Component (Browser In-Memory Audio)
# ---------------------------------------------------------
def render_web_audio_player(tokens, bpm=120, quarter_length=0.5, key="synth"):
    """
    Renders an HTML5 Web Audio synthesizer in the browser so users can listen
    to any generated composition without needing FluidSynth or external soundfonts!
    """
    events = extract_piano_roll_data(tokens, quarter_length=quarter_length)
    if not events:
        st.info("No musical events to play.")
        return

    # Prepare JSON sequence for JavaScript Web Audio API
    audio_sequence = []
    # Calculate note duration in seconds based on bpm
    seconds_per_quarter = 60.0 / max(40, bpm)
    for ev in events:
        freq = 440.0 * (2.0 ** ((ev["midi_pitch"] - 69) / 12.0))
        audio_sequence.append({
            "pitch": ev["pitch_name"],
            "midi": ev["midi_pitch"],
            "freq": round(freq, 2),
            "start": round(ev["start_time"] * seconds_per_quarter * (1.0 / quarter_length) * 0.5, 3),
            "dur": round(seconds_per_quarter * 0.85, 3),
            "is_accent": ev["is_accent"],
        })

    json_data = json.dumps(audio_sequence)

    html_code = f"""
    <div style="background: #111827; border: 1px solid #263247; border-radius: 10px; padding: 14px; margin-top: 10px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <button id="playBtn_{key}" onclick="startPlayback_{key}()" style="
                    background: linear-gradient(180deg, #1677FF 0%, #0F5FD6 100%);
                    color: white; border: 1px solid #36A3FF; border-radius: 8px;
                    padding: 8px 18px; font-weight: 600; cursor: pointer; font-size: 0.9rem;
                    box-shadow: 0 0 12px rgba(22, 119, 255, 0.4); display: flex; align-items: center; gap: 6px;">
                    <span id="playIcon_{key}">▶</span> Play Composition
                </button>
                <button id="stopBtn_{key}" onclick="stopPlayback_{key}()" style="
                    background: #151D2C; color: #9AA6BA; border: 1px solid #263247;
                    border-radius: 8px; padding: 8px 14px; font-weight: 500; cursor: pointer; font-size: 0.9rem;">
                    ⏹ Stop
                </button>
            </div>
            <div style="display: flex; align-items: center; gap: 15px;">
                <span id="status_{key}" style="font-size: 0.8rem; color: #9AA6BA; font-family: monospace;">Ready • {len(audio_sequence)} notes</span>
                <span style="background: rgba(22, 119, 255, 0.15); color: #36A3FF; border: 1px solid rgba(22, 119, 255, 0.3); border-radius: 6px; padding: 3px 8px; font-size: 0.72rem; font-weight: 600;">
                    WebAudio PolySynth
                </span>
            </div>
        </div>
        <div style="width: 100%; background: #080B12; height: 6px; border-radius: 3px; margin-top: 12px; overflow: hidden; border: 1px solid #263247;">
            <div id="progBar_{key}" style="width: 0%; height: 100%; background: linear-gradient(90deg, #1677FF, #FF244F); transition: width 0.1s linear;"></div>
        </div>
    </div>

    <script>
    (function() {{
        const sequence = {json_data};
        let audioCtx = null;
        let isPlaying = false;
        let scheduledNodes = [];
        let totalDuration = 0;
        let animFrame = null;
        let startTime = 0;

        if (sequence.length > 0) {{
            const lastNote = sequence[sequence.length - 1];
            totalDuration = lastNote.start + lastNote.dur + 0.5;
        }}

        function getAudioContext() {{
            if (!audioCtx) {{
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                audioCtx = new AudioContext();
            }}
            if (audioCtx.state === 'suspended') {{
                audioCtx.resume();
            }}
            return audioCtx;
        }}

        window.startPlayback_{key} = function() {{
            if (isPlaying) {{
                stopPlayback_{key}();
                return;
            }}

            const ctx = getAudioContext();
            isPlaying = true;
            scheduledNodes = [];
            document.getElementById('playIcon_{key}').innerText = '⏸';
            document.getElementById('status_{key}').innerText = 'Playing...';
            document.getElementById('status_{key}').style.color = '#31D6A0';

            const now = ctx.currentTime + 0.05;
            startTime = now;

            sequence.forEach(note => {{
                const noteStart = now + note.start;
                const noteDur = note.dur;

                // Polyphonic Dual-Oscillator Synthesis with Envelope
                const osc1 = ctx.createOscillator();
                const osc2 = ctx.createOscillator();
                const gain = ctx.createGain();

                // Electric piano/acoustic timbre combination
                osc1.type = note.is_accent ? 'triangle' : 'sine';
                osc2.type = 'sine';
                osc1.frequency.setValueAtTime(note.freq, noteStart);
                osc2.frequency.setValueAtTime(note.freq * 2.0, noteStart); // Harmonic overtone

                const peakVol = note.is_accent ? 0.28 : 0.20;

                // Envelope: Attack -> Decay -> Release
                gain.gain.setValueAtTime(0.0001, noteStart);
                gain.gain.exponentialRampToValueAtTime(peakVol, noteStart + 0.015);
                gain.gain.exponentialRampToValueAtTime(peakVol * 0.4, noteStart + 0.08);
                gain.gain.exponentialRampToValueAtTime(0.0001, noteStart + noteDur);

                osc1.connect(gain);
                osc2.connect(gain);
                gain.connect(ctx.destination);

                osc1.start(noteStart);
                osc2.start(noteStart);
                osc1.stop(noteStart + noteDur + 0.05);
                osc2.stop(noteStart + noteDur + 0.05);

                scheduledNodes.push(osc1, osc2, gain);
            }});

            updateProgress_{key}();
        }};

        window.stopPlayback_{key} = function() {{
            isPlaying = false;
            if (audioCtx) {{
                scheduledNodes.forEach(node => {{
                    try {{ node.stop(); }} catch(e) {{}}
                    try {{ node.disconnect(); }} catch(e) {{}}
                }});
            }}
            scheduledNodes = [];
            if (animFrame) cancelAnimationFrame(animFrame);
            document.getElementById('playIcon_{key}').innerText = '▶';
            document.getElementById('status_{key}').innerText = 'Stopped';
            document.getElementById('status_{key}').style.color = '#9AA6BA';
            document.getElementById('progBar_{key}').style.width = '0%';
        }};

        function updateProgress_{key}() {{
            if (!isPlaying || !audioCtx) return;
            const current = audioCtx.currentTime - startTime;
            if (current >= totalDuration) {{
                stopPlayback_{key}();
                return;
            }}
            const pct = Math.min(100, (current / totalDuration) * 100);
            document.getElementById('progBar_{key}').style.width = pct + '%';
            animFrame = requestAnimationFrame(updateProgress_{key});
        }}
    }})();
    </script>
    """
    st.components.v1.html(html_code, height=95)


# ---------------------------------------------------------
# Interactive Piano Roll Visualization
# ---------------------------------------------------------
def render_piano_roll_plot(tokens, quarter_length=0.5):
    """
    Renders a Plotly interactive Piano Roll chart with dark DAW styling.
    """
    events = extract_piano_roll_data(tokens, quarter_length=quarter_length)
    if not events:
        st.info("No notes available for Piano Roll display.")
        return

    # Build Gantt-style rectangular bars for each note
    fig = go.Figure()

    for ev in events:
        color = "#FF244F" if ev["is_accent"] else "#1677FF"
        hover_text = f"Note: <b>{ev['pitch_name']}</b><br>MIDI Pitch: {ev['midi_pitch']}<br>Start: {ev['start_time']}s<br>Duration: {ev['duration']}s"

        fig.add_trace(go.Bar(
            x=[ev["duration"]],
            y=[ev["midi_pitch"]],
            base=[ev["start_time"]],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color="#36A3FF" if not ev["is_accent"] else "#FF5C7E", width=1),
                opacity=0.9,
            ),
            hoverinfo="text",
            hovertext=hover_text,
            showlegend=False,
        ))

    # Calculate bounds & unique pitch labels
    pitches = [e["midi_pitch"] for e in events]
    min_pitch = min(pitches) - 1
    max_pitch = max(pitches) + 1
    max_time = max(e["end_time"] for e in events) + 0.5

    # Musical pitch names for y-axis ticks
    pitch_to_name = {e["midi_pitch"]: e["pitch_name"] for e in events}
    unique_pitches = sorted(list(set(pitches)))
    tick_texts = [f"{pitch_to_name.get(p, str(p))} ({p})" for p in unique_pitches]

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0D111B",
        paper_bgcolor="#111827",
        margin=dict(l=65, r=20, t=35, b=40),
        height=340,
        title=dict(
            text="Interactive Piano-Roll (Electric Blue: Melody | Neon Red: Accent Beats)",
            font=dict(size=12, color="#9AA6BA", family="Plus Jakarta Sans"),
            x=0.01,
        ),
        xaxis=dict(
            title=dict(text="Time Offset (Quarters)", font=dict(color="#9AA6BA", size=11)),
            range=[0, max_time],
            gridcolor="#263247",
            zerolinecolor="#263247",
            tickfont=dict(color="#9AA6BA"),
        ),
        yaxis=dict(
            title=dict(text="Musical Pitch", font=dict(color="#9AA6BA", size=11)),
            range=[min_pitch - 0.5, max_pitch + 0.5],
            gridcolor="#263247",
            zerolinecolor="#263247",
            tickmode="array",
            tickvals=unique_pitches,
            ticktext=tick_texts,
            tickfont=dict(color="#36A3FF", size=10),
        ),
        barmode='overlay',
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# ---------------------------------------------------------
# Sidebar & Global Navigation State
# ---------------------------------------------------------
PAGES = [
    "Dashboard",
    "Dataset Lab",
    "Model Training",
    "Music Generator",
    "My Compositions",
    "Model & Settings",
    "About Project",
]


def init_navigation():
    """Initializes page routing state and backward navigation stack."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dashboard"
    if "page_history" not in st.session_state:
        st.session_state["page_history"] = []


def navigate_to(page_name: str):
    """Pushes current page to history stack and transitions to target page."""
    if page_name in PAGES:
        curr = st.session_state.get("current_page", "Dashboard")
        if curr != page_name:
            if "page_history" not in st.session_state:
                st.session_state["page_history"] = []
            if not st.session_state["page_history"] or st.session_state["page_history"][-1] != curr:
                st.session_state["page_history"].append(curr)
            st.session_state["current_page"] = page_name
        st.rerun()


def navigate_back():
    """Pops the most recent page from history and navigates backward."""
    if "page_history" in st.session_state and st.session_state["page_history"]:
        prev_page = st.session_state["page_history"].pop()
        st.session_state["current_page"] = prev_page
    else:
        st.session_state["current_page"] = "Dashboard"
    st.rerun()


def render_top_navigation():
    """
    Renders a prominent backward navigation button and breadcrumb bar at the top of the studio.
    """
    init_navigation()
    current = st.session_state.get("current_page", "Dashboard")
    history = st.session_state.get("page_history", [])

    col_back, col_trail, col_quick = st.columns([1.6, 3.2, 2.2])

    with col_back:
        has_history = len(history) > 0
        back_target = history[-1] if has_history else ("Dashboard" if current != "Dashboard" else None)
        if back_target:
            if st.button(f"⬅ Back to {back_target}", key="global_top_back_btn", use_container_width=True):
                navigate_back()
        else:
            st.button("⬅ Back (Root)", disabled=True, key="global_top_back_disabled", use_container_width=True)

    with col_trail:
        crumb_items = []
        if history:
            for h in history[-2:]:
                crumb_items.append(f"<span style='color: #9AA6BA;'>{h}</span>")
        crumb_items.append(f"<span style='color: #36A3FF; font-weight: 700;'>{current}</span>")
        trail_html = " <span style='color: #263247;'>/</span> ".join(crumb_items)

        st.markdown(f"""
        <div style="background: #0D111B; border: 1px solid #263247; border-radius: 8px; padding: 7px 12px; font-size: 0.82rem; display: flex; align-items: center; gap: 8px; margin-top: 2px;">
            <span style="color: #FF244F; font-size: 0.85rem;">⚡</span>
            <span style="color: #9AA6BA;">Studio</span> <span style="color: #263247;">/</span> {trail_html}
        </div>
        """, unsafe_allow_html=True)

    with col_quick:
        q_c1, q_c2 = st.columns(2)
        with q_c1:
            if current != "Dashboard":
                if st.button("🎛 Dashboard", key="top_quick_dash", use_container_width=True):
                    navigate_to("Dashboard")
        with q_c2:
            if current != "Music Generator":
                if st.button("⚡ Generator", key="top_quick_gen", use_container_width=True):
                    navigate_to("Music Generator")

    st.markdown("<hr style='border-color: #263247; margin: 12px 0 18px 0;'>", unsafe_allow_html=True)


def render_sidebar():
    init_navigation()
    current_page = st.session_state.get("current_page", "Dashboard")
    if current_page not in PAGES:
        current_page = "Dashboard"
        st.session_state["current_page"] = "Dashboard"

    current_idx = PAGES.index(current_page)

    with st.sidebar:
        st.markdown("""
        <div class="brand-header">
            <div style="background: linear-gradient(135deg, #1677FF 0%, #FF244F 100%); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; box-shadow: 0 0 15px rgba(22, 119, 255, 0.5);">
                ⚡
            </div>
            <div>
                <div class="brand-title">NeuralWave AI</div>
                <div class="brand-sub">AI Music Studio</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.75rem; color: #9AA6BA; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;'>Studio Navigation</div>", unsafe_allow_html=True)

        # Radio button dynamically synced to current_page
        selected_page = st.radio(
            "Studio Navigation",
            PAGES,
            index=current_idx,
            key=f"sidebar_radio_{current_page}",
            label_visibility="collapsed",
        )

        if selected_page != current_page:
            navigate_to(selected_page)

        # Dedicated Sidebar Backward Action
        history = st.session_state.get("page_history", [])
        if history:
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button(f"⬅ Back to {history[-1]}", key="sidebar_back_action_btn", use_container_width=True):
                navigate_back()

        st.markdown("<hr style='border-color: #263247; margin: 20px 0;'>", unsafe_allow_html=True)

        # Quick System Monitor
        st.markdown("<div style='font-size: 0.72rem; color: #9AA6BA; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;'>Engine Monitor</div>", unsafe_allow_html=True)
        env = get_system_environment_info()
        device_lbl = "CUDA GPU" if env["cuda_available"] else "CPU Mode"
        badge_style = "badge-blue" if env["cuda_available"] else "badge-red"

        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #263247; border-radius: 8px; padding: 10px; font-size: 0.78rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #9AA6BA;">Hardware:</span>
                <span class="{badge_style}">{device_lbl}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #9AA6BA;">PyTorch:</span>
                <span style="color: #F4F7FF; font-family: monospace;">v{env['torch_version'].split('+')[0]}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #9AA6BA;">Audio Engine:</span>
                <span style="color: #31D6A0;">WebAudio + MIDI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------
def page_dashboard(cfg):
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <span class="badge-blue" style="margin-bottom: 8px;">Studio Workspace</span>
        <h1 style="font-size: 2.6rem; font-weight: 800; margin: 6px 0; letter-spacing: -0.03em;">
            Turn Patterns Into Music.
        </h1>
        <p style="color: #9AA6BA; font-size: 1.05rem; max-width: 720px; line-height: 1.5;">
            NeuralWave AI uses deep recurrent neural networks (LSTM) to learn polyphonic musical grammar from MIDI datasets and compose expressive, downloadable compositions in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Check status of components
    data_dir = Path(cfg["dataset"]["data_dir"])
    prep_dir = Path(cfg["dataset"]["preprocessed_dir"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    best_ckpt = ckpt_dir / "best_model.pt"

    midi_files = discover_midi_files(data_dir)
    prep_exists = (prep_dir / "dataset.pt").exists()
    model_exists = best_ckpt.exists()
    compositions = list_compositions()

    # Determine Studio State
    if model_exists:
        state_badge = '<span class="badge-success">● Model Ready for Generation</span>'
    elif prep_exists:
        state_badge = '<span class="badge-blue">● Dataset Preprocessed & Ready</span>'
    elif len(midi_files) > 0:
        state_badge = '<span class="badge-blue">● MIDI Files Discovered</span>'
    else:
        state_badge = '<span class="badge-red">● Workspace Unconfigured (Get Started)</span>'

    st.markdown(f"<div style='margin-bottom: 16px;'>{state_badge}</div>", unsafe_allow_html=True)

    # Top Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="daw-card">
            <div class="daw-stat-val">{len(midi_files)}</div>
            <div class="daw-stat-label">MIDI Files in Dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        vocab_str = "—"
        if prep_exists:
            try:
                p_data = load_preprocessed_dataset(prep_dir)
                vocab_str = str(len(p_data["vocab_to_idx"]))
            except Exception:
                vocab_str = "Ready"
        st.markdown(f"""
        <div class="daw-card">
            <div class="daw-stat-val">{vocab_str}</div>
            <div class="daw-stat-label">Learned Note Vocabulary</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        loss_str = "—"
        if model_exists:
            val_info = validate_model_checkpoint(best_ckpt)
            if val_info.get("recorded_loss") is not None:
                loss_str = f"{val_info['recorded_loss']:.3f}"
            else:
                loss_str = "Trained"
        st.markdown(f"""
        <div class="daw-card">
            <div class="daw-stat-val" style="color: #36A3FF;">{loss_str}</div>
            <div class="daw-stat-label">Best Training Loss</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="daw-card">
            <div class="daw-stat-val" style="color: #FF244F;">{len(compositions)}</div>
            <div class="daw-stat-label">Compositions Created</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Action Buttons
    col_act1, col_act2, col_act3 = st.columns([1, 1, 2])
    with col_act1:
        if st.button("⚡ Generate New Music", use_container_width=True):
            navigate_to("Music Generator")
    with col_act2:
        if st.button("🎛 Train Neural Model", use_container_width=True):
            navigate_to("Model Training")
    with col_act3:
        if len(midi_files) == 0:
            if st.button("✨ Load Demo Synthetic Dataset (1-Click)", use_container_width=True):
                data_dir.mkdir(parents=True, exist_ok=True)
                create_synthetic_midi(data_dir / "synthetic_demo.mid", bpm=120)
                st.success("Loaded synthetic demo MIDI fixture! You can now preprocess and train.")
                st.rerun()

    st.markdown("<hr style='border-color: #263247; margin: 24px 0;'>", unsafe_allow_html=True)

    # Training Loss Graph or Onboarding
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("Training Loss Curve")
        meta_path = ckpt_dir / "training_meta.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    t_meta = json.load(f)
                history = t_meta.get("history", [])
                if history:
                    epochs = [h["epoch"] for h in history]
                    train_losses = [h["train_loss"] for h in history]
                    val_losses = [h["val_loss"] for h in history if h.get("val_loss") is not None]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=epochs, y=train_losses,
                        mode="lines+markers",
                        name="Train Loss",
                        line=dict(color="#1677FF", width=3),
                        marker=dict(size=6, color="#36A3FF"),
                    ))
                    if len(val_losses) == len(epochs):
                        fig.add_trace(go.Scatter(
                            x=epochs, y=val_losses,
                            mode="lines+markers",
                            name="Validation Loss",
                            line=dict(color="#FF244F", width=2, dash="dash"),
                            marker=dict(size=6, color="#FF244F"),
                        ))
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="#0D111B",
                        paper_bgcolor="#111827",
                        margin=dict(l=40, r=20, t=30, b=30),
                        height=260,
                        xaxis=dict(title=dict(text="Epoch", font=dict(color="#9AA6BA", size=11)), gridcolor="#263247", tickfont=dict(color="#9AA6BA")),
                        yaxis=dict(title=dict(text="Cross-Entropy Loss", font=dict(color="#9AA6BA", size=11)), gridcolor="#263247", tickfont=dict(color="#9AA6BA")),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No training history recorded yet.")
            except Exception as e:
                st.info(f"Could not read training history: {e}")
        else:
            st.markdown("""
            <div class="daw-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">🎹</div>
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 6px;">No Model Trained Yet</div>
                <p style="color: #9AA6BA; font-size: 0.88rem; max-width: 400px; margin: 0 auto 16px auto;">
                    Prepare a MIDI collection in Dataset Lab and run a training session to see real-time cross-entropy loss tracking.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        st.subheader("Recent Compositions")
        if compositions:
            latest = compositions[0]
            st.markdown(f"""
            <div class="daw-card-accent">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span class="badge-red">Latest Output</span>
                    <span style="font-size: 0.75rem; color: #9AA6BA;">{latest.get('created_at', '')}</span>
                </div>
                <div style="font-weight: 700; font-size: 1.15rem; color: #F4F7FF; margin-bottom: 4px;">
                    {latest.get('title', 'NeuralWave Composition')}
                </div>
                <div style="font-size: 0.85rem; color: #9AA6BA; margin-bottom: 12px;">
                    {latest.get('num_notes', 0)} notes • {latest.get('bpm', 120)} BPM • {latest.get('duration_seconds', 0.0)}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            if latest.get("sample_tokens"):
                st.caption("Live WebAudio Playback Preview:")
                render_web_audio_player(latest["sample_tokens"], bpm=latest.get("bpm", 120), key="dash_preview")
        else:
            st.markdown("""
            <div class="daw-card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 2.5rem; margin-bottom: 10px;">🎧</div>
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 6px;">No Compositions Yet</div>
                <p style="color: #9AA6BA; font-size: 0.88rem; max-width: 380px; margin: 0 auto;">
                    Once you train a model, visit the Music Generator to synthesize original tracks.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Page 2: Dataset Lab
# ---------------------------------------------------------
def page_dataset_lab(cfg):
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <span class="badge-blue">Data Engineering</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 4px 0;">Dataset Lab</h1>
        <p style="color: #9AA6BA;">Upload MIDI files, inspect musical pitch ranges, and tokenize note sequences for PyTorch training.</p>
    </div>
    """, unsafe_allow_html=True)

    data_dir = Path(cfg["dataset"]["data_dir"])
    prep_dir = Path(cfg["dataset"]["preprocessed_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)

    # File Upload Section
    c_up1, c_up2 = st.columns([2, 1])

    with c_up1:
        st.subheader("Upload MIDI Music")
        uploaded_files = st.file_uploader(
            "Upload .mid, .midi, or a .zip archive containing MIDI files",
            type=["mid", "midi", "zip"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            saved_count = 0
            for uf in uploaded_files:
                if uf.name.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(uf.read())) as zf:
                            for member in zf.namelist():
                                if member.lower().endswith((".mid", ".midi")) and not member.startswith("__MACOSX"):
                                    extracted_data = zf.read(member)
                                    target_p = data_dir / Path(member).name
                                    with open(target_p, "wb") as out_f:
                                        out_f.write(extracted_data)
                                    saved_count += 1
                    except Exception as e:
                        st.error(f"Error extracting ZIP: {e}")
                else:
                    target_p = data_dir / uf.name
                    with open(target_p, "wb") as out_f:
                        out_f.write(uf.read())
                    saved_count += 1

            if saved_count > 0:
                st.success(f"Successfully imported {saved_count} MIDI files into {data_dir.name}/")
                st.rerun()

    with c_up2:
        st.subheader("Synthetic Test Data")
        st.markdown("<p style='font-size: 0.85rem; color: #9AA6BA;'>For quick offline demonstration or unit testing without external files:</p>", unsafe_allow_html=True)
        if st.button("Generate Synthetic Sample MIDI"):
            synth_p = data_dir / "synthetic_sample.mid"
            create_synthetic_midi(synth_p, title="Synthetic Test Fixture", bpm=120)
            st.success("Created synthetic test MIDI file!")
            st.rerun()

    # Discover and inspect current files
    current_files = discover_midi_files(data_dir)
    st.markdown("<hr style='border-color: #263247; margin: 20px 0;'>", unsafe_allow_html=True)

    st.subheader(f"Discovered MIDI Files ({len(current_files)})")

    if current_files:
        # Table of files
        files_data = []
        for f in current_files[:20]:
            sz_kb = round(f.stat().st_size / 1024, 2)
            files_data.append({"Filename": f.name, "Size (KB)": sz_kb, "Path": str(f.relative_to(ROOT_DIR))})
        st.dataframe(files_data, use_container_width=True)
        if len(current_files) > 20:
            st.caption(f"Showing first 20 of {len(current_files)} discovered files.")

        # Preprocessing Trigger
        st.markdown("<hr style='border-color: #263247; margin: 20px 0;'>", unsafe_allow_html=True)
        st.subheader("Preprocess & Tokenize")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            seq_len = st.slider("Sequence Window Length (Notes)", min_value=4, max_value=64, value=int(cfg["dataset"]["sequence_length"]), step=2)
        with p_col2:
            min_notes = st.number_input("Minimum Notes per Song", min_value=4, max_value=50, value=int(cfg["dataset"]["min_notes_per_file"]))

        if st.button("🚀 Run Preprocessing Pipeline", use_container_width=True):
            with st.spinner("Parsing MIDI files, constructing vocabulary, and generating sequence tensors..."):
                try:
                    summary = preprocess_midi_dataset(
                        data_dir=data_dir,
                        output_dir=prep_dir,
                        sequence_length=seq_len,
                        min_notes_per_file=min_notes,
                    )
                    st.success(f"Preprocessing Complete! Extracted {summary['total_notes_extracted']} notes across {summary['valid_files']} valid files.")
                    st.info(f"Generated {summary['sequence_count']} training sequences. Vocabulary size: {summary['vocab_size']} unique tokens.")
                except Exception as e:
                    st.error(f"Preprocessing failed: {str(e)}")

        # If already preprocessed, display metadata
        if (prep_dir / "dataset.pt").exists():
            st.markdown("<hr style='border-color: #263247; margin: 16px 0;'>", unsafe_allow_html=True)
            st.markdown("<span class='badge-success'>Preprocessed Dataset Ready</span>", unsafe_allow_html=True)
            try:
                with open(prep_dir / "preprocessing_meta.json", "r", encoding="utf-8") as pf:
                    p_info = json.load(pf)["summary"]
                st.json(p_info)
            except Exception:
                pass
    else:
        st.info("No MIDI files found in `data/midi/`. Upload files or click 'Generate Synthetic Sample MIDI' above.")


# ---------------------------------------------------------
# Page 3: Model Training
# ---------------------------------------------------------
def page_model_training(cfg):
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <span class="badge-blue">Deep Learning Engine</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 4px 0;">Model Training</h1>
        <p style="color: #9AA6BA;">Configure PyTorch LSTM hyperparameters, monitor live cross-entropy loss, and save model checkpoints.</p>
    </div>
    """, unsafe_allow_html=True)

    prep_dir = Path(cfg["dataset"]["preprocessed_dir"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])

    if not (prep_dir / "dataset.pt").exists():
        st.warning("⚠️ No preprocessed dataset found. Please visit Dataset Lab first to preprocess your MIDI files.")
        if st.button("Go to Dataset Lab"):
            navigate_to("Dataset Lab")
        return

    # Check dataset specs
    p_data = load_preprocessed_dataset(prep_dir)
    vocab_size = len(p_data["vocab_to_idx"])
    sample_count = len(p_data["inputs"])

    st.markdown(f"""
    <div class="daw-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <div style="font-weight: 700; color: #F4F7FF;">Dataset Snapshot</div>
            <div style="font-size: 0.85rem; color: #9AA6BA;">Vocabulary: {vocab_size} unique pitch/chord tokens • Sequences: {sample_count} samples</div>
        </div>
        <span class="badge-success">Ready for Training</span>
    </div>
    """, unsafe_allow_html=True)

    # Hyperparameter Inputs
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=int(cfg["training"]["epochs"]), step=1)
        batch_size = st.select_slider("Batch Size", options=[4, 8, 16, 32, 64], value=int(cfg["training"]["batch_size"]))
        lr = st.select_slider("Learning Rate", options=[0.0005, 0.001, 0.002, 0.005, 0.01], value=float(cfg["training"]["learning_rate"]))

    with col_p2:
        embedding_dim = st.select_slider("Embedding Dimension", options=[32, 64, 128, 256], value=int(cfg["model"]["embedding_dim"]))
        hidden_dim = st.select_slider("LSTM Hidden Units", options=[64, 128, 256, 512], value=int(cfg["model"]["hidden_dim"]))
        num_layers = st.slider("LSTM Layers", min_value=1, max_value=4, value=int(cfg["model"]["num_layers"]))

    with col_p3:
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=float(cfg["model"]["dropout"]), step=0.05)
        device_choice = st.selectbox("Execution Device", options=["auto", "cpu", "cuda"], index=0)
        seed = st.number_input("Random Seed", value=int(cfg["training"]["seed"]))

    st.markdown("<hr style='border-color: #263247; margin: 16px 0;'>", unsafe_allow_html=True)

    # Memory / CPU Notice
    if device_choice == "cpu" or not torch.cuda.is_available():
        st.caption("ℹ️ Running on CPU. Training 10–20 epochs on a small dataset typically takes under 15 seconds.")

    # Live Training Session Container
    start_train = st.button("🔥 Start Neural Network Training", use_container_width=True)

    if start_train:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        chart_slot = st.empty()

        train_losses = []
        val_losses = []
        epoch_nums = []

        def training_callback(epoch, total_epochs, t_loss, v_loss, elapsed):
            progress_bar.progress(epoch / total_epochs)
            v_str = f" | Val Loss: {v_loss:.4f}" if v_loss is not None else ""
            status_text.markdown(f"**Epoch {epoch}/{total_epochs}** — Train Loss: `{t_loss:.4f}`{v_str} — Elapsed: `{elapsed:.1f}s`")

            epoch_nums.append(epoch)
            train_losses.append(t_loss)
            if v_loss is not None:
                val_losses.append(v_loss)

            # Live loss graph update
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epoch_nums, y=train_losses, mode="lines+markers", name="Train Loss", line=dict(color="#1677FF", width=3)))
            if val_losses:
                fig.add_trace(go.Scatter(x=epoch_nums, y=val_losses, mode="lines+markers", name="Val Loss", line=dict(color="#FF244F", width=2, dash="dash")))
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0D111B",
                paper_bgcolor="#111827",
                height=260,
                margin=dict(l=30, r=20, t=20, b=20),
                xaxis=dict(title=dict(text="Epoch", font=dict(color="#9AA6BA", size=11)), gridcolor="#263247", tickfont=dict(color="#9AA6BA")),
                yaxis=dict(title=dict(text="Loss", font=dict(color="#9AA6BA", size=11)), gridcolor="#263247", tickfont=dict(color="#9AA6BA")),
            )
            chart_slot.plotly_chart(fig, use_container_width=True)

        try:
            results = train_model(
                dataset_dir=prep_dir,
                checkpoint_dir=ckpt_dir,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device_name=device_choice,
                seed=seed,
                progress_callback=training_callback,
            )
            st.success(f"🎉 Training {results['status'].capitalize()}! Best Loss: {results['best_loss']} in {results['training_time_seconds']}s.")
            st.info(f"Model checkpoint saved to: `{ckpt_dir / 'best_model.pt'}`")
        except Exception as e:
            st.error(f"Training interrupted with error: {str(e)}")


# ---------------------------------------------------------
# Page 4: Music Generator (Main Showcase)
# ---------------------------------------------------------
def page_music_generator(cfg):
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <span class="badge-red">Creative Synthesis</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 4px 0;">Music Generator</h1>
        <p style="color: #9AA6BA;">Generate original polyphonic compositions autoregressively from trained neural checkpoints.</p>
    </div>
    """, unsafe_allow_html=True)

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    available_ckpts = list(ckpt_dir.glob("*.pt"))

    if not available_ckpts:
        st.markdown("""
        <div class="daw-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 2.5rem; margin-bottom: 12px;">🎹</div>
            <div style="font-weight: 700; font-size: 1.2rem; margin-bottom: 8px;">No Model Checkpoints Found</div>
            <p style="color: #9AA6BA; font-size: 0.9rem; max-width: 480px; margin: 0 auto 16px auto;">
                Please train a model in Model Training or run the pipeline verification script to generate a valid checkpoint.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Model Training"):
            navigate_to("Model Training")
        return

    # Checkpoint Selection
    ckpt_names = [c.name for c in available_ckpts]
    selected_ckpt_name = st.selectbox("Select Model Checkpoint", options=ckpt_names, index=0)
    selected_ckpt_path = ckpt_dir / selected_ckpt_name

    # Checkpoint Metadata Inspector
    ckpt_val = validate_model_checkpoint(selected_ckpt_path)
    if not ckpt_val["valid"]:
        st.error(f"Selected checkpoint is invalid: {ckpt_val.get('error')}")
        return

    st.markdown(f"""
    <div style="background: #0D111B; border: 1px solid #263247; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; font-size: 0.82rem; display: flex; gap: 20px; flex-wrap: wrap;">
        <span>Vocab Size: <b style="color: #F4F7FF;">{ckpt_val['vocab_size']}</b></span>
        <span>Hidden Dim: <b style="color: #36A3FF;">{ckpt_val['hidden_dim']}</b></span>
        <span>Recorded Loss: <b style="color: #31D6A0;">{ckpt_val.get('recorded_loss', 'N/A')}</b></span>
        <span>Epochs: <b style="color: #FF244F;">{ckpt_val.get('epochs_recorded', 'N/A')}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Quick Style Presets for User Experience
    st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #9AA6BA; text-transform: uppercase; margin-bottom: 8px;'>✨ Quick Style & Mood Presets</div>", unsafe_allow_html=True)
    ps1, ps2, ps3, ps4 = st.columns(4)
    with ps1:
        if st.button("🎹 Classical Nocturne", use_container_width=True):
            st.session_state["p_notes"] = 48
            st.session_state["p_temp"] = 0.65
            st.session_state["p_bpm"] = 96
            st.session_state["p_inst"] = "Acoustic Grand Piano"
            st.rerun()
    with ps2:
        if st.button("⚡ Cyberpunk Synth", use_container_width=True):
            st.session_state["p_notes"] = 64
            st.session_state["p_temp"] = 0.85
            st.session_state["p_bpm"] = 128
            st.session_state["p_inst"] = "Synthesizer"
            st.rerun()
    with ps3:
        if st.button("🎻 Cinematic Strings", use_container_width=True):
            st.session_state["p_notes"] = 36
            st.session_state["p_temp"] = 0.60
            st.session_state["p_bpm"] = 80
            st.session_state["p_inst"] = "Violin"
            st.rerun()
    with ps4:
        if st.button("🎷 Midnight Jazz", use_container_width=True):
            st.session_state["p_notes"] = 56
            st.session_state["p_temp"] = 1.05
            st.session_state["p_bpm"] = 120
            st.session_state["p_inst"] = "Saxophone"
            st.rerun()

    init_notes = st.session_state.get("p_notes", int(cfg["generation"]["num_notes"]))
    init_temp = st.session_state.get("p_temp", float(cfg["generation"]["temperature"]))
    init_bpm = st.session_state.get("p_bpm", int(cfg["generation"]["bpm"]))
    init_inst = st.session_state.get("p_inst", cfg["generation"]["instrument"])
    inst_keys = list(INSTRUMENT_MAP.keys())
    inst_idx = inst_keys.index(init_inst) if init_inst in inst_keys else 0

    # Generation Parameters
    g_col1, g_col2, g_col3 = st.columns(3)

    with g_col1:
        num_notes = st.slider("Note Sequence Length", min_value=16, max_value=128, value=init_notes, step=8,
                              help="Total number of consecutive musical tokens to synthesize.")
        temperature = st.slider("Temperature (Randomness)", min_value=0.2, max_value=1.5, value=init_temp, step=0.05,
                                help="Lower values produce structured, repetitive melodies; higher values produce creative, experimental variations.")

    with g_col2:
        bpm = st.slider("Tempo (BPM)", min_value=60, max_value=200, value=init_bpm, step=5,
                        help="Beats per minute for playback speed.")
        instrument = st.selectbox("Instrument Program", options=inst_keys, index=inst_idx,
                                  help="General MIDI instrument timbre.")

    with g_col3:
        comp_title = st.text_input("Composition Title", value=f"NeuralWave Track #{int(time.time()) % 1000}")
        use_random_seed = st.checkbox("Custom Random Seed", value=False)
        seed_val = st.number_input("Seed Value", value=42) if use_random_seed else None

    st.markdown("<hr style='border-color: #263247; margin: 16px 0;'>", unsafe_allow_html=True)

    # Big Neon Action Button
    st.markdown('<div class="neon-action-btn">', unsafe_allow_html=True)
    generate_btn = st.button("⚡ Synthesize New Composition", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Storage for active generation session
    if generate_btn or st.session_state.get("trigger_regenerate", False):
        st.session_state["trigger_regenerate"] = False
        out_dir = Path(cfg["generation"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        file_base = f"comp_{int(time.time())}"
        mid_target = out_dir / f"{file_base}.mid"

        with st.spinner("Executing PyTorch LSTM autoregressive inference & synthesizing MIDI stream..."):
            try:
                gen_res = generate_composition(
                    checkpoint_path=selected_ckpt_path,
                    output_path=mid_target,
                    num_notes=num_notes,
                    temperature=temperature,
                    bpm=bpm,
                    instrument=instrument,
                    random_seed=seed_val,
                    title=comp_title,
                )

                # Persist composition record
                meta_record = gen_res["metadata"]
                save_composition_record(file_base, meta_record)
                st.session_state["latest_composition"] = gen_res
                st.success(f"✨ Composition successfully generated! ({len(gen_res['tokens'])} notes)")
            except Exception as e:
                st.error(f"Generation error: {str(e)}")

    # Display Active Composition
    if "latest_composition" in st.session_state:
        comp = st.session_state["latest_composition"]
        meta = comp["metadata"]

        st.markdown("<hr style='border-color: #263247; margin: 20px 0;'>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="daw-card-accent">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="badge-red">Active Composition</span>
                <span style="font-size: 0.8rem; color: #9AA6BA;">{meta.get('created_at')}</span>
            </div>
            <h2 style="font-size: 1.5rem; font-weight: 700; margin: 0 0 6px 0; color: #F4F7FF;">{meta.get('title')}</h2>
            <div style="display: flex; gap: 20px; font-size: 0.85rem; color: #9AA6BA; flex-wrap: wrap;">
                <span>Instrument: <b style="color: #F4F7FF;">{meta.get('instrument')}</b></span>
                <span>Tempo: <b style="color: #F4F7FF;">{meta.get('bpm')} BPM</b></span>
                <span>Events: <b style="color: #36A3FF;">{meta.get('num_notes')} notes</b></span>
                <span>Duration: <b style="color: #31D6A0;">{meta.get('duration_seconds')}s</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Audio Player Section
        st.subheader("Auditory Playback")
        render_web_audio_player(comp["tokens"], bpm=meta.get("bpm", 120), key="gen_play")

        # Download & Variation Buttons
        midi_path = Path(comp["midi_path"])
        if midi_path.exists():
            with open(midi_path, "rb") as mf:
                midi_bytes = mf.read()
            d_c1, d_c2, d_c3 = st.columns([1, 1, 2])
            with d_c1:
                st.download_button(
                    label="📥 Download Standard MIDI (.mid)",
                    data=midi_bytes,
                    file_name=midi_path.name,
                    mime="audio/midi",
                    use_container_width=True,
                )
            with d_c2:
                if st.button("🔄 Generate Another Variation", use_container_width=True):
                    st.session_state["trigger_regenerate"] = True
                    st.rerun()
            with d_c3:
                st.caption(f"File size: {len(midi_bytes)} bytes. Compatible with MuseScore, GarageBand, FL Studio, Ableton, and Windows Media Player.")

        # Piano Roll Visualization
        st.subheader("Piano-Roll Visualization")
        render_piano_roll_plot(comp["tokens"])


# ---------------------------------------------------------
# Page 5: My Compositions (Library)
# ---------------------------------------------------------
def page_my_compositions(cfg):
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <span class="badge-blue">Studio Library</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 4px 0;">My Compositions</h1>
        <p style="color: #9AA6BA;">Manage, listen to, and download all generated musical pieces in your studio catalog.</p>
    </div>
    """, unsafe_allow_html=True)

    compositions = list_compositions()

    if not compositions:
        st.info("No compositions found in `outputs/`. Generate your first track in the Music Generator!")
        return

    st.markdown(f"<div style='margin-bottom: 16px; color: #9AA6BA; font-size: 0.88rem;'>Stored Tracks: <b style='color: #F4F7FF;'>{len(compositions)}</b></div>", unsafe_allow_html=True)

    for comp in compositions:
        c_id = comp.get("id", "track")
        with st.expander(f"🎵 {comp.get('title', 'Untitled')} — {comp.get('created_at', '')}", expanded=False):
            col_info, col_actions = st.columns([2, 1])

            with col_info:
                st.markdown(f"""
                <div style="font-size: 0.85rem; color: #9AA6BA; line-height: 1.6;">
                    • <b>Notes:</b> {comp.get('num_notes', 'N/A')}<br>
                    • <b>Tempo:</b> {comp.get('bpm', 120)} BPM<br>
                    • <b>Duration:</b> {comp.get('duration_seconds', 0.0)} seconds<br>
                    • <b>Instrument:</b> {comp.get('instrument', 'Standard')}
                </div>
                """, unsafe_allow_html=True)

                # Web Audio Player if tokens exist in metadata
                if comp.get("sample_tokens"):
                    render_web_audio_player(comp["sample_tokens"], bpm=comp.get("bpm", 120), key=f"lib_{c_id}")

            with col_actions:
                # MIDI Download
                midi_p = Path(comp.get("midi_path", ""))
                if midi_p.exists():
                    with open(midi_p, "rb") as f:
                        st.download_button(
                            label="📥 Download .mid",
                            data=f.read(),
                            file_name=midi_p.name,
                            mime="audio/midi",
                            key=f"dl_{c_id}",
                        )

                # Rename
                new_name = st.text_input("Rename Track", value=comp.get("title", ""), key=f"ren_{c_id}")
                if st.button("Update Title", key=f"btn_ren_{c_id}"):
                    if rename_composition(c_id, new_name):
                        st.success("Renamed!")
                        st.rerun()

                # Delete
                if st.button("🗑 Delete Track", key=f"del_{c_id}"):
                    delete_composition(c_id)
                    st.success("Deleted composition.")
                    st.rerun()


# ---------------------------------------------------------
# Page 6: Model & Settings
# ---------------------------------------------------------
def page_settings(cfg):
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <span class="badge-blue">System Diagnostics</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 4px 0;">Model & Settings</h1>
        <p style="color: #9AA6BA;">Inspect system runtime, verify dependencies, and configure application paths.</p>
    </div>
    """, unsafe_allow_html=True)

    env = get_system_environment_info()
    dirs = validate_directories(ROOT_DIR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hardware & Software Runtime")
        st.markdown(f"""
        <div class="daw-card">
            <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #263247;"><td style="padding: 6px 0; color: #9AA6BA;">Python Runtime</td><td style="text-align: right; color: #F4F7FF; font-family: monospace;">{env['python_version']}</td></tr>
                <tr style="border-bottom: 1px solid #263247;"><td style="padding: 6px 0; color: #9AA6BA;">PyTorch Version</td><td style="text-align: right; color: #F4F7FF; font-family: monospace;">{env['torch_version']}</td></tr>
                <tr style="border-bottom: 1px solid #263247;"><td style="padding: 6px 0; color: #9AA6BA;">music21 Version</td><td style="text-align: right; color: #F4F7FF; font-family: monospace;">{env['music21_version']}</td></tr>
                <tr style="border-bottom: 1px solid #263247;"><td style="padding: 6px 0; color: #9AA6BA;">Execution Device</td><td style="text-align: right; color: #36A3FF; font-weight: 600;">{env['device_name']}</td></tr>
                <tr><td style="padding: 6px 0; color: #9AA6BA;">CUDA Available</td><td style="text-align: right; color: {'#31D6A0' if env['cuda_available'] else '#FFBF5E'};">{env['cuda_available']}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Workspace Directory Health")
        dir_rows = ""
        for name, d_info in dirs.items():
            status_dot = "🟢" if d_info["exists"] and d_info["writable"] else "🔴"
            dir_rows += f"""
            <tr style="border-bottom: 1px solid #263247;">
                <td style="padding: 6px 0; color: #9AA6BA;">{name}</td>
                <td style="text-align: right; font-family: monospace; font-size: 0.8rem; color: #F4F7FF;">{d_info['file_count']} files {status_dot}</td>
            </tr>
            """
        st.markdown(f"""
        <div class="daw-card">
            <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                {dir_rows}
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #263247; margin: 16px 0;'>", unsafe_allow_html=True)

    st.subheader("Audio Synthesis Engine (FluidSynth)")
    renderer = env["audio_renderer"]
    if renderer["available"]:
        st.success("FluidSynth and SoundFont are detected! Direct WAV audio rendering is supported.")
    else:
        st.markdown("""
        <div class="daw-card">
            <div style="font-weight: 700; color: #FFBF5E; margin-bottom: 6px;">FluidSynth Not Detected (WebAudio Active)</div>
            <p style="font-size: 0.85rem; color: #9AA6BA; line-height: 1.5; margin: 0;">
                The studio is using the built-in <b>HTML5 WebAudio PolySynth</b> for zero-dependency browser playback.
                To enable optional offline rendering from MIDI to WAV audio files, install FluidSynth and a General MIDI SoundFont (.sf2).
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #263247; margin: 16px 0;'>", unsafe_allow_html=True)

    # Danger Zone / Reset
    st.subheader("Workspace Maintenance")
    st.caption("Safely clears generated compositions from `outputs/` without modifying raw datasets or model weights.")
    confirm_reset = st.checkbox("I confirm I want to clear all generated compositions.")
    if st.button("Clear Generated Compositions", disabled=not confirm_reset):
        count = reset_generated_artifacts(confirm=True)
        st.success(f"Cleared {count} generated files from outputs.")
        st.rerun()


# ---------------------------------------------------------
# Page 7: About Project
# ---------------------------------------------------------
def page_about():
    st.markdown("""
    <div style="padding: 10px 0 16px 0;">
        <span class="badge-blue">Internship Portfolio</span>
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 4px 0;">About NeuralWave AI</h1>
        <p style="color: #9AA6BA;">Deep learning architecture, musical representations, and CodeAlpha AI internship documentation.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="daw-card">
            <h3 style="font-size: 1.15rem; color: #36A3FF; margin-top: 0;">How the LSTM Generates Music</h3>
            <p style="font-size: 0.88rem; color: #9AA6BA; line-height: 1.6;">
                Music is fundamentally a sequence of symbolic events over time. Just as Large Language Models predict the next word token in a sentence, NeuralWave AI predicts the next musical token (pitch, chord, or rest).
            </p>
            <ol style="font-size: 0.85rem; color: #9AA6BA; line-height: 1.7; padding-left: 20px;">
                <li><b>Tokenization:</b> MIDI files are parsed with <code>music21</code> into pitch and chord symbols.</li>
                <li><b>Sequence Windows:</b> Overlapping sliding windows of length $L$ predict target note $y_{t+1}$.</li>
                <li><b>Song Boundaries:</b> Sequences never cross song boundaries to preserve key relationships.</li>
                <li><b>Sampling:</b> Temperature-scaled softmax sampling controls melodic creativity vs. repetition.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="daw-card">
            <h3 style="font-size: 1.15rem; color: #FF244F; margin-top: 0;">CodeAlpha Internship Specs</h3>
            <p style="font-size: 0.88rem; color: #9AA6BA; line-height: 1.6;">
                <b>Role:</b> Senior AI Engineer & Full-Stack Developer<br>
                <b>Primary Output:</b> Valid Standard MIDI (.mid) files + WebAudio playback<br>
                <b>Frameworks:</b> PyTorch 2.x, music21, Streamlit, Plotly, NumPy<br>
                <b>Execution:</b> Verified CPU and CUDA compatibility
            </p>
            <p style="font-size: 0.85rem; color: #9AA6BA; line-height: 1.5;">
                <b>Honest Limitations:</b> Next-token LSTMs learn local harmonic and melodic transitions effectively, but struggle with global multi-minute thematic cohesion without attention mechanisms or massive datasets.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="daw-card">
        <h3 style="font-size: 1.15rem; color: #31D6A0; margin-top: 0;">Licensing & Public Datasets</h3>
        <p style="font-size: 0.88rem; color: #9AA6BA; line-height: 1.6;">
            Code is provided under the open-source <b>MIT License</b>. Datasets like the <b>Lakh MIDI Dataset (LMD)</b> are distributed under their respective creator licenses (Colin Raffel, CC-BY 4.0). Users should ensure commercial rights before training on proprietary copyrighted scores.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Router
# ---------------------------------------------------------
def main():
    cfg = load_config()
    init_navigation()

    # Render sidebar with fully interactive and synchronized selection
    render_sidebar()

    # Render top global navigation bar with backward button & breadcrumbs
    render_top_navigation()

    active_page = st.session_state.get("current_page", "Dashboard")

    if active_page == "Dashboard":
        page_dashboard(cfg)
    elif active_page == "Dataset Lab":
        page_dataset_lab(cfg)
    elif active_page == "Model Training":
        page_model_training(cfg)
    elif active_page == "Music Generator":
        page_music_generator(cfg)
    elif active_page == "My Compositions":
        page_my_compositions(cfg)
    elif active_page == "Model & Settings":
        page_settings(cfg)
    elif active_page == "About Project":
        page_about()


if __name__ == "__main__":
    main()
