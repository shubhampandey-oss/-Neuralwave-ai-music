# NeuralWave AI — Troubleshooting & FAQ

This guide addresses common setup, training, audio playback, and runtime issues.

---

## 1. MIDI Parsing & Dataset Errors

### Issue: `ValueError: No MIDI files found in directory`
- **Cause**: The `data/midi/` directory is empty or contains non-MIDI files.
- **Fix**: Upload `.mid` or `.midi` files in the **Dataset Lab** page, or click **"Generate Synthetic Sample MIDI"** for instant offline testing.

### Issue: `File contains too few musical elements`
- **Cause**: The MIDI file contains fewer notes than the `min_notes_per_file` threshold (default: 8 notes), or is an empty track.
- **Fix**: The preprocessor automatically skips empty tracks and logs a warning. If working with short musical phrases, lower `min_notes_per_file` in `config.yaml` or on the Dataset Lab page.

### Issue: Tokenizer Memory Error on Massive Datasets
- **Cause**: Attempting to load tens of thousands of MIDI files into RAM at once.
- **Fix**: Set `max_files: 100` in `config.yaml` to parse a manageable subset of the dataset.

---

## 2. Model Training & PyTorch

### Issue: `RuntimeError: CUDA out of memory`
- **Cause**: GPU VRAM exhausted due to large batch size or high hidden dimension.
- **Fix**: Reduce `batch_size` to 16 or 8, or select `device: "cpu"` in the training panel.

### Issue: Training is slow on CPU
- **Cause**: High number of LSTM layers (e.g., 4 layers) or huge vocabulary.
- **Fix**: Use 1 or 2 LSTM layers and a sequence length of 16–32 for snappy CPU training (usually takes 5–15 seconds for 10 epochs on demo datasets).

### Issue: Loss does not decrease
- **Cause**: Learning rate too high or dataset too small.
- **Fix**: Use a learning rate between `0.001` and `0.003` with the default AdamW optimizer.

---

## 3. Audio Playback & Synthesis

### Issue: Browser does not play audio
- **Cause**: Browser blocked autoplay, or WebAudio context is suspended.
- **Fix**: Click the **▶ Play Composition** button in the WebAudio PolySynth player. Modern browsers require a user click before playing WebAudio streams.

### Issue: `FluidSynth not detected`
- **Cause**: FluidSynth command-line synthesizer is not installed on system PATH.
- **Fix**:
  - The studio automatically falls back to the zero-dependency **WebAudio PolySynth** inside the browser for real-time auditory playback!
  - If you need offline rendered `.wav` files:
    - **Windows**: Run `winget install FluidSynth` or `choco install fluidsynth`. Download a General MIDI SoundFont (`.sf2`) and set its path in `config.yaml`.
    - **Ubuntu/Debian**: `sudo apt install fluidsynth fluid-soundfont-gm`
    - **macOS**: `brew install fluidsynth`

---

## 4. Environment & Dependency Conflicts

### Issue: `Command 'streamlit' is not recognized`
- **Cause**: Python `Scripts/` directory is not in your system environment PATH.
- **Fix**: Run using the Python module launcher:
  ```bash
  py -3.11 -m streamlit run app.py
  ```
