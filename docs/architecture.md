# NeuralWave AI — System Architecture

## Overview
NeuralWave AI is a symbolic music composition studio powered by deep recurrent neural networks (LSTM) built with PyTorch and music21. It transforms raw polyphonic and monophonic MIDI files into sequential token spaces, trains an autoregressive neural network to model musical structure, and synthesizes novel compositions.

```mermaid
graph TD
    subgraph Data Layer
        A[Raw MIDI Files .mid/.midi] --> B[music21 Stream Parser]
        B --> C[Token Extraction: Notes, Chords, Rests]
        C --> D[Vocabulary Builder & Tokenizer]
        D --> E[Bounded Sliding Windows X -> y]
        E --> F[Sequence Tensors dataset.pt]
    end

    subgraph Deep Learning Layer
        F --> G[MusicSequenceDataset & DataLoader]
        G --> H[Token Embedding Layer d=128]
        H --> I[Multi-Layer LSTM d=256]
        I --> J[Dropout & FC Projection]
        J --> K[Softmax Logits over Vocabulary]
        K --> L[CrossEntropyLoss + AdamW]
        L --> M[Model Checkpoints best_model.pt]
    end

    subgraph Generation & Audio Layer
        M --> N[Autoregressive Sampler]
        N -->|Temperature & Top-K| O[Predicted Token Sequence]
        O --> P[music21 Stream Reconstructor]
        P --> Q[MIDI Exporter .mid]
        Q --> R[FluidSynth / SoundFont Renderer]
        R --> S[Audio WAV Output & Web Audio]
        Q --> T[Piano-Roll Plotly Visualizer]
    end

    subgraph User Experience
        U[Streamlit Dashboard & DAW Interface]
        U --> Data Layer
        U --> Deep Learning Layer
        U --> Generation & Audio Layer
    end
```

## Symbolic Representation Pipeline
1. **Pitches & Chords**: Single pitches are encoded as standard scientific pitch strings (e.g., `C4`, `D#5`). Polyphonic simultaneous notes (chords) are sorted by MIDI number and joined with period delimiters (e.g., `C4.E4.G4`). Rests are recorded as `REST`.
2. **Song Boundary Preservation**: Sequences are strictly bounded within individual files. When sliding sequence windows of length $L$ are generated, they never cross song borders, preventing false harmonic leaps between disparate musical keys.
3. **Autoregressive Generation**: The network predicts the next token $y_t \sim P(y_t | y_{t-1}, \dots, y_{t-L})$ using temperature scaling $z / T$ and Top-K filtering.
4. **Deterministic Validation**: Every generated stream is checked for valid MIDI events, BPM calibration, and note counts before being presented to the user.
