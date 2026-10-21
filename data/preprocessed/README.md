# Preprocessed Sequences

This folder contains preprocessed sequence data saved after parsing MIDI files:
- `dataset.pt`: PyTorch dictionary containing training tensor sequences, targets, and vocabulary.
- `vocab.json`: Human-readable token-to-index mapping for inspection.
- `preprocessing_meta.json`: Summary statistics (total tokens, unique pitches, notes per song).
