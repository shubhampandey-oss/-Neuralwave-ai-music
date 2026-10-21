"""
Command-Line Interface for Composing Music with NeuralWave AI.
Usage:
    python scripts/generate.py --checkpoint models/best_model.pt --output outputs/my_track.mid --num-notes 64 --temp 0.85
"""

import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.generate import generate_composition


def main():
    parser = argparse.ArgumentParser(description="Generate new MIDI composition using trained NeuralWave AI model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Path for generated output .mid file")
    parser.add_argument("--num-notes", type=int, default=None, help="Number of notes to generate")
    parser.add_argument("--temp", type=float, default=None, help="Sampling temperature (e.g. 0.8)")
    parser.add_argument("--bpm", type=int, default=None, help="Tempo in beats per minute")
    parser.add_argument("--instrument", type=str, default=None, help="Instrument name")
    parser.add_argument("--seed-note", type=str, default=None, help="Optional starting pitch (e.g. C4)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, cuda)")
    args = parser.parse_args()

    cfg = load_config()

    ckpt_path = Path(args.checkpoint or (Path(cfg["training"]["checkpoint_dir"]) / "best_model.pt"))
    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        print("Please train a model first using: python scripts/train.py or through the Streamlit web dashboard.")
        sys.exit(1)

    default_out_name = f"composition_{int(time.time())}.mid"
    out_path = Path(args.output or (Path(cfg["generation"]["output_dir"]) / default_out_name))
    num_notes = args.num_notes or cfg["generation"]["num_notes"]
    temp = args.temp or cfg["generation"]["temperature"]
    bpm = args.bpm or cfg["generation"]["bpm"]
    instrument = args.instrument or cfg["generation"]["instrument"]
    seed_note = args.seed_note
    rand_seed = args.seed or cfg["generation"]["seed"]

    print("=" * 60)
    print("  NEURALWAVE AI — MUSIC GENERATOR")
    print("=" * 60)
    print(f"Model Checkpoint:  {ckpt_path}")
    print(f"Output File:       {out_path}")
    print(f"Note Count:        {num_notes}")
    print(f"Temperature:       {temp}")
    print(f"Tempo (BPM):       {bpm}")
    print(f"Instrument:        {instrument}")
    print(f"Seed Pitch:        {seed_note or 'Auto-selected'}")
    print(f"Random Seed:       {rand_seed}")
    print("-" * 60)

    result = generate_composition(
        checkpoint_path=ckpt_path,
        output_path=out_path,
        num_notes=num_notes,
        temperature=temp,
        bpm=bpm,
        instrument=instrument,
        seed_token=seed_note,
        random_seed=rand_seed,
        title=f"NeuralWave Compo ({instrument})",
        device_name=args.device,
    )

    print(f"[SUCCESS] Composition successfully generated!")
    print(f"MIDI File:         {result['midi_path']}")
    print(f"Duration:          {result['metadata']['duration_seconds']} seconds")
    print(f"Events:            {len(result['tokens'])} notes")
    print(f"First 10 notes:    {result['tokens'][:10]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
