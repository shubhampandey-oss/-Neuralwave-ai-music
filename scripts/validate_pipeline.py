"""
End-to-End Pipeline Verification Script for NeuralWave AI.
Executes an automated audit of the entire workflow on CPU:
Synthetic Data -> Preprocessing -> Model Init -> 1-Epoch Training -> Checkpoint Save -> Load -> Autoregressive Generation -> MIDI Export -> MIDI Validation.
"""

from pathlib import Path
import sys
import os

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.midi_utils import create_synthetic_midi, validate_midi_file
from src.preprocessing import preprocess_midi_dataset
from src.train import train_model
from src.generate import generate_composition
from src.validation import get_system_environment_info, validate_model_checkpoint


def run_pipeline_validation() -> bool:
    print("=" * 70)
    print("  NEURALWAVE AI — END-TO-END PIPELINE VERIFICATION")
    print("=" * 70)

    # 1. System info
    env_info = get_system_environment_info()
    print(f"[*] Python: {env_info['python_version']} | PyTorch: {env_info['torch_version']} | music21: {env_info['music21_version']}")
    print(f"[*] Hardware Device: {env_info['device_name']} (CUDA available: {env_info['cuda_available']})")

    # 2. Synthetic MIDI Generation
    test_data_dir = ROOT_DIR / "data" / "midi"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    synth_file = test_data_dir / "synthetic_sample.mid"
    print(f"\n[1/7] Creating synthetic test MIDI at {synth_file.name}...")
    create_synthetic_midi(synth_file, title="Synthetic Pipeline Fixture", bpm=120)
    assert synth_file.exists() and synth_file.stat().st_size > 0, "Synthetic MIDI creation failed!"
    print("      -> Successfully generated synthetic test MIDI fixture.")

    # 3. Preprocessing
    preprocessed_dir = ROOT_DIR / "data" / "preprocessed"
    print(f"\n[2/7] Preprocessing dataset into {preprocessed_dir} (seq_len=8 for quick test)...")
    prep_meta = preprocess_midi_dataset(
        data_dir=test_data_dir,
        output_dir=preprocessed_dir,
        sequence_length=8,
        min_notes_per_file=4,
    )
    print(f"      -> Discovered {prep_meta['discovered_files']} files, extracted {prep_meta['total_notes_extracted']} notes.")
    print(f"      -> Built vocabulary of {prep_meta['vocab_size']} tokens, {prep_meta['sequence_count']} sequences.")
    assert (preprocessed_dir / "dataset.pt").exists(), "Preprocessed dataset.pt not created!"

    # 4. Training
    models_dir = ROOT_DIR / "models"
    print(f"\n[3/7] Training NeuralWaveLSTM for 2 miniature test epochs on CPU...")
    train_results = train_model(
        dataset_dir=preprocessed_dir,
        checkpoint_dir=models_dir,
        epochs=2,
        batch_size=4,
        learning_rate=0.005,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
        dropout=0.1,
        device_name="cpu",
        val_split=0.0,
    )
    print(f"      -> Training {train_results['status']} in {train_results['training_time_seconds']}s.")
    print(f"      -> Recorded Loss: {train_results['final_loss']}")
    best_ckpt = models_dir / "best_model.pt"
    assert best_ckpt.exists(), "best_model.pt was not saved!"

    # 5. Checkpoint Verification
    print(f"\n[4/7] Validating saved model checkpoint {best_ckpt.name}...")
    ckpt_val = validate_model_checkpoint(best_ckpt)
    assert ckpt_val["valid"], f"Checkpoint validation failed: {ckpt_val.get('error')}"
    print(f"      -> Checkpoint valid: vocab_size={ckpt_val['vocab_size']}, epochs={ckpt_val['epochs_recorded']}.")

    # 6. Autoregressive Generation & MIDI Export
    outputs_dir = ROOT_DIR / "outputs"
    test_output_mid = outputs_dir / "pipeline_test.mid"
    print(f"\n[5/7] Autoregressively generating composition (32 notes) with NeuralWaveLSTM...")
    gen_result = generate_composition(
        checkpoint_path=best_ckpt,
        output_path=test_output_mid,
        num_notes=32,
        temperature=0.85,
        bpm=120,
        instrument="Acoustic Grand Piano",
        random_seed=42,
        title="Pipeline Verification Composition",
        device_name="cpu",
    )
    print(f"      -> Successfully generated {len(gen_result['tokens'])} symbolic notes.")
    print(f"      -> Sample notes: {gen_result['tokens'][:6]}")
    assert test_output_mid.exists(), "Output MIDI file does not exist!"

    # 7. Output MIDI Verification
    print(f"\n[6/7] Re-opening and verifying exported MIDI structure...")
    val_mid = validate_midi_file(test_output_mid)
    assert val_mid["valid"], f"Generated MIDI validation failed: {val_mid.get('error')}"
    print(f"      -> Valid MIDI file confirmed! Notes: {val_mid['note_count']}, Duration: {val_mid['duration_seconds']}s.")

    # 8. Piano Roll extraction check
    print(f"\n[7/7] Testing Piano Roll visualization data extraction...")
    assert len(gen_result["piano_roll"]) > 0, "No piano roll events extracted!"
    print(f"      -> Piano roll events extracted: {len(gen_result['piano_roll'])} note bars.")

    print("\n" + "=" * 70)
    print("  [SUCCESS] ALL 7 STAGES OF THE NEURALWAVE AI PIPELINE PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    try:
        success = run_pipeline_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FAILED] Pipeline validation error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
