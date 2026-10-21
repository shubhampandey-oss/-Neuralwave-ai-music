"""
Automated unit tests for PyTorch NeuralWaveLSTM architecture and gradient steps.
"""

import pytest
import torch
import torch.nn as nn
from src.model import NeuralWaveLSTM, resolve_device


def test_model_initialization():
    vocab_size = 50
    model = NeuralWaveLSTM(vocab_size=vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2)
    assert model.vocab_size == 50
    assert model.count_parameters() > 0


def test_model_forward_pass_shape():
    batch_size = 4
    seq_len = 16
    vocab_size = 40
    model = NeuralWaveLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, num_layers=1)

    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    logits, hidden = model(dummy_input)

    assert logits.shape == (batch_size, vocab_size)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2


def test_model_training_step():
    vocab_size = 30
    model = NeuralWaveLSTM(vocab_size=vocab_size, embedding_dim=32, hidden_dim=64, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randint(0, vocab_size, (2, 8), dtype=torch.long)
    y = torch.tensor([5, 12], dtype=torch.long)

    # Initial forward & loss
    model.train()
    optimizer.zero_grad()
    logits, _ = model(x)
    initial_loss = criterion(logits, y)
    initial_loss.backward()
    optimizer.step()

    assert initial_loss.item() > 0


def test_resolve_device():
    cpu_dev = resolve_device("cpu")
    assert cpu_dev.type == "cpu"
    auto_dev = resolve_device("auto")
    assert auto_dev.type in ["cpu", "cuda"]
