"""
PyTorch neural network architecture for symbolic music generation.
Multi-layer LSTM with token embedding and multi-layer classification head.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn


class NeuralWaveLSTM(nn.Module):
    """
    Autoregressive LSTM network for predicting musical pitch/chord tokens.
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_projection = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim // 2, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights for training stability."""
        nn.init.xavier_uniform_(self.fc_projection.weight)
        nn.init.zeros_(self.fc_projection.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Tensor of token indices, shape (batch_size, seq_len)
            hidden: Optional (h_0, c_0) tuple
        Returns:
            logits: Next token unnormalized logits, shape (batch_size, vocab_size)
            hidden: Updated (h_n, c_n) tuple
        """
        # Embed tokens: (B, L) -> (B, L, E)
        embedded = self.embedding(x)

        # LSTM forward: (B, L, E) -> (B, L, H)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Autoregressive next token prediction: use the output of the final time step
        last_step = lstm_out[:, -1, :]  # (B, H)
        dropped = self.dropout(last_step)
        projected = self.relu(self.fc_projection(dropped))
        logits = self.classifier(projected)  # (B, V)

        return logits, hidden

    def count_parameters(self) -> int:
        """Counts total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Returns architecture configuration."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Selects target execution device ('cpu', 'cuda', or 'auto').
    """
    req_lower = requested.strip().lower()
    if req_lower == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if req_lower == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
