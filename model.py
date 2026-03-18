"""
MLP model for character-level space prediction.

Takes character-level embeddings (from Minerva last hidden state)
and predicts P(space) after each character.
"""

import torch
import torch.nn as nn


class SpacePredictorMLP(nn.Module):
    """
    3-layer MLP for binary space prediction at each character position.

    Architecture:
        Linear(hidden_dim → 512) → ReLU → Dropout
        Linear(512 → 256)        → ReLU → Dropout
        Linear(256 → 1)          → Sigmoid

    Input shape:  (batch, seq_len, hidden_dim)
    Output shape: (batch, seq_len, 1)
    """

    def __init__(self, hidden_dim: int = 2048, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) character-level embeddings
        Returns:
            (batch, seq_len) space probabilities in [0, 1]
        """
        logits = self.net(x).squeeze(-1)  # (batch, seq_len)
        return torch.sigmoid(logits)
