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
        
        self.hidden_dim = hidden_dim
        
        # 1. Feature Extractor (Contesto Locale)
        # kernel_size=5 con padding=2 significa che per il token [i], 
        # la rete guarda [i-2, i-1, i, i+1, i+2]
        self.conv = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=256, 
            kernel_size=5, 
            padding=2
        )
        
        # 2. Stabilizzazione del Segnale
        self.norm = nn.LayerNorm(256)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        # 3. Classificatore Finale
        self.classifier = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) character-level embeddings
        Returns:
            (batch, seq_len) space probabilities in [0, 1]
        """
        x = x.transpose(1, 2)
        
        # Applichiamo la convoluzione spaziale
        x = self.conv(x)
        
        # Torniamo alla shape standard (batch, seq_len, features) per LayerNorm e Linear
        x = x.transpose(1, 2)
        
        # Raffinamento e classificazione
        x = self.norm(x)
        x = self.gelu(x)
        x = self.drop(x)
        logits = self.classifier(x).squeeze(-1)  # (batch, seq_len)
        return torch.sigmoid(logits)
