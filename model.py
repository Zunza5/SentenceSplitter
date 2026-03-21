import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Probabilities (after sigmoid)
            targets: Binary labels (0 or 1)
        """
        # Avoid log(0)
        p = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        # Binary cross entropy
        bce_loss = - (targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
        
        # Focal weight
        pt = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = torch.where(targets == 1, self.alpha, 1.0)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiScaleConv1d(nn.Module):
    """
    Multi-scale CNN block to capture both local punctuation and wider context.
    Uses three parallel branches with different receptive fields.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        
        # Split out_channels among 3 branches to keep parameter count efficient
        branch_channels = out_channels // 3
        remainder = out_channels - (2 * branch_channels)
        
        # Branch 1: Local context (e.g., standard period + space)
        self.branch1 = nn.Conv1d(
            in_channels, branch_channels, kernel_size=3, padding=1
        )
        
        # Branch 2: Medium context (e.g., abbreviations like 'Dott.', 'ecc.')
        self.branch2 = nn.Conv1d(
            in_channels, branch_channels, kernel_size=5, padding=2
        )
        
        # Branch 3: Wide context using dilation (e.g., quotes, brackets)
        self.branch3 = nn.Conv1d(
            in_channels, remainder, kernel_size=7, padding=6, dilation=2
        )
        
        # Spatial Dropout: drops entire feature channels.
        # Expects input shape (batch, channels, seq_len)
        self.spatial_dropout = nn.Dropout1d(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        """
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # Concatenate features from all scales along the channel dimension
        merged = torch.cat([out1, out2, out3], dim=1)
        
        # Apply Spatial Dropout to the concatenated feature maps
        return self.spatial_dropout(merged)


class SpacePredictorMLP(nn.Module):
    """
    Lightweight model for character-level sentence boundary prediction.
    Combines Qwen embeddings and Multi-Scale CNNs. 
    Self-Attention removed to prevent overfitting on small datasets.
    """
    def __init__(self, hidden_dim: int = 2048, cnn_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Residual Projection
        # Projects original LLM embeddings to match CNN output dimension
        self.residual_proj = nn.Linear(hidden_dim, cnn_dim)
        
        # 2. Multi-scale Feature Extractor with integrated Spatial Dropout
        self.multi_scale_conv = MultiScaleConv1d(
            in_channels=hidden_dim, out_channels=cnn_dim, dropout=dropout
        )
        
        # 3. Stabilization Layers
        self.norm = nn.LayerNorm(cnn_dim)
        self.gelu = nn.GELU()
        
        # Standard dropout for the fully connected layers
        self.drop = nn.Dropout(dropout)
        
        # 4. Deep Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim, 128),
            nn.GELU(),
            self.drop,
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            logits: (batch, seq_len) raw logits. Do not apply Sigmoid here.
        """
        # Save projected input for residual connection
        res = self.residual_proj(x)
        
        # CNN requires shape (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Apply multi-scale convolutions and spatial dropout
        x_conv = self.multi_scale_conv(x_conv)
        
        # Transpose back to (batch, seq_len, channels)
        x_conv = x_conv.transpose(1, 2)
        
        # Add residual connection and apply activation
        x = x_conv + res
        x = self.norm(x)
        x = x + self.gelu(x) 
        
        # Final classification to generate logits
        logits = self.classifier(x).squeeze(-1)
        
        return torch.sigmoid(logits)