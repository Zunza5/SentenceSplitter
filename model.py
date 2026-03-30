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
        
        # Split out_channels among 4 branches to keep parameter count efficient
        branch_channels = out_channels // 4
        remainder = out_channels - (3 * branch_channels)
        
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
            in_channels, branch_channels, kernel_size=7, padding=6, dilation=2
        )

        self.branch4 = nn.Conv1d(
            in_channels, remainder, 
            kernel_size=11, 
            padding=10, 
            dilation=2
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
        out4 = self.branch4(x)
        
        # Concatenate features from all scales along the channel dimension
        merged = torch.cat([out1, out2, out3, out4], dim=1)
        
        # Apply Spatial Dropout to the concatenated feature maps
        return self.spatial_dropout(merged)


class ExpertBlock(nn.Module):
    """
    Single Transformer expert with residual connection and layer norm.
    Uses nn.TransformerEncoderLayer for a standard self-attention + FFN block.
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        out = self.transformer_layer(x)
        return self.norm(out + x)  # residual + norm


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with hard top-k routing.

    The router produces per-token logits over experts. Only the top-k experts
    per token are kept, then re-normalized and used to mix expert outputs.
    This is a sparse hard routing variant where non-selected experts receive
    exactly zero weight for that token.
    """
    NUM_EXPERTS = 8
    TOP_K = 2

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dropout: float = 0.1,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K,
    ):
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be >= 2")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(d_model, self.num_experts)
        self.experts = nn.ModuleList([
            ExpertBlock(d_model, nhead=nhead, dropout=dropout)
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) bool, True for valid positions
        Returns:
            output: (batch, seq_len, d_model) — weighted expert mixture
            aux_loss: scalar — load-balancing loss
        """
        # Router probabilities over all experts: (batch, seq_len, num_experts)
        gate_logits = self.router(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Hard top-k routing: keep only the best-k experts per token.
        topk_vals, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        hard_weights = torch.zeros_like(gate_probs)
        hard_weights.scatter_(-1, topk_idx, topk_vals)
        hard_weights = hard_weights / hard_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # Sparse execution: run only experts selected by top-k in this batch.
        if mask is not None:
            active_selector = hard_weights * mask.unsqueeze(-1).float()
        else:
            active_selector = hard_weights
        active_experts = (active_selector.sum(dim=(0, 1)) > 0).nonzero(as_tuple=False).flatten().tolist()

        output = torch.zeros_like(x)
        for expert_idx in active_experts:
            expert_out = self.experts[expert_idx](x)  # (B, S, d_model)
            weight = hard_weights[:, :, expert_idx].unsqueeze(-1)  # (B, S, 1)
            output = output + (expert_out * weight)

        # ── Load-balancing auxiliary loss (Switch Transformer style) ──────
        # f_i = fraction of tokens dispatched to expert i (based on argmax)
        # P_i = mean router probability for expert i
        # L_balance = num_experts * sum(f_i * P_i)
        if mask is not None:
            # Only consider valid (non-padding) positions
            valid_probs = gate_probs[mask]  # (num_valid, num_experts)
        else:
            valid_probs = gate_probs.reshape(-1, self.num_experts)

        if valid_probs.numel() == 0:
            aux_loss = torch.tensor(0.0, device=x.device)
            return output, aux_loss

        # f_i: fraction of tokens where expert i has the highest gate
        assignments = valid_probs.argmax(dim=-1)  # (num_valid,)
        f = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f[i] = (assignments == i).float().mean()

        # P_i: mean probability assigned to expert i
        P = valid_probs.mean(dim=0)  # (num_experts,)

        aux_loss = self.num_experts * (f * P).sum()

        return output, aux_loss


class SpacePredictorMLP(nn.Module):
    """
    Character-level space prediction model with Mixture of Experts.
    
    Architecture:
      1. Multi-scale CNN extracts local features at different receptive fields
      2. MoE layer: a router dispatches CNN features to 8 Transformer experts
         using hard top-2 routing
      3. Classifier head produces per-character binary predictions
    """
    def __init__(
        self,
        hidden_dim: int = 2048,
        cnn_dim: int = 256,
        d_model: int | None = None,
        dropout: float = 0.3,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()

        if d_model is not None:
            cnn_dim = d_model
        
        self.hidden_dim = hidden_dim
        self.cnn_dim = cnn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 1. Residual Projection
        self.residual_proj = nn.Linear(hidden_dim, cnn_dim)
        
        # 2. Multi-scale Feature Extractor with integrated Spatial Dropout
        self.multi_scale_conv = MultiScaleConv1d(
            in_channels=hidden_dim, out_channels=cnn_dim, dropout=dropout
        )
        
        # 3. Mixture of Experts (replaces single self-attention)
        self.moe = MoELayer(
            d_model=cnn_dim,
            nhead=4,
            dropout=dropout,
            num_experts=num_experts,
            top_k=top_k,
        )

        # 4. Stabilization Layers
        self.norm = nn.LayerNorm(cnn_dim)
        self.gelu = nn.GELU()
        
        # Standard dropout for the fully connected layers
        self.drop = nn.Dropout(dropout)
        
        # 5. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim, 128),
            nn.GELU(),
            self.drop,
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) bool mask for valid positions (optional)
        Returns:
            preds: (batch, seq_len) probabilities after sigmoid
            aux_loss: scalar load-balancing loss from the MoE router
        """
        # Save projected input for residual connection
        res = self.residual_proj(x)
        
        # CNN requires shape (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        x_conv = self.multi_scale_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # back to (batch, seq_len, cnn_dim)
        
        # Add residual connection and apply activation
        x = x_conv + res
        x = self.norm(x)
        x = x + self.gelu(x)

        # Mixture of Experts
        x_moe, aux_loss = self.moe(x, mask=mask)
        x = x_moe + x  # residual around MoE

        x = self.norm(x)
        
        # Final classification
        logits = self.classifier(x).squeeze(-1)
        
        return torch.sigmoid(logits), aux_loss