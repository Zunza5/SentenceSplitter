import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Stable Focal Loss for binary classification operating directly on logits.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw outputs from the model (before sigmoid)
            targets: Binary labels (0 or 1)
        """
        # Numerically stable Binary Cross Entropy directly from logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Calculate probabilities for the focal weight
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        
        focal_weight = (1 - pt) ** self.gamma
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
    Semplice Feed-Forward Network per il MoE.
    Molto più leggero e matematicamente corretto rispetto alla Self-Attention in questo step.
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        # Ignoriamo 'nhead' ma lo teniamo per compatibilità con l'init originale
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


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
        # 1. Router probabilities
        gate_logits = self.router(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 2. Hard top-k routing
        topk_vals, topk_idx = torch.topk(gate_probs, k=self.top_k, dim=-1)
        hard_weights = torch.zeros_like(gate_probs)
        hard_weights.scatter_(-1, topk_idx, topk_vals)
        hard_weights = hard_weights / hard_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(x)
        
        # TROVATA PER MAC/MPS: Calcolo denso ma con MLP leggeri.
        # Evita masking e gather/scatter che distruggono le prestazioni di Apple Silicon.
        
        # Identifichiamo quali esperti sono attivi nel batch per saltare quelli totalmente inutilizzati
        if mask is not None:
            active_selector = hard_weights * mask.unsqueeze(-1).float()
        else:
            active_selector = hard_weights
            
        expert_usage = active_selector.sum(dim=(0, 1))
        active_experts = (expert_usage > 0).nonzero(as_tuple=False).flatten().tolist()

        for expert_idx in active_experts:
            # Calcolo super-ottimizzato (MatMul puro) su tutta la sequenza
            expert_out = self.experts[expert_idx](x) 
            
            # Moltiplichiamo per i pesi (i token non assegnati a questo esperto hanno peso 0)
            weight = hard_weights[:, :, expert_idx].unsqueeze(-1)
            output = output + (expert_out * weight)

        # ── Load-balancing auxiliary loss ──────
        if mask is not None:
            valid_probs = gate_probs[mask.bool()]
        else:
            valid_probs = gate_probs.reshape(-1, self.num_experts)

        if valid_probs.numel() == 0:
            aux_loss = torch.tensor(0.0, device=x.device)
            return output, aux_loss

        assignments = valid_probs.argmax(dim=-1)
        f = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f[i] = (assignments == i).float().mean()

        P = valid_probs.mean(dim=0)
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
        
        return logits, aux_loss