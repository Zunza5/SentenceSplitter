import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with label smoothing.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Label smoothing converts hard targets {0, 1} to soft targets
    {smoothing/2, 1 - smoothing/2}, preventing overconfidence.
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Probabilities (after sigmoid)
            targets: Binary labels (0 or 1)
        """
        # Apply label smoothing: 0 -> ε/2, 1 -> 1 - ε/2
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing / 2

        # Avoid log(0)
        p = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        # Binary cross entropy
        bce_loss = - (targets * torch.log(p) + (1 - targets) * torch.log(1 - p))
        
        # Focal weight
        pt = torch.where(targets >= 0.5, p, 1 - p)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = torch.where(targets >= 0.5, self.alpha, 1.0)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


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
    Mixture of Experts layer with 4 Transformer experts and a learned router.

    The router produces per-token gating weights via softmax over 4 logits.
    All experts process the full input (no sparse routing) and outputs are
    combined as a weighted sum. An auxiliary load-balancing loss encourages
    even distribution of tokens across experts.
    """
    NUM_EXPERTS = 4

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.router = nn.Linear(d_model, self.NUM_EXPERTS)
        self.experts = nn.ModuleList([
            ExpertBlock(d_model, nhead=nhead, dropout=dropout)
            for _ in range(self.NUM_EXPERTS)
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
        # Router gating logits: (batch, seq_len, num_experts)
        gate_logits = self.router(x)
        
        # Softmax over all for auxiliary loss (load-balancing)
        all_probs = F.softmax(gate_logits, dim=-1)

        # Top-2 routing: select top 2 experts per token
        top_k_logits, top_k_indices = torch.topk(gate_logits, k=2, dim=-1)
        # Softmax only over the top 2 to get focused weights
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Map top-k probabilities back to the full number of experts (sparse gateway)
        # gate_probs: (batch, seq_len, num_experts)
        gate_probs = torch.zeros_like(gate_logits).scatter_(-1, top_k_indices, top_k_probs.to(gate_logits.dtype))

        # Run all experts (running all 4 is more efficient than sparse indexing here)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (batch, seq_len, d_model, num_experts)

        # Weighted combination: only top-2 experts contribute
        output = (expert_outputs * gate_probs.unsqueeze(2)).sum(dim=-1)

        # ── Load-balancing auxiliary loss (Switch Transformer style) ──────
        # We use all_probs to encourage balanced exploration
        if mask is not None:
            valid_probs = all_probs[mask]
        else:
            valid_probs = all_probs.reshape(-1, self.NUM_EXPERTS)

        # f is the fraction of tokens where expert i is the top-1 choice
        assignments = valid_probs.argmax(dim=-1)
        f = torch.zeros(self.NUM_EXPERTS, device=x.device)
        for i in range(self.NUM_EXPERTS):
            f[i] = (assignments == i).float().mean()

        # P is the average probability mass assigned to each expert
        P = valid_probs.mean(dim=0)
        
        # Loss minimizes non-uniformity
        aux_loss = self.NUM_EXPERTS * (f * P).sum()

        return output, aux_loss


class MultiScaleConv1d(nn.Module):
    """
    Multi-scale 1D CNN with parallel branches at different kernel sizes.
    
    Each branch captures patterns at a different receptive field:
      - Kernel 3: punctuation + space patterns  (". ")
      - Kernel 5: short abbreviations           ("Art. ")
      - Kernel 7: wider context around boundary  (". A")
    
    All branches are concatenated and projected back to d_model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Each branch outputs d_model // 3 channels (rounded up for k=3)
        branch_dim = d_model // 3
        branch_dim_last = d_model - 2 * branch_dim  # handle remainder
        
        self.conv3 = nn.Conv1d(d_model, branch_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, branch_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(d_model, branch_dim_last, kernel_size=7, padding=3)
        
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Conv1d expects (batch, channels, seq_len)
        x_t = x.transpose(1, 2)
        
        # Parallel branches
        c3 = self.act(self.conv3(x_t))   # (B, branch_dim, S)
        c5 = self.act(self.conv5(x_t))   # (B, branch_dim, S)
        c7 = self.act(self.conv7(x_t))   # (B, branch_dim_last, S)
        
        # Concatenate and transpose back
        out = torch.cat([c3, c5, c7], dim=1)  # (B, d_model, S)
        out = out.transpose(1, 2)              # (B, S, d_model)
        
        out = self.drop(out)
        out = self.norm(out)
        
        return out


class SpacePredictorMLP(nn.Module):
    """
    Token-level boundary prediction model.
    
    Architecture:
      1. Linear projection from LLM hidden_dim → d_model
      2. Learnable positional encoding
      3. Multi-scale CNN (kernels 3, 5, 7) for local pattern extraction
      4. MoE layer: 4 Transformer experts with learned routing
      5. Classifier head → per-token binary prediction
    """
    def __init__(self, hidden_dim: int = 2048, d_model: int = 512, dropout: float = 0.3, max_seq_len: int = 4096):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 1. Linear Projection + LayerNorm
        self.input_proj = nn.Linear(hidden_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # 2. Learnable Positional Encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 3. Multi-scale CNN (local patterns: punctuation, abbreviations, etc.)
        self.cnn = MultiScaleConv1d(d_model=d_model, dropout=dropout)
        
        # 4. Mixture of Experts (4 Transformer experts, 4 heads each)
        self.moe = MoELayer(d_model=d_model, nhead=4, dropout=dropout)

        # 5. Post-MoE normalization
        self.post_norm = nn.LayerNorm(d_model)
        
        self.drop = nn.Dropout(dropout)
        
        # 6. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
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
        # 1. Project from LLM hidden_dim to d_model
        x = self.input_proj(x)
        x = self.input_norm(x)

        # 2. Add learnable positional encoding
        #seq_len = x.size(1)
        #positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        #x = x + self.pos_embedding(positions)

        # 3. Multi-scale CNN (local features) with residual
        x_cnn = self.cnn(x)
        x = x + x_cnn  # residual around CNN

        # 4. Mixture of Experts (global context) with residual
        x_moe, aux_loss = self.moe(x, mask=mask)
        x = x + x_moe  # residual around MoE

        # 5. Final normalization
        x = self.post_norm(x)
        
        # 6. Classification
        logits = self.classifier(x).squeeze(-1)
        
        return torch.sigmoid(logits), aux_loss
class TransformerLayerWrapper(nn.Module):
    """
    Wraps a set of transformer layers to be used in the fine-tuning pipeline.
    Handles the attention mask conversion if necessary.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) bool, True for non-padding tokens
        """
        # Convert bool mask to transformers attention mask (0 for mask, 1 for keep -> then to additive mask)
        # However, Qwen layers usually expect 4D attention mask 
        # (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len)
        # For simplicity, if attention_mask is provided, we can try to pass it as is
        # but most transformer blocks want it in a specific broadcastable shape.
        
        extended_mask = None
        if attention_mask is not None:
            # Transformers usually uses: (batch, 1, 1, seq_len)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.to(dtype=x.dtype)  # fp16/fp32
            extended_mask = (1.0 - extended_mask) * torch.finfo(x.dtype).min

        for layer in self.layers:
            # QwenDecoderLayer returns a tuple (hidden_states, self_attn_weights ..., etc.)
            # We must be careful about the output format. Qwen2/Qwen3 typically returns a tuple
            # where the first element is the hidden_states.
            layer_outputs = layer(x, attention_mask=extended_mask)
            if isinstance(layer_outputs, (tuple, list)):
                x = layer_outputs[0]
            else:
                x = layer_outputs
            
        return x


class FineTuneSentenceSplitter(nn.Module):
    """
    Combines fine-tunable transformer layers with the SpacePredictorMLP.
    """
    def __init__(self, transformer_layers: nn.ModuleList, predictor: SpacePredictorMLP):
        super().__init__()
        self.transformer_layers = TransformerLayerWrapper(transformer_layers)
        self.predictor = predictor

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, hidden_dim) intermediate embeddings
            mask: (batch, seq_len) bool mask
        """
        # 1. Pass through transformer layers
        x = self.transformer_layers(x, attention_mask=mask)
        
        # 2. Pass through the MLP predictor
        return self.predictor(x, mask=mask)
