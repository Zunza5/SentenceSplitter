import mlx.core as mx
import mlx.nn as nn
from typing import Optional, List, Tuple

class ExpertBlock(nn.Module):
    """
    Single Transformer expert in MLX.
    Using standard LayerNorm -> MSA -> residual -> LayerNorm -> FFN -> residual.
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.msa = nn.MultiHeadAttention(d_model, nhead)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # MSA
        h = self.ln1(x)
        
        # If mask is 2D (B, S), convert to 4D additive mask for MLX MHA
        if mask is not None and mask.ndim == 2:
            attn_mask = (1.0 - mask.astype(h.dtype)) * -1e9
            attn_mask = attn_mask[:, None, None, :]
        else:
            attn_mask = mask

        h = self.msa(h, h, h, attn_mask)
        x = x + self.dropout(h)
        # FFN
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x


class MoELayer(nn.Module):
    """
    Mixture of Experts layer in MLX.
    """
    NUM_EXPERTS = 4

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.router = nn.Linear(d_model, self.NUM_EXPERTS)
        self.experts = [
            ExpertBlock(d_model, nhead=nhead, dropout=dropout)
            for _ in range(self.NUM_EXPERTS)
        ]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        # Router logits
        gate_logits = self.router(x)
        gate_probs = mx.softmax(gate_logits, axis=-1)
        
        # Combine expert outputs (Dense MoE)
        expert_outputs = mx.stack([expert(x, mask) for expert in self.experts], axis=-1)
        # expert_outputs: (batch, seq, d_model, NUM_EXPERTS)
        # gate_probs: (batch, seq, NUM_EXPERTS)
        
        output = mx.sum(expert_outputs * mx.expand_dims(gate_probs, 2), axis=-1)

        # ── Load-balancing auxiliary loss ──────
        top_1_indices = mx.argmax(gate_logits, axis=-1)
        
        if mask is not None:
             mask_f = mask.astype(mx.float32)
             valid_tokens = mx.sum(mask_f) + 1e-6
             
             # P: mean probability of each expert over valid tokens
             mask_expanded = mx.expand_dims(mask_f, -1) # (B, S, 1)
             masked_probs = gate_probs * mask_expanded
             P = mx.sum(masked_probs, axis=(0, 1)) / valid_tokens
             
             # f: fraction of valid tokens where expert i is top-1
             f_list = []
             for i in range(self.NUM_EXPERTS):
                 is_top1 = (top_1_indices == i).astype(mx.float32)
                 f_list.append(mx.sum(is_top1 * mask_f) / valid_tokens)
             f = mx.stack(f_list)
        else:
             valid_tokens = gate_logits.size // gate_logits.shape[-1]
             
             # P: mean probability over all tokens
             P = mx.mean(gate_probs, axis=(0, 1))
             
             # f: fraction of all tokens where expert i is top-1
             f_list = []
             for i in range(self.NUM_EXPERTS):
                 is_top1 = (top_1_indices == i).astype(mx.float32)
                 f_list.append(mx.mean(is_top1))
             f = mx.stack(f_list)
             
        aux_loss = self.NUM_EXPERTS * mx.sum(f * P)

        return output, aux_loss


class MultiScaleConv1d(nn.Module):
    """
    Multi-scale 1D CNN in MLX.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        branch_dim = d_model // 3
        branch_dim_last = d_model - 2 * branch_dim
        
        # MLX Conv1d expects (N, L, C) by default!
        self.conv3 = nn.Conv1d(d_model, branch_dim, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, branch_dim, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(d_model, branch_dim_last, kernel_size=7, padding=3)
        
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    
    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, S, C) - MLX default is (N, L, C)
        c3 = self.act(self.conv3(x))
        c5 = self.act(self.conv5(x))
        c7 = self.act(self.conv7(x))
        
        out = mx.concatenate([c3, c5, c7], axis=-1)
        return self.norm(self.drop(out))


class SpacePredictorMLP(nn.Module):
    """
    Sentence boundary predictor in MLX.
    """
    def __init__(self, hidden_dim: int = 2048, d_model: int = 512, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.cnn = MultiScaleConv1d(d_model, dropout)
        self.moe = MoELayer(d_model, nhead=4, dropout=dropout)
        self.post_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        x = self.input_norm(self.input_proj(x))
        
        # CNN (residual)
        x = x + self.cnn(x)
        
        # MoE (residual)
        x_moe, aux_loss = self.moe(x, mask)
        x = x + x_moe
        
        x = self.post_norm(x)
        logits = mx.squeeze(self.classifier(x), -1)
        return mx.sigmoid(logits), aux_loss


class LoRALinear(nn.Module):
    """
    LoRA wrapper for MLX Linear and QuantizedLinear layers.
    """
    def __init__(
        self,
        layer: nn.Module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.layer = layer
        self.layer.freeze() # Freeze the original layer
        
        # Determine logical input/output dims
        if hasattr(layer, "scales") and hasattr(layer, "group_size"):
             # For MLX QuantizedLinear (including mxfp8)
             # Logical dims are inferred from the scales shape and group size
             self.output_dims = layer.scales.shape[0]
             self.input_dims = layer.scales.shape[1] * layer.group_size
        elif hasattr(layer, "weight"):
             # For standard nn.Linear
             self.output_dims, self.input_dims = layer.weight.shape
        else:
             # Fallback for custom quantized layers or others
             self.input_dims = getattr(layer, "input_dims", 1024)
             self.output_dims = getattr(layer, "output_dims", 1024)

        self.lora_a = mx.random.normal((self.input_dims, r)) * (1 / mx.sqrt(r))
        self.lora_b = mx.zeros((r, self.output_dims))
        self.scale = lora_alpha / r
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else (lambda x: x)

    def __call__(self, x: mx.array) -> mx.array:
        result = self.layer(x)
        # LoRA path: (x @ a) @ b
        # Using a @ b instead of (x @ a) @ b if it's more efficient, but (x @ a) @ b is better for small r
        lora = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return result + self.scale * lora


def apply_lora_to_module(module: nn.Module, r: int = 8, lora_alpha: int = 16):
    """
    Recursively applies LoRA to Linear and QuantizedLinear layers in a module.
    Specifically targets attention projections (q, k, v, o).
    """
    for name, child in module.named_modules():
        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            if isinstance(child, (nn.Linear, nn.QuantizedLinear)):
                # Replace the child with a LoRA-wrapped version
                # We need to find the parent and set the attribute
                path = name.split(".")
                curr = module
                for part in path[:-1]:
                    curr = getattr(curr, part)
                
                # Wrap it
                setattr(curr, path[-1], LoRALinear(child, r=r, lora_alpha=lora_alpha))


class TransformerLayerWrapper(nn.Module):
    """
    Wraps MLX transformer layers for fine-tuning.
    """
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = layers

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # MLX attention masks are typically additive or broadcastable
        # We'll assume a standard MLX mask format or handle it if needed
        for layer in self.layers:
            # Most MLX-LM layers take (x, mask)
            x = layer(x, mask=mask)
        return x


class FineTuneSentenceSplitterMLX(nn.Module):
    """
    Combined MLX model for fine-tuning.
    """
    def __init__(self, transformer_layers: List[nn.Module], predictor: SpacePredictorMLP):
        super().__init__()
        self.transformer_layers = TransformerLayerWrapper(transformer_layers)
        self.predictor = predictor

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        # mask is (B, S) boolean mask
        # Convert to 4D additive mask for MLX Transformer layers
        if mask is not None and mask.ndim == 2:
            attn_mask = (1.0 - mask.astype(x.dtype)) * -1e9
            attn_mask = attn_mask[:, None, None, :]
        else:
            attn_mask = mask

        x = self.transformer_layers(x, mask=attn_mask)
        # Note: predictor (the MLP) expects the original boolean mask for MoE balancing
        return self.predictor(x, mask=mask)
