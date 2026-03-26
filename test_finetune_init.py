import torch
import torch.nn as nn
from wordSplitter.model import SpacePredictorMLP, FineTuneSentenceSplitter
from wordSplitter.embeddings import load_language_model, MODEL_NAME
from transformers import AutoConfig

def test_init():
    print(f"Testing initialization for {MODEL_NAME}...")
    device = torch.device("cpu")
    
    # Check config
    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = config.num_hidden_layers
    print(f"Model has {num_layers} layers.")
    
    fine_tune_layers = 2
    effective_layer_idx = num_layers - fine_tune_layers - 1
    print(f"Targeting intermediate layer: {effective_layer_idx}")
    
    # Load model
    llm_full, _ = load_language_model("transformers", device)
    
    # Extract layers
    all_layers = llm_full.model.layers
    fine_tune_start = len(all_layers) - fine_tune_layers
    target_layers = nn.ModuleList([all_layers[i] for i in range(fine_tune_start, len(all_layers))])
    
    print(f"Extracted {len(target_layers)} transformer layers.")
    
    # Initialize MLP
    mlp = SpacePredictorMLP(hidden_dim=config.hidden_size, d_model=256)
    
    # Wrap
    model = FineTuneSentenceSplitter(target_layers, mlp).to(device)
    print("FineTuneSentenceSplitter initialized successfully.")
    
    # Test forward pass with dummy data
    batch_size = 2
    seq_len = 16
    dummy_input = torch.randn(batch_size, seq_len, config.hidden_size).to(device)
    dummy_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to(device)
    
    print("Running dummy forward pass...")
    preds, aux_loss = model(dummy_input, mask=dummy_mask)
    print(f"Forward pass complete. Predictions shape: {preds.shape}")
    
    # Check gradients
    print("Checking gradients...")
    loss = preds.mean()
    loss.backward()
    
    # Check if a parameter in the transformer layers has a gradient
    sample_param = next(model.transformer_layers.parameters())
    if sample_param.grad is not None:
        print("✓ Gradients are flowing to transformer layers.")
    else:
        print("✗ Gradients NOT found in transformer layers.")

if __name__ == "__main__":
    test_init()
