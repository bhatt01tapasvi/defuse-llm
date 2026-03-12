"""
Efficient MLP modules that support selective neuron computation.

Instead of zeroing out activations and still performing full-size matrix multiplies,
these modules pre-slice weight matrices to only include the kept neurons, resulting
in physically smaller matmuls that are genuinely faster on GPU.

Tier 0: Contiguous pre-sliced weights (eliminates wasted FLOPs)
Tier 1: torch.compile with CUDA graphs (kernel fusion + launch overhead elimination)
Tier 2: Fused gate+up projection (single matmul instead of two)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientLlamaMLP(nn.Module):
    """Drop-in replacement for LLaMA/Qwen3/Mistral MLP with selective neuron computation.
    
    These architectures all share the same MLP structure:
        gate_proj: (intermediate_size, hidden_size)
        up_proj:   (intermediate_size, hidden_size) 
        down_proj: (hidden_size, intermediate_size)
        output = down_proj(act_fn(gate_proj(x)) * up_proj(x))
    """
    
    def __init__(self, original_mlp):
        super().__init__()
        # Keep references to original full-size layers (no copy)
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn
        
        # Reduced weight storage (populated after neuron selection)
        self._reduced_ready = False
        self._kept_indices = None
        # Tier 0: Pre-sliced contiguous weights
        self._down_w_reduced = None   # (hidden_size, k)
        # Tier 2: Fused gate+up weight
        self._gate_up_w_reduced = None  # (2*k, hidden_size)
    
    def prepare_reduced_weights(self, kept_indices: torch.Tensor):
        """Pre-slice weights to only the neurons we want to keep.
        
        Args:
            kept_indices: 1D tensor of neuron indices to keep, shape (k,)
        
        Tier 0: Makes sliced weights contiguous for fast cuBLAS GEMV.
        Tier 2: Fuses gate_proj and up_proj rows into a single (2*k, hidden) weight
                 so we do ONE matmul instead of two, reading x only once.
        """
        # Ensure indices are on the same device as weights (critical for multi-GPU)
        weight_device = self.gate_proj.weight.device
        kept_indices = kept_indices.to(weight_device)
        self._kept_indices = kept_indices
        
        # Tier 0: Pre-slice and make contiguous
        gate_w_reduced = self.gate_proj.weight.data[kept_indices].contiguous()  # (k, hidden)
        up_w_reduced = self.up_proj.weight.data[kept_indices].contiguous()      # (k, hidden)
        self._down_w_reduced = self.down_proj.weight.data[:, kept_indices].contiguous()  # (hidden, k)
        
        # Tier 2: Fuse gate + up into single weight for one matmul
        self._gate_up_w_reduced = torch.cat([gate_w_reduced, up_w_reduced], dim=0)  # (2*k, hidden)
        
        self._reduced_ready = True
        
        # Free the individual slices (fused weight replaces them)
        del gate_w_reduced, up_w_reduced
    
    def clear_reduced_weights(self):
        """Revert to full computation (e.g., on new prompt)."""
        self._reduced_ready = False
        self._kept_indices = None
        self._gate_up_w_reduced = None
        self._down_w_reduced = None
    
    def forward(self, x):
        if self._reduced_ready:
            # EFFICIENT PATH: smaller matmuls with fused gate+up
            # x: (batch, seq_len, hidden_size)
            
            # Tier 2: Single fused matmul for gate + up projections
            # (B, S, hidden) @ (2*k, hidden).T → (B, S, 2*k)
            gate_up_out = F.linear(x, self._gate_up_w_reduced)
            gate_out, up_out = gate_up_out.chunk(2, dim=-1)  # each (B, S, k)
            
            # Activation + elementwise multiply
            intermediate = self.act_fn(gate_out) * up_out  # (B, S, k)
            
            # Down projection with reduced weight
            # (B, S, k) @ (hidden, k).T → (B, S, hidden)
            output = F.linear(intermediate, self._down_w_reduced)
            return output
        else:
            # FULL PATH: standard computation (during prefill / score accumulation)
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class EfficientGPT2MLP(nn.Module):
    """Drop-in replacement for GPT2 MLP with selective neuron computation.
    
    GPT2 MLP structure:
        c_fc:   first projection (hidden → intermediate), uses Conv1D
        act:    activation function (GELU)
        c_proj: second projection (intermediate → hidden), uses Conv1D
        output = c_proj(act(c_fc(x)))
    
    Note: GPT2 uses Conv1D where weight is (input, output) instead of (output, input).
          c_fc.weight:   (hidden_size, intermediate_size)
          c_proj.weight: (intermediate_size, hidden_size)
    """
    
    def __init__(self, original_mlp):
        super().__init__()
        # Keep references to original layers
        self.c_fc = original_mlp.c_fc
        self.c_proj = original_mlp.c_proj
        self.act = original_mlp.act
        # Preserve dropout if present
        self.dropout = original_mlp.dropout if hasattr(original_mlp, 'dropout') else None
        
        # Reduced weight storage
        self._reduced_ready = False
        self._kept_indices = None
        self._fc_w_reduced = None      # (hidden, k) — Conv1D format
        self._fc_b_reduced = None      # (k,) — bias for c_fc
        self._proj_w_reduced = None    # (k, hidden) — Conv1D format
        self._proj_b_reduced = None    # (hidden,) — bias stays full (not sliced)
    
    def prepare_reduced_weights(self, kept_indices: torch.Tensor):
        """Pre-slice weights for GPT2 Conv1D layers.
        
        GPT2 Conv1D weight shapes:
            c_fc.weight:   (hidden_size, intermediate_size) — select columns
            c_proj.weight: (intermediate_size, hidden_size) — select rows
        """
        # Ensure indices are on the same device as weights (critical for multi-GPU)
        weight_device = self.c_fc.weight.device
        kept_indices = kept_indices.to(weight_device)
        self._kept_indices = kept_indices
        
        # c_fc: (hidden, intermediate) → keep columns at kept_indices → (hidden, k)
        self._fc_w_reduced = self.c_fc.weight.data[:, kept_indices].contiguous()
        if self.c_fc.bias is not None:
            self._fc_b_reduced = self.c_fc.bias.data[kept_indices].contiguous()
        
        # c_proj: (intermediate, hidden) → keep rows at kept_indices → (k, hidden) 
        self._proj_w_reduced = self.c_proj.weight.data[kept_indices, :].contiguous()
        # c_proj bias is full hidden_size, keep as-is
        self._proj_b_reduced = self.c_proj.bias.data if self.c_proj.bias is not None else None
        
        self._reduced_ready = True
    
    def clear_reduced_weights(self):
        """Revert to full computation."""
        self._reduced_ready = False
        self._kept_indices = None
        self._fc_w_reduced = None
        self._fc_b_reduced = None
        self._proj_w_reduced = None
        self._proj_b_reduced = None
    
    def forward(self, x):
        if self._reduced_ready:
            # EFFICIENT PATH: reduced-size matmuls
            # GPT2 Conv1D: output = input @ weight + bias
            # c_fc reduced: x @ fc_w_reduced + fc_b_reduced → (B, S, k)
            hidden_states = x @ self._fc_w_reduced
            if self._fc_b_reduced is not None:
                hidden_states = hidden_states + self._fc_b_reduced
            
            # Activation
            hidden_states = self.act(hidden_states)  # (B, S, k)
            
            # c_proj reduced: hidden_states @ proj_w_reduced + proj_b → (B, S, hidden)
            hidden_states = hidden_states @ self._proj_w_reduced
            if self._proj_b_reduced is not None:
                hidden_states = hidden_states + self._proj_b_reduced
            
            if self.dropout is not None:
                hidden_states = self.dropout(hidden_states)
            
            return hidden_states
        else:
            # FULL PATH: standard GPT2 MLP computation
            hidden_states = self.c_fc(x)
            hidden_states = self.act(hidden_states)
            hidden_states = self.c_proj(hidden_states)
            if self.dropout is not None:
                hidden_states = self.dropout(hidden_states)
            return hidden_states


class EfficientGPTNeoXMLP(nn.Module):
    """Drop-in replacement for GPT-NeoX MLP with selective neuron computation.
    
    GPT-NeoX MLP structure:
        dense_h_to_4h: (hidden_size → intermediate_size) nn.Linear
        act:           activation function (GELU)
        dense_4h_to_h: (intermediate_size → hidden_size) nn.Linear
        output = dense_4h_to_h(act(dense_h_to_4h(x)))
    """
    
    def __init__(self, original_mlp):
        super().__init__()
        self.dense_h_to_4h = original_mlp.dense_h_to_4h
        self.dense_4h_to_h = original_mlp.dense_4h_to_h
        self.act = original_mlp.act
        
        # Reduced weight storage
        self._reduced_ready = False
        self._kept_indices = None
        self._fc_w_reduced = None    # (k, hidden) — rows of dense_h_to_4h
        self._fc_b_reduced = None    # (k,)
        self._proj_w_reduced = None  # (hidden, k) — columns of dense_4h_to_h
        self._proj_b_reduced = None  # (hidden,) — bias stays full
    
    def prepare_reduced_weights(self, kept_indices: torch.Tensor):
        """Pre-slice weights for GPT-NeoX Linear layers.
        
        nn.Linear weight shapes:
            dense_h_to_4h.weight: (intermediate_size, hidden_size) — select rows
            dense_4h_to_h.weight: (hidden_size, intermediate_size) — select columns
        """
        # Ensure indices are on the same device as weights (critical for multi-GPU)
        weight_device = self.dense_h_to_4h.weight.device
        kept_indices = kept_indices.to(weight_device)
        self._kept_indices = kept_indices
        
        # dense_h_to_4h: (intermediate, hidden) → keep rows → (k, hidden)
        self._fc_w_reduced = self.dense_h_to_4h.weight.data[kept_indices].contiguous()
        if self.dense_h_to_4h.bias is not None:
            self._fc_b_reduced = self.dense_h_to_4h.bias.data[kept_indices].contiguous()
        
        # dense_4h_to_h: (hidden, intermediate) → keep columns → (hidden, k)
        self._proj_w_reduced = self.dense_4h_to_h.weight.data[:, kept_indices].contiguous()
        self._proj_b_reduced = self.dense_4h_to_h.bias.data if self.dense_4h_to_h.bias is not None else None
        
        self._reduced_ready = True
    
    def clear_reduced_weights(self):
        """Revert to full computation."""
        self._reduced_ready = False
        self._kept_indices = None
        self._fc_w_reduced = None
        self._fc_b_reduced = None
        self._proj_w_reduced = None
        self._proj_b_reduced = None
    
    def forward(self, x):
        if self._reduced_ready:
            # EFFICIENT PATH
            # dense_h_to_4h reduced: F.linear(x, w_reduced, b_reduced) → (B, S, k)
            hidden_states = F.linear(x, self._fc_w_reduced, self._fc_b_reduced)
            hidden_states = self.act(hidden_states)  # (B, S, k)
            # dense_4h_to_h reduced: F.linear(hidden, proj_w_reduced, proj_b) → (B, S, hidden)
            hidden_states = F.linear(hidden_states, self._proj_w_reduced, self._proj_b_reduced)
            return hidden_states
        else:
            # FULL PATH
            hidden_states = self.dense_h_to_4h(x)
            hidden_states = self.act(hidden_states)
            hidden_states = self.dense_4h_to_h(hidden_states)
            return hidden_states


def replace_mlps(model, model_type: str):
    """Replace standard MLP modules with EfficientMLP modules in-place.
    
    This is called once after model loading. The EfficientMLP modules hold
    references to the original weight tensors (no copy), so memory usage
    doesn't increase.
    
    Args:
        model: The loaded HuggingFace model
        model_type: One of 'gpt2', 'llama', 'qwen3', 'mistral', 'gpt_neox'
    
    Returns:
        dict: Mapping of layer_name → EfficientMLP module reference
    """
    mlp_refs = {}
    
    if model_type == 'gpt2':
        for i, block in enumerate(model.transformer.h):
            efficient_mlp = EfficientGPT2MLP(block.mlp)
            block.mlp = efficient_mlp
            mlp_refs[f'layer_{i}'] = efficient_mlp
            
    elif model_type in ('llama', 'qwen3', 'mistral'):
        for i, layer in enumerate(model.model.layers):
            efficient_mlp = EfficientLlamaMLP(layer.mlp)
            layer.mlp = efficient_mlp
            mlp_refs[f'layer_{i}'] = efficient_mlp
            
    elif model_type == 'gpt_neox':
        gpt_neox_model = model.gpt_neox if hasattr(model, 'gpt_neox') else model
        for i, layer in enumerate(gpt_neox_model.layers):
            efficient_mlp = EfficientGPTNeoXMLP(layer.mlp)
            layer.mlp = efficient_mlp
            mlp_refs[f'layer_{i}'] = efficient_mlp
    else:
        raise ValueError(f"Unknown model type for MLP replacement: {model_type}")
    
    print(f"Replaced {len(mlp_refs)} MLP modules with EfficientMLP ({model_type})")
    return mlp_refs


def compile_mlps(model, model_type: str):
    """Apply torch.compile to each EfficientMLP for kernel fusion and CUDA graphs.
    
    Tier 1 optimization: During autoregressive generation, tensor shapes are fixed
    (batch=1, seq_len=1), making CUDA graph capture highly effective. This eliminates
    per-kernel launch overhead and fuses elementwise operations.
    
    Call this AFTER replace_mlps().
    
    Args:
        model: The model with EfficientMLP modules already installed
        model_type: One of 'gpt2', 'llama', 'qwen3', 'mistral', 'gpt_neox'
    """
    compiled_count = 0
    
    if model_type == 'gpt2':
        for block in model.transformer.h:
            if isinstance(block.mlp, EfficientGPT2MLP):
                block.mlp = torch.compile(block.mlp, mode="reduce-overhead", fullgraph=True)
                compiled_count += 1
                
    elif model_type in ('llama', 'qwen3', 'mistral'):
        for layer in model.model.layers:
            if isinstance(layer.mlp, EfficientLlamaMLP):
                layer.mlp = torch.compile(layer.mlp, mode="reduce-overhead", fullgraph=True)
                compiled_count += 1
                
    elif model_type == 'gpt_neox':
        gpt_neox_model = model.gpt_neox if hasattr(model, 'gpt_neox') else model
        for layer in gpt_neox_model.layers:
            if isinstance(layer.mlp, EfficientGPTNeoXMLP):
                layer.mlp = torch.compile(layer.mlp, mode="reduce-overhead", fullgraph=True)
                compiled_count += 1
    
    print(f"Applied torch.compile to {compiled_count} EfficientMLP modules (mode=reduce-overhead)")
