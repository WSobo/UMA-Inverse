"""
src/models/pairmixer_block.py
─────────────────────────────
Implementation of the PairMixer Block as described in:
"Triangle Multiplication is All You Need for Biomolecular Structure Representations"

This module strips out the computationally expensive Sequence Updates and 
Triangle Attention layers found in normal AlphaFold3/Boltz Pairformer blocks.
It relies purely on Triangle Multiplication for spatial reasoning.
"""

import torch
from torch import nn, Tensor

from boltz.model.layers.transition import Transition

class CustomTriangleMultiplicationOutgoing(nn.Module):
    """
    Explicit cuBLAS Matmul implementation of Outgoing Triangle Multiplication.
    Computes: z_ij = sum_k (z_ik * z_jk) via batched torch.matmul instead of einsum.
    """
    def __init__(self, dim: int, c_hidden_mul: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(c_hidden_mul)
        self.linear_a = nn.Linear(dim, c_hidden_mul)
        self.linear_b = nn.Linear(dim, c_hidden_mul)
        self.gate_a = nn.Linear(dim, c_hidden_mul)
        self.gate_b = nn.Linear(dim, c_hidden_mul)
        self.linear_out = nn.Linear(c_hidden_mul, dim)
        self.gate_out = nn.Linear(dim, dim)

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        z_norm = self.norm(z)
        
        # 1. Project z to get left and right update tensors
        left_proj = self.linear_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))  # [B, L, L, C]
        right_proj = self.linear_b(z_norm) * torch.sigmoid(self.gate_b(z_norm)) # [B, L, L, C]
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            left_proj = left_proj * mask_expanded
            right_proj = right_proj * mask_expanded

        # 3. Explicit Matmul (Outgoing: bikc, bjkc -> bijc)
        left_permuted = left_proj.permute(0, 3, 1, 2)  # [B, C, L(i), L(k)]
        right_permuted = right_proj.permute(0, 3, 2, 1) # [B, C, L(k), L(j)]
        
        # 4. Direct cuBLAS Matmul
        z_out = left_permuted @ right_permuted # [B, C, L(i), L(j)]
        
        # 5. Permute back, make contiguous, and apply LayerNorm to stabilize summation variance
        z_out = z_out.permute(0, 2, 3, 1).contiguous() # [B, L, L, C]
        z_out = self.norm_out(z_out)
        
        z_update = self.linear_out(z_out) * torch.sigmoid(self.gate_out(z_norm))
        
        return z_update

class CustomTriangleMultiplicationIncoming(nn.Module):
    """
    Explicit cuBLAS Matmul implementation of Incoming Triangle Multiplication.
    Computes: z_ij = sum_k (z_ki * z_kj) via batched torch.matmul instead of einsum.
    """
    def __init__(self, dim: int, c_hidden_mul: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(c_hidden_mul)
        self.linear_a = nn.Linear(dim, c_hidden_mul)
        self.linear_b = nn.Linear(dim, c_hidden_mul)
        self.gate_a = nn.Linear(dim, c_hidden_mul)
        self.gate_b = nn.Linear(dim, c_hidden_mul)
        self.linear_out = nn.Linear(c_hidden_mul, dim)
        self.gate_out = nn.Linear(dim, dim)

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        z_norm = self.norm(z)
        
        # 1. Project z
        left_proj = self.linear_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))  # [B, L, L, C]
        right_proj = self.linear_b(z_norm) * torch.sigmoid(self.gate_b(z_norm)) # [B, L, L, C]
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            left_proj = left_proj * mask_expanded
            right_proj = right_proj * mask_expanded

        # 3. Explicit Matmul (Incoming: bkic, bkjc -> bijc)
        # We need sum over k. 
        # Left: bkic -> want shape [B, C, L(i), L(k)] => dim 2 is i, dim 1 is k
        left_permuted = left_proj.permute(0, 3, 2, 1)  
        # Right: bkjc -> want shape [B, C, L(k), L(j)] => dim 1 is k, dim 2 is j
        right_permuted = right_proj.permute(0, 3, 1, 2)
        
        # 4. Direct cuBLAS Matmul
        z_out = left_permuted @ right_permuted # [B, C, L(i), L(j)]
        
        # 5. Permute back, make contiguous, and apply LayerNorm to stabilize summation variance
        z_out = z_out.permute(0, 2, 3, 1).contiguous() # [B, L, L, C]
        z_out = self.norm_out(z_out)
        
        z_update = self.linear_out(z_out) * torch.sigmoid(self.gate_out(z_norm))
        
        return z_update


class PairMixerBlock(nn.Module):
    """
    A single layer of the PairMixer backbone.
    Updates only the pair representation (z) using explicit Matmul Triangle Multiplication.
    The sequence representation (s) passes through untouched.
    """
    
    def __init__(self, c_z: int, c_hidden_mul: int = 128, drop_rate: float = 0.0):
        """
        Args:
            c_z: Dimension of the pair representation.
            c_hidden_mul: Hidden dimension inside the triangle multiplication.
            drop_rate: Dropout probability.
        """
        super().__init__()
        
        # 1. Custom Triangle Multiplication (Incoming Edges)
        self.tri_mul_in = CustomTriangleMultiplicationIncoming(dim=c_z, c_hidden_mul=c_hidden_mul)
        
        # 2. Custom Triangle Multiplication (Outgoing Edges)
        self.tri_mul_out = CustomTriangleMultiplicationOutgoing(dim=c_z, c_hidden_mul=c_hidden_mul)
        
        # 3. Pair Transition (Feed-Forward Network applied across all pairs)
        self.transition = Transition(
            dim=c_z, 
            hidden=c_z * 4
        )

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass for a PairMixer block.
        
        Args:
            z: Pair representation tensor of shape [B, L, L, C_z]
            mask: Optional pair mask of shape [B, L, L] (1=valid, 0=padding)
            
        Returns:
            z_out: Updated pair representation of shape [B, L, L, C_z]
        """
        # Ensure mask is a tensor to satisfy the interface
        if mask is None:
            mask = torch.ones(z.shape[:3], device=z.device, dtype=z.dtype)

        # PEARL INSIGHT: Highly accelerated CUDA kernels for triangle multiplications
        # using explicit torch.matmul to bypass einsum VRAM spikes.

        # --- 1. Triangle Multiplication (Incoming) ---
        z = z + self.tri_mul_in(z, mask=mask)
        
        # --- 2. Triangle Multiplication (Outgoing) ---
        z = z + self.tri_mul_out(z, mask=mask)
        
        # --- 3. Pair Transition (FFN) ---
        z = z + self.transition(z)

        return z
