"""
src/models/pairmixer_block.py
─────────────────────────────
PairMixer-style pair block — a from-scratch reimplementation of the PairMixer
layer for the UMA-Inverse encoder.

Defining recipe vs. the standard Pairformer (AlphaFold3 / Boltz-1):
  - Drops Triangle Attention  (O(L³) memory) → up to 4x faster inference
  - Drops the Sequence / MSA track (pure pair-space reasoning)
  - Keeps Triangle Multiplication (outgoing + incoming) as the only geometric mix
  - Reported net result in the paper: ~34% lower training cost at matched accuracy

Reference:
  Ouyang-Zhang, Murugan, Diaz, Scarpellini, Bowen, Gruver, Klivans,
  Krähenbühl, Faust, Al-Shedivat. "Triangle Multiplication is All You Need
  for Biomolecular Structure Representations." arXiv:2510.18870 (2025).
  Genesis Research / UT Austin. Official code:
  https://github.com/genesistherapeutics/pairmixer

This is NOT a bit-exact port of the official repo. Faithful pieces: the
pair-only block structure, both triangle-multiplication directions, the
AF3/Boltz input-gate / output-gate-from-normed-input formulation, and pair
masking. Deliberate divergences in this reimplementation:
  - Triangle contraction uses an explicit (rearrange + matmul) instead of the
    reference einsum — mathematically identical, no fp32 up-cast here.
  - PairTransition is a plain GELU MLP, not the reference SwiGLU transition.
  - No AF2/AF3 row/column structured dropout (dropout lives in the transition).
  - Block order is incoming → outgoing (reference uses outgoing → incoming).
  - PyTorch-default init and bias-carrying projections.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class PairTransition(nn.Module):
    """Pre-norm 2-layer MLP applied independently to every pair (i, j).

    Replaces the boltz/AF3 Transition import with a self-contained
    implementation so the module has zero external dependencies.
    """

    def __init__(self, dim: int, transition_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dim * transition_mult
        self.norm = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, hidden)
        self.ff2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, z: Tensor) -> Tensor:
        z = self.norm(z)
        return self.drop(self.ff2(F.gelu(self.ff1(z))))


class TriangleMultiplicationOutgoing(nn.Module):
    """Outgoing Triangle Multiplication.

    z_ij ← Σ_k  sigmoid(gate_ik) ⊙ proj_ik  ·  sigmoid(gate_jk) ⊙ proj_jk

    Implemented with an explicit torch.matmul instead of the reference
    torch.einsum; the two are mathematically identical.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.linear_a = nn.Linear(dim, hidden_dim)
        self.linear_b = nn.Linear(dim, hidden_dim)
        self.gate_a = nn.Linear(dim, hidden_dim)
        self.gate_b = nn.Linear(dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, dim)
        self.gate_out = nn.Linear(dim, dim)

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            z:    [B, N, N, dim]
            mask: [B, N, N] float or bool (1 = valid, 0 = padding)
        Returns:
            z_update: [B, N, N, dim]
        """
        z_norm = self.norm(z)

        left = self.linear_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))   # [B, N, N, H]
        right = self.linear_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))  # [B, N, N, H]

        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=z.dtype)
            left = left * m
            right = right * m

        # Outgoing: z_ij = Σ_k left_ik · right_jk
        left_p  = rearrange(left,  'b i k h -> b h i k')
        right_p = rearrange(right, 'b j k h -> b h k j')

        z_out = left_p @ right_p                              # [B, H, i, j]
        z_out = rearrange(z_out, 'b h i j -> b i j h')
        z_out = self.norm_out(z_out)

        return self.linear_out(z_out) * torch.sigmoid(self.gate_out(z_norm))


class TriangleMultiplicationIncoming(nn.Module):
    """Incoming Triangle Multiplication.

    z_ij ← Σ_k  sigmoid(gate_ki) ⊙ proj_ki  ·  sigmoid(gate_kj) ⊙ proj_kj
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.linear_a = nn.Linear(dim, hidden_dim)
        self.linear_b = nn.Linear(dim, hidden_dim)
        self.gate_a = nn.Linear(dim, hidden_dim)
        self.gate_b = nn.Linear(dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, dim)
        self.gate_out = nn.Linear(dim, dim)

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            z:    [B, N, N, dim]
            mask: [B, N, N] float or bool (1 = valid, 0 = padding)
        Returns:
            z_update: [B, N, N, dim]
        """
        z_norm = self.norm(z)

        left = self.linear_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))   # [B, N, N, H]
        right = self.linear_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))  # [B, N, N, H]

        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=z.dtype)
            left = left * m
            right = right * m

        # Incoming: z_ij = Σ_k left_ki · right_kj
        left_p  = rearrange(left,  'b k i h -> b h i k')
        right_p = rearrange(right, 'b k j h -> b h k j')

        z_out = left_p @ right_p                              # [B, H, i, j]
        z_out = rearrange(z_out, 'b h i j -> b i j h')
        z_out = self.norm_out(z_out)

        return self.linear_out(z_out) * torch.sigmoid(self.gate_out(z_norm))


# ──────────────────────────────────────────────────────────────────────────────
# Main block
# ──────────────────────────────────────────────────────────────────────────────

class PairMixerBlock(nn.Module):
    """Single PairMixer layer.

    Execution order (residual throughout):
        Z = Z + TriMulIn(Z)
        Z = Z + TriMulOut(Z)
        Z = Z + Transition(Z)

    No sequence updates, no triangle attention — pure pair-space reasoning.
    This is the core of the PairMixer efficiency gain.

    Note: the reference applies outgoing before incoming; this block applies
    incoming before outgoing (see module docstring).
    """

    def __init__(
        self,
        pair_dim: int,
        hidden_dim: int,
        transition_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
            pair_dim:        Channel width of the pair tensor Z.
            hidden_dim:      Hidden width inside each triangle multiplication
                             (controls the low-rank bottleneck).
            transition_mult: FFN expansion factor for PairTransition.
            dropout:         Dropout rate applied inside PairTransition.
        """
        super().__init__()
        self.tri_mul_in = TriangleMultiplicationIncoming(dim=pair_dim, hidden_dim=hidden_dim)
        self.tri_mul_out = TriangleMultiplicationOutgoing(dim=pair_dim, hidden_dim=hidden_dim)
        self.transition = PairTransition(dim=pair_dim, transition_mult=transition_mult, dropout=dropout)

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Args:
            z:    [B, N, N, pair_dim]  — dense pair representation
            mask: [B, N, N]            — 1 for valid pairs, 0 for padding
        Returns:
            z:    [B, N, N, pair_dim]
        """
        z = z + self.tri_mul_in(z, mask=mask)
        z = z + self.tri_mul_out(z, mask=mask)
        z = z + self.transition(z)
        return z
