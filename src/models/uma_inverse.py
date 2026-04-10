import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .pairmixer_block import PairMixerBlock


class RBFEmbedding(nn.Module):
    def __init__(self, num_rbf: int = 32, max_distance: float = 24.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, max_distance, num_rbf)
        self.register_buffer("centers", centers)
        spacing = max_distance / max(1, num_rbf - 1)
        self.gamma = 1.0 / (spacing * spacing + 1e-8)

    def forward(self, distances: Tensor) -> Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * (diff * diff))


class PairMixerEncoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        pair_dim: int,
        pair_hidden_dim: int,
        transition_mult: int,
        dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            [
                PairMixerBlock(
                    pair_dim=pair_dim,
                    hidden_dim=pair_hidden_dim,
                    transition_mult=transition_mult,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, z: Tensor, pair_mask: Tensor) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                z = checkpoint(block, z, pair_mask, use_reentrant=False)
            else:
                z = block(z, pair_mask)
        return z


class UMAInverse(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        residue_input_dim = int(config.get("residue_input_dim", 6))
        ligand_input_dim = int(config.get("ligand_input_dim", 6))
        node_dim = int(config.get("node_dim", 128))
        pair_dim = int(config.get("pair_dim", 128))

        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.thermal_noise_std = float(config.get("thermal_noise_std", 0.0))

        self.residue_in = nn.Linear(residue_input_dim, node_dim)
        self.ligand_in = nn.Linear(ligand_input_dim, node_dim)
        self.node_norm = nn.LayerNorm(node_dim)

        self.pair_i = nn.Linear(node_dim, pair_dim, bias=False)
        self.pair_j = nn.Linear(node_dim, pair_dim, bias=False)
        self.rbf = RBFEmbedding(
            num_rbf=int(config.get("num_rbf", 32)),
            max_distance=float(config.get("max_distance", 24.0)),
        )
        self.rbf_proj = nn.Linear(int(config.get("num_rbf", 32)), pair_dim, bias=False)

        self.encoder = PairMixerEncoder(
            num_blocks=int(config.get("num_pairmixer_blocks", 6)),
            pair_dim=pair_dim,
            pair_hidden_dim=int(config.get("pair_hidden_dim", 128)),
            transition_mult=int(config.get("pair_transition_mult", 4)),
            dropout=float(config.get("dropout", 0.1)),
            gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )

        # Pair-to-node readout: feed encoder's geometric knowledge back to residue nodes.
        # After the encoder refines z, this projects the mean-pooled pair rows back into
        # node space so the decoder sees updated structural context (not just raw embeddings).
        self.node_readout = nn.Linear(pair_dim, node_dim)

        # Separate residue-protein and residue-ligand contexts for the decoder.
        # Avoids ligand signal being diluted (ligand atoms are ~8% of a typical mean pool).
        self.ctx_proj = nn.Linear(2 * pair_dim, pair_dim)

        self.ar_pair_to_scalar = nn.Linear(pair_dim, 1, bias=False)
        self.token_embedding = nn.Embedding(21, node_dim)

        self.decoder = nn.Sequential(
            nn.LayerNorm(node_dim + pair_dim),
            nn.Linear(node_dim + pair_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 21),
        )

    def _init_pair(self, node_repr: Tensor, coords: Tensor, pair_mask: Tensor) -> Tensor:
        node_i = self.pair_i(node_repr).unsqueeze(2)
        node_j = self.pair_j(node_repr).unsqueeze(1)

        dist = torch.cdist(coords.float(), coords.float())
        if self.training and self.thermal_noise_std > 0.0:
            noise = torch.randn_like(dist) * self.thermal_noise_std
            dist = torch.clamp(dist + noise, min=0.0)

        rbf = self.rbf_proj(self.rbf(dist).to(node_repr.dtype))
        z = node_i + node_j + rbf
        return z * pair_mask.unsqueeze(-1).to(dtype=z.dtype)

    def _ligand_aware_context(self, z: Tensor, pair_mask: Tensor, residue_count: int) -> Tensor:
        # Residue-protein context: how each residue relates to the rest of the chain.
        rr = z[:, :residue_count, :residue_count, :]
        rr_w = pair_mask[:, :residue_count, :residue_count].to(dtype=z.dtype)
        rr_ctx = (rr * rr_w.unsqueeze(-1)).sum(dim=2) / rr_w.sum(dim=2, keepdim=True).clamp_min(1.0)

        # Residue-ligand context: how each residue specifically relates to ligand atoms.
        # Kept separate so binding-site residues get a distinct, undiluted ligand signal.
        n_total = z.shape[2]
        if n_total > residue_count:
            rl = z[:, :residue_count, residue_count:, :]
            rl_w = pair_mask[:, :residue_count, residue_count:].to(dtype=z.dtype)
            rl_ctx = (rl * rl_w.unsqueeze(-1)).sum(dim=2) / rl_w.sum(dim=2, keepdim=True).clamp_min(1.0)
        else:
            rl_ctx = torch.zeros_like(rr_ctx)

        return self.ctx_proj(torch.cat([rr_ctx, rl_ctx], dim=-1))

    def _autoregressive_context(
        self,
        z: Tensor,
        sequence: Optional[Tensor],
        residue_mask: Tensor,
        decoding_order: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, residue_count = residue_mask.shape
        if sequence is None:
            return torch.zeros(
                (batch_size, residue_count, self.node_dim),
                dtype=z.dtype,
                device=z.device,
            )

        rr = z[:, :residue_count, :residue_count, :]
        token_emb = self.token_embedding(sequence.clamp(min=0, max=20))

        score = self.ar_pair_to_scalar(rr).squeeze(-1)
        score = score / math.sqrt(float(self.pair_dim))

        if decoding_order is None:
            decoding_order = torch.arange(residue_count, device=z.device).unsqueeze(0).expand(batch_size, -1)

        decoding_order_i = decoding_order.unsqueeze(2)  # [B, L, 1]
        decoding_order_j = decoding_order.unsqueeze(1)  # [B, 1, L]
        causal = decoding_order_i > decoding_order_j    # True if j was decoded before i

        valid = causal & residue_mask[:, :, None].bool() & residue_mask[:, None, :].bool()

        score = score.masked_fill(~valid, -1e4)
        probs = torch.softmax(score, dim=-1)
        probs = probs * valid.to(dtype=probs.dtype)
        # Avoid division by zero for the first decoded residue
        denominator = probs.sum(dim=-1, keepdim=True)
        probs = torch.where(denominator > 0, probs / denominator, torch.zeros_like(probs))

        return torch.einsum("bij,bjc->bic", probs, token_emb)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        residue_coords = batch["residue_coords"]
        residue_features = batch["residue_features"]
        residue_mask = batch["residue_mask"].bool()

        ligand_coords = batch["ligand_coords"]
        ligand_features = batch["ligand_features"]
        ligand_mask = batch["ligand_mask"].bool()

        residue_count = residue_coords.shape[1]

        residue_repr = self.residue_in(residue_features)
        ligand_repr = self.ligand_in(ligand_features)

        node_repr = torch.cat([residue_repr, ligand_repr], dim=1)
        node_repr = self.node_norm(node_repr)
        coords = torch.cat([residue_coords, ligand_coords], dim=1)
        node_mask = torch.cat([residue_mask, ligand_mask], dim=1)

        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        z = self._init_pair(node_repr=node_repr, coords=coords, pair_mask=pair_mask)
        z = self.encoder(z, pair_mask.to(dtype=z.dtype))

        # Pair-to-node readout: mean-pool encoder pair rows → residue node updates.
        # This lets 6 blocks of geometric reasoning inform the per-residue representation
        # rather than being compressed entirely into the ligand context below.
        z_pooled = z[:, :residue_count, :, :].mean(dim=2)      # [B, L, pair_dim]
        node_repr_res = node_repr[:, :residue_count, :] + self.node_readout(z_pooled)

        ligand_context = self._ligand_aware_context(z, pair_mask, residue_count=residue_count)
        ar_context = self._autoregressive_context(
            z=z,
            sequence=batch.get("sequence"),
            residue_mask=residue_mask,
            decoding_order=batch.get("decoding_order"),
        )

        decoder_input = torch.cat(
            [node_repr_res + ar_context, ligand_context],
            dim=-1,
        )
        logits = self.decoder(decoder_input)

        return {
            "logits": logits,
            "pair_repr": z,
        }
