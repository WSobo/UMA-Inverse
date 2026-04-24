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

        # v2 phase 1: per-element embedding replaces the 6-bin one-hot linear
        # projection. Vocab = 120: slot 0 is reserved for padding (so padded
        # ligand positions in a batch contribute a zero vector), slots 1-118
        # cover the periodic table, and slot 119 is the "unknown element"
        # sentinel emitted by pdb_parser.py for exotic atoms.
        self.ligand_featurizer = str(config.get("ligand_featurizer", "onehot6"))
        if self.ligand_featurizer == "onehot6":
            self.ligand_in = nn.Linear(ligand_input_dim, node_dim)
        elif self.ligand_featurizer == "atomic_number_embedding":
            self.ligand_in = nn.Embedding(120, node_dim, padding_idx=0)
        else:
            raise ValueError(
                f"unknown ligand_featurizer={self.ligand_featurizer!r}; "
                "expected 'onehot6' or 'atomic_number_embedding'"
            )
        # v2 phase 2: purely informational — the model sees only residue_coords
        # and doesn't care which backbone atom produced them. Stored so the
        # checkpoint manifest and debug dumps can report the anchor used at
        # training time; the authoritative value still lives under data.*.
        self.residue_anchor = str(config.get("residue_anchor", "ca"))
        if self.residue_anchor not in ("ca", "cb"):
            raise ValueError(
                f"unknown residue_anchor={self.residue_anchor!r}; "
                "expected 'ca' or 'cb'"
            )
        self.node_norm = nn.LayerNorm(node_dim)

        # Relative position encoding for the residue-residue pair tensor.
        # Many residues share near-identical 6D dihedral features (e.g. all
        # helical positions), and the encoder alone cannot distinguish them.
        # Relpos gives each (i, j) pair a sequence-offset signature, which is
        # both necessary for the 1-batch overfit sanity check to converge and
        # standard in every modern structure-conditioned sequence model
        # (ProteinMPNN, AF2/3, Boltz, ESM-IF).
        self.relpos_max = int(config.get("relpos_max", 32))
        self.relpos_emb = nn.Embedding(2 * self.relpos_max + 2, pair_dim)
        # Last index (2*relpos_max + 1) reserved for "not a residue-residue pair"
        # (i.e. at least one of i,j is a ligand atom) — separate learnable bias.

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
        # Attention-weighted pool (not mean): each residue learns which j positions —
        # including ligand atoms — deserve weight in its per-residue summary. A uniform
        # mean averages away the position-specific signal the encoder just computed,
        # which was the primary bottleneck blocking the 1-batch overfit sanity check.
        self.pair_readout_attn = nn.Linear(pair_dim, 1, bias=False)
        self.node_readout = nn.Linear(pair_dim, node_dim)

        # Separate residue-protein and residue-ligand contexts for the decoder.
        # Avoids ligand signal being diluted (ligand atoms are ~8% of a typical mean pool).
        self.ctx_proj = nn.Linear(2 * pair_dim, pair_dim)

        # Multi-head AR attention: pair slot (i,j) → per-head attention logit
        # (previously a single scalar score — too narrow for teacher forcing to leak
        # useful identity information, causing the 1-batch overfit to plateau).
        self.ar_num_heads = int(config.get("ar_num_heads", 4))
        if node_dim % self.ar_num_heads != 0:
            raise ValueError(
                f"node_dim ({node_dim}) must be divisible by ar_num_heads "
                f"({self.ar_num_heads})."
            )
        self.ar_head_dim = node_dim // self.ar_num_heads
        self.ar_pair_to_attn = nn.Linear(pair_dim, self.ar_num_heads, bias=False)
        self.ar_value = nn.Linear(node_dim, node_dim, bias=False)
        self.ar_out = nn.Linear(node_dim, node_dim, bias=False)

        self.token_embedding = nn.Embedding(21, node_dim)

        # Decoder input: [node_repr_res | ar_context | ligand_context].
        # ar_context is a separate channel (not summed into node_repr_res) so the
        # decoder has a direct token-identity signal independent of the structural
        # representation — matches LigandMPNN's per-position node||prev-token concat.
        self.decoder = nn.Sequential(
            nn.LayerNorm(2 * node_dim + pair_dim),
            nn.Linear(2 * node_dim + pair_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, 21),
        )

    def _init_pair(
        self, node_repr: Tensor, coords: Tensor, pair_mask: Tensor, residue_count: int
    ) -> Tensor:
        node_i = self.pair_i(node_repr).unsqueeze(2)
        node_j = self.pair_j(node_repr).unsqueeze(1)

        dist = torch.cdist(coords.float(), coords.float())
        if self.training and self.thermal_noise_std > 0.0:
            noise = torch.randn_like(dist) * self.thermal_noise_std
            dist = torch.clamp(dist + noise, min=0.0)

        rbf = self.rbf_proj(self.rbf(dist).to(node_repr.dtype))

        # Relative position bias: clamp (i-j) to [-relpos_max, +relpos_max],
        # reserve the last bin for any pair involving a ligand atom.
        B, N, _ = node_repr.shape
        idx = torch.arange(N, device=node_repr.device)
        rel = idx.unsqueeze(1) - idx.unsqueeze(0)  # [N, N]
        rel = rel.clamp(-self.relpos_max, self.relpos_max) + self.relpos_max  # [0, 2*max]
        is_ligand_pair = (idx.unsqueeze(1) >= residue_count) | (idx.unsqueeze(0) >= residue_count)
        rel = torch.where(is_ligand_pair, torch.full_like(rel, 2 * self.relpos_max + 1), rel)
        relpos = self.relpos_emb(rel).unsqueeze(0).to(node_repr.dtype)  # [1, N, N, pair_dim]

        z = node_i + node_j + rbf + relpos
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

        H, D = self.ar_num_heads, self.ar_head_dim

        # Per-head attention logits from pair tensor: [B, L, L, H] → [B, H, L, L]
        logits = self.ar_pair_to_attn(rr) / math.sqrt(float(self.pair_dim))
        logits = logits.permute(0, 3, 1, 2)

        if decoding_order is None:
            decoding_order = torch.arange(
                residue_count, device=z.device
            ).unsqueeze(0).expand(batch_size, -1)

        decoding_order_i = decoding_order.unsqueeze(2)
        decoding_order_j = decoding_order.unsqueeze(1)
        causal = decoding_order_i > decoding_order_j

        valid = causal & residue_mask[:, :, None].bool() & residue_mask[:, None, :].bool()
        valid_h = valid.unsqueeze(1)  # [B, 1, L, L] → broadcasts over heads

        logits = logits.masked_fill(~valid_h, -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * valid_h.to(dtype=weights.dtype)
        # The first-decoded residue has no valid j — weight row sums to 0 and stays zero.
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(denominator > 0, weights / denominator, torch.zeros_like(weights))

        # Per-head values from token embeddings: [B, L, node_dim] → [B, H, L, D]
        values = self.ar_value(token_emb).view(batch_size, residue_count, H, D)
        values = values.permute(0, 2, 1, 3)

        # Weighted sum: [B, H, L, L] @ [B, H, L, D] → [B, H, L, D] → [B, L, node_dim]
        ctx = torch.matmul(weights, values).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(batch_size, residue_count, H * D)
        return self.ar_out(ctx)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        residue_coords = batch["residue_coords"]
        residue_features = batch["residue_features"]
        residue_mask = batch["residue_mask"].bool()

        ligand_coords = batch["ligand_coords"]
        ligand_mask = batch["ligand_mask"].bool()

        residue_count = residue_coords.shape[1]

        residue_repr = self.residue_in(residue_features)
        if self.ligand_featurizer == "onehot6":
            ligand_repr = self.ligand_in(batch["ligand_features"])
        else:  # "atomic_number_embedding"
            ligand_repr = self.ligand_in(batch["ligand_atomic_numbers"])

        node_repr = torch.cat([residue_repr, ligand_repr], dim=1)
        node_repr = self.node_norm(node_repr)
        coords = torch.cat([residue_coords, ligand_coords], dim=1)
        node_mask = torch.cat([residue_mask, ligand_mask], dim=1)

        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        z = self._init_pair(
            node_repr=node_repr,
            coords=coords,
            pair_mask=pair_mask,
            residue_count=residue_count,
        )
        z = self.encoder(z, pair_mask.to(dtype=z.dtype))

        # Attention-based pair-to-node readout: learned weights over the pair row so
        # each residue focuses on the j positions (residues or ligand atoms) that
        # matter for its own identity. Masked softmax keeps pad columns at zero mass.
        z_res = z[:, :residue_count, :, :]                                      # [B, L, N, pair_dim]
        readout_logits = self.pair_readout_attn(z_res).squeeze(-1)              # [B, L, N]
        readout_logits = readout_logits.masked_fill(
            ~node_mask.unsqueeze(1).bool(), -1e4
        )
        readout_w = torch.softmax(readout_logits, dim=-1)                       # [B, L, N]
        z_pooled = torch.einsum("bln,blnd->bld", readout_w, z_res)              # [B, L, pair_dim]
        node_repr_res = node_repr[:, :residue_count, :] + self.node_readout(z_pooled)

        ligand_context = self._ligand_aware_context(z, pair_mask, residue_count=residue_count)
        ar_context = self._autoregressive_context(
            z=z,
            sequence=batch.get("sequence"),
            residue_mask=residue_mask,
            decoding_order=batch.get("decoding_order"),
        )

        decoder_input = torch.cat(
            [node_repr_res, ar_context, ligand_context],
            dim=-1,
        )
        logits = self.decoder(decoder_input)

        return {
            "logits": logits,
            "pair_repr": z,
        }
