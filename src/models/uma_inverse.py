import math

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


# LigandMPNN periodic-table features (model_utils.py:753-1121, extracted
# verbatim and extended by one slot so UMA-Inverse's atomic_number=119
# "unknown" sentinel routes to the same group=0/period=0 missing bucket as
# LigandMPNN's index-0 missing slot).
_PERIODIC_GROUP: tuple[int, ...] = (
    0, 1, 18, 1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    0,  # 119 → "unknown" → same bucket as the index-0 padding slot.
)
_PERIODIC_PERIOD: tuple[int, ...] = (
    0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    0,  # 119 sentinel
)
assert len(_PERIODIC_GROUP) == 120 and len(_PERIODIC_PERIOD) == 120


class LigandMPNNAtomicFeaturizer(nn.Module):
    """v3 ligand atom-type embedding — direct port of LigandMPNN's
    type_linear pipeline (model_utils.py:1284-1295).

    Given atomic numbers [B, M], emit:
        [B, M, 120] one-hot of atomic number
      ⊕ [B, M, 19]  one-hot of periodic-table group  (1..18 + 0 missing)
      ⊕ [B, M, 8]   one-hot of period                (1..7 + 0 missing)
      = [B, M, 147]  → linear → [B, M, node_dim]

    Reserves index 0 for both padding and unknown (matching UMA's existing
    convention via the `_PERIODIC_*` sentinels above).
    """

    def __init__(self, node_dim: int) -> None:
        super().__init__()
        group  = torch.tensor(_PERIODIC_GROUP,  dtype=torch.long)
        period = torch.tensor(_PERIODIC_PERIOD, dtype=torch.long)
        self.register_buffer("group_table", group, persistent=False)
        self.register_buffer("period_table", period, persistent=False)
        self.proj = nn.Linear(120 + 19 + 8, node_dim, bias=False)
        self.node_dim = node_dim

    def forward(self, atomic_numbers: Tensor) -> Tensor:
        # atomic_numbers: [B, M] long, values in [0, 119].
        z = atomic_numbers.clamp(min=0, max=119).long()
        atom_oh   = nn.functional.one_hot(z, num_classes=120)            # [B, M, 120]
        group_oh  = nn.functional.one_hot(self.group_table[z],  num_classes=19)  # [B, M, 19]
        period_oh = nn.functional.one_hot(self.period_table[z], num_classes=8)   # [B, M, 8]
        feat = torch.cat([atom_oh, group_oh, period_oh], dim=-1).to(self.proj.weight.dtype)
        return self.proj(feat)


class LigandLocalEnvEmbedding(nn.Module):
    """v3 phase 3 — per-ligand-atom local-environment feature.

    For each ligand atom, encodes the distances to its K nearest neighbour
    ligand atoms (within the same ligand context). Gives the network a
    'second-shell' signal that the dense pair attention can compose with the
    raw [M, M] cdist block — the cheapest feature-level analog of
    LigandMPNN's intra-ligand message passing without changing the
    architecture.
    """

    def __init__(
        self,
        num_neighbors: int,
        num_rbf: int,
        max_distance: float,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.k = int(num_neighbors)
        self.max_distance = float(max_distance)
        self.rbf = RBFEmbedding(num_rbf=num_rbf, max_distance=max_distance)
        self.proj = nn.Linear(self.k * num_rbf, out_dim, bias=False)
        self.out_dim = out_dim

    def forward(self, ligand_coords: Tensor, ligand_mask: Tensor) -> Tensor:
        # ligand_coords: [B, M, 3], ligand_mask: [B, M] (bool)
        B, M, _ = ligand_coords.shape
        if M == 0:
            return torch.zeros(
                (B, 0, self.out_dim),
                dtype=ligand_coords.dtype,
                device=ligand_coords.device,
            )
        dists = torch.cdist(ligand_coords.float(), ligand_coords.float())  # [B, M, M]
        eye = torch.eye(M, dtype=torch.bool, device=ligand_coords.device).unsqueeze(0)
        valid_pair = ligand_mask.unsqueeze(2) & ligand_mask.unsqueeze(1)  # [B, M, M]
        invalid = eye | ~valid_pair
        dists = dists.masked_fill(invalid, float("inf"))
        # Use min(M-1, k) actual neighbours; pad short rows to k with max_distance.
        k_eff = min(self.k, max(M - 1, 0))
        if k_eff == 0:
            d_padded = torch.full(
                (B, M, self.k), self.max_distance,
                dtype=ligand_coords.dtype, device=ligand_coords.device,
            )
        else:
            topk = torch.topk(dists, k=k_eff, dim=-1, largest=False).values  # [B, M, k_eff]
            topk = torch.where(
                torch.isfinite(topk),
                topk,
                torch.full_like(topk, self.max_distance),
            )
            if k_eff < self.k:
                pad = torch.full(
                    (B, M, self.k - k_eff), self.max_distance,
                    dtype=topk.dtype, device=topk.device,
                )
                topk = torch.cat([topk, pad], dim=-1)
            d_padded = topk.to(ligand_coords.dtype)
        rbf_feats = self.rbf(d_padded.float()).to(ligand_coords.dtype)  # [B, M, k, num_rbf]
        rbf_flat = rbf_feats.reshape(B, M, -1)
        out = self.proj(rbf_flat)
        return out * ligand_mask.unsqueeze(-1).to(out.dtype)


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
    def __init__(self, config: dict) -> None:
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
        elif self.ligand_featurizer == "ligandmpnn_atomic":
            # v3 — LigandMPNN-faithful atomic + group + period featurizer.
            self.ligand_in = LigandMPNNAtomicFeaturizer(node_dim)
        else:
            raise ValueError(
                f"unknown ligand_featurizer={self.ligand_featurizer!r}; "
                "expected 'onehot6', 'atomic_number_embedding', or "
                "'ligandmpnn_atomic'"
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
        num_rbf = int(config.get("num_rbf", 32))
        self.rbf = RBFEmbedding(
            num_rbf=num_rbf,
            max_distance=float(config.get("max_distance", 24.0)),
        )
        self.rbf_proj = nn.Linear(num_rbf, pair_dim, bias=False)

        # v2 phase 3: when backbone_full is selected, the residue-residue
        # [L,L] block of the pair tensor is driven by 5 backbone-atom-pair
        # distances (Cα-Cα, Cα-N, Cα-C, N-O, O-C) rather than a single
        # anchor-to-anchor cdist. The [L,M]/[M,L]/[M,M] blocks still use the
        # single-atom distance — ligand atoms have no backbone analogue, so
        # phase 2's residue anchor still governs the residue-ligand side.
        # pair_distance_atoms — controls the [L, L] residue-residue block.
        #
        # 'anchor_only'      : v1, single anchor-anchor cdist.
        # 'backbone_full'    : v2, 5 backbone-pair distances
        #                       (Cα-Cα, Cα-N, Cα-C, N-O, O-C).
        # 'backbone_full_25' : v3, the full LigandMPNN backbone-pair set
        #                       — 5 self-pairs (Cα-Cα, N-N, C-C, O-O, Cβ-Cβ)
        #                       + all 20 ordered cross-pairs across {N,Cα,C,O,Cβ}.
        #                       Mirrors model_utils.py:1208-1232.
        self.pair_distance_atoms = str(config.get("pair_distance_atoms", "anchor_only"))
        if self.pair_distance_atoms not in (
            "anchor_only", "backbone_full", "backbone_full_25",
        ):
            raise ValueError(
                f"unknown pair_distance_atoms={self.pair_distance_atoms!r}; "
                "expected 'anchor_only', 'backbone_full', or 'backbone_full_25'"
            )

        # v3 phase 1 — extend backbone_full to the [L, M] / [M, L] residue-
        # ligand blocks. Default 'anchor_only' = v2 behaviour (single Cβ-anchor
        # × ligand distance). 'backbone_full' = 5 protein atoms
        # (N, Cα, C, O, Cβ) × ligand atoms — matches LigandMPNN
        # D_N_Y / D_Ca_Y / D_C_Y / D_O_Y / D_Cb_Y (model_utils.py:1297-1307).
        self.pair_distance_atoms_ligand = str(
            config.get("pair_distance_atoms_ligand", "anchor_only")
        )
        if self.pair_distance_atoms_ligand not in ("anchor_only", "backbone_full"):
            raise ValueError(
                f"unknown pair_distance_atoms_ligand={self.pair_distance_atoms_ligand!r}; "
                "expected 'anchor_only' or 'backbone_full'"
            )

        # Multi-atom RBF projections. Two separate Linear layers because the
        # 5-pair (v2 / v3 L-M) and 25-pair (v3 L-L) input dims differ. They
        # don't share weights — the 25-pair block has structurally richer
        # signal that the 5-pair projection's row span can't represent
        # losslessly anyway.
        needs_rbf_multi_5 = (
            self.pair_distance_atoms == "backbone_full"
            or self.pair_distance_atoms_ligand == "backbone_full"
        )
        if needs_rbf_multi_5:
            self.rbf_proj_multi = nn.Linear(5 * num_rbf, pair_dim, bias=False)
        if self.pair_distance_atoms == "backbone_full_25":
            self.rbf_proj_multi_25 = nn.Linear(25 * num_rbf, pair_dim, bias=False)

        # v3 phase 2 — frame-relative angle features (sin/cos of ligand-atom
        # bearing in each residue's local N-Cα-C frame). [B, L, M, 4] feature
        # block (cosθ, sinθ, cosφ, sinφ) projected into pair_dim and added to
        # the [L, M] / [M, L] block.
        self.frame_relative_angles = bool(config.get("frame_relative_angles", False))
        if self.frame_relative_angles:
            self.frame_angle_proj = nn.Linear(4, pair_dim, bias=False)

        # v3 phase 3 — intra-ligand multi-distance node-level enrichment.
        self.intra_ligand_multidist = bool(config.get("intra_ligand_multidist", False))
        if self.intra_ligand_multidist:
            self.ligand_local_env = LigandLocalEnvEmbedding(
                num_neighbors=int(config.get("intra_ligand_num_neighbors", 3)),
                num_rbf=num_rbf,
                max_distance=float(config.get("intra_ligand_max_distance", 8.0)),
                out_dim=node_dim,
            )

        # v3 phase 4 — Gaussian σ (Å) noise added to raw residue + ligand
        # coords at the top of forward(), training-only. Independent of the
        # legacy thermal_noise_std, which jitters post-cdist distances and is
        # kept untouched for the v3 ≡ v2 fallback regression test.
        self.coord_noise_std = float(config.get("coord_noise_std", 0.0))

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
        self,
        node_repr: Tensor,
        coords: Tensor,
        pair_mask: Tensor,
        residue_count: int,
        residue_backbone_coords: Tensor | None = None,
        residue_ligand_frame_angles: Tensor | None = None,
    ) -> Tensor:
        node_i = self.pair_i(node_repr).unsqueeze(2)
        node_j = self.pair_j(node_repr).unsqueeze(1)

        dist = torch.cdist(coords.float(), coords.float())
        if self.training and self.thermal_noise_std > 0.0:
            noise = torch.randn_like(dist) * self.thermal_noise_std
            dist = torch.clamp(dist + noise, min=0.0)

        rbf = self.rbf_proj(self.rbf(dist).to(node_repr.dtype))
        rbf_cloned = False

        needs_backbone = (
            self.pair_distance_atoms in ("backbone_full", "backbone_full_25")
            or self.pair_distance_atoms_ligand == "backbone_full"
        )
        if needs_backbone and residue_backbone_coords is None:
            raise ValueError(
                "pair_distance_atoms*='backbone_full*' requires "
                "residue_backbone_coords to be passed into _init_pair "
                "(set data.return_backbone_coords=True upstream)."
            )
        if needs_backbone:
            bb = residue_backbone_coords.float()
            n_at  = bb[:, :, 0, :]
            ca_at = bb[:, :, 1, :]
            c_at  = bb[:, :, 2, :]
            o_at  = bb[:, :, 3, :]
            # Always derive Cβ inline from N/Cα/C — matches LigandMPNN
            # (model_utils.py:1200-1203) and stays consistent regardless of
            # data.residue_anchor (also keeps Cβ derived from any noisy X
            # so coord-noise propagates correctly into Cβ-* distances).
            b_vec = ca_at - n_at
            c_vec = c_at - ca_at
            a_vec = torch.linalg.cross(b_vec, c_vec, dim=-1)
            cb_at = -0.58273431 * a_vec + 0.56802827 * b_vec - 0.54067466 * c_vec + ca_at

        def _maybe_thermal_noise(dists: list[Tensor]) -> list[Tensor]:
            if self.training and self.thermal_noise_std > 0.0:
                return [
                    torch.clamp(d + torch.randn_like(d) * self.thermal_noise_std, min=0.0)
                    for d in dists
                ]
            return dists

        # v2 phase 3: replace the [L, L] residue-residue block of the single-
        # atom RBF with a multi-atom RBF computed from 5 backbone pair
        # distances (Cα-Cα, Cα-N, Cα-C, N-O, O-C). v3 'backbone_full_25'
        # below uses the full LigandMPNN 25-pair set instead.
        if self.pair_distance_atoms == "backbone_full":
            ll_pair_dists = _maybe_thermal_noise([
                torch.cdist(ca_at, ca_at),  # Cα-Cα
                torch.cdist(ca_at, n_at),   # Cα-N
                torch.cdist(ca_at, c_at),   # Cα-C
                torch.cdist(n_at,  o_at),   # N-O
                torch.cdist(o_at,  c_at),   # O-C
            ])
            ll_rbf_stack = torch.cat(
                [self.rbf(d).to(node_repr.dtype) for d in ll_pair_dists], dim=-1,
            )  # [B, L, L, 5*num_rbf]
            ll_multi_rbf = self.rbf_proj_multi(ll_rbf_stack)  # [B, L, L, pair_dim]
            # Replace the [L, L] slice. Clone first so autograd doesn't see an
            # in-place modification of the earlier linear-projection output.
            if not rbf_cloned:
                rbf = rbf.clone()
                rbf_cloned = True
            rbf[:, :residue_count, :residue_count, :] = ll_multi_rbf

        # v3 — full LigandMPNN 25-pair backbone set for the [L, L] block.
        # Mirrors model_utils.py:1208-1232 verbatim: 5 self-pairs +
        # 20 ordered cross-pairs across {N, Cα, C, O, Cβ}. This is the
        # feature-matched analog of LigandMPNN's protein edge featurizer
        # (KNN-graph topology is *not* matched — that's the architectural
        # variable v3 is testing).
        elif self.pair_distance_atoms == "backbone_full_25":
            atoms = (n_at, ca_at, c_at, o_at, cb_at)  # indices 0..4
            ll_pair_dists = _maybe_thermal_noise([
                # 5 self-pairs (LigandMPNN order: Ca-Ca, N-N, C-C, O-O, Cb-Cb)
                torch.cdist(ca_at, ca_at),
                torch.cdist(n_at,  n_at),
                torch.cdist(c_at,  c_at),
                torch.cdist(o_at,  o_at),
                torch.cdist(cb_at, cb_at),
                # Forward direction: from {Ca, N, Cb, O} into the rest
                torch.cdist(ca_at, n_at),    # Ca-N
                torch.cdist(ca_at, c_at),    # Ca-C
                torch.cdist(ca_at, o_at),    # Ca-O
                torch.cdist(ca_at, cb_at),   # Ca-Cb
                torch.cdist(n_at,  c_at),    # N-C
                torch.cdist(n_at,  o_at),    # N-O
                torch.cdist(n_at,  cb_at),   # N-Cb
                torch.cdist(cb_at, c_at),    # Cb-C
                torch.cdist(cb_at, o_at),    # Cb-O
                torch.cdist(o_at,  c_at),    # O-C
                # Reverse direction (LigandMPNN encodes both since edges are
                # asymmetric in their KNN graph; for dense attention we keep
                # both for parity of feature content)
                torch.cdist(n_at,  ca_at),   # N-Ca
                torch.cdist(c_at,  ca_at),   # C-Ca
                torch.cdist(o_at,  ca_at),   # O-Ca
                torch.cdist(cb_at, ca_at),   # Cb-Ca
                torch.cdist(c_at,  n_at),    # C-N
                torch.cdist(o_at,  n_at),    # O-N
                torch.cdist(cb_at, n_at),    # Cb-N
                torch.cdist(c_at,  cb_at),   # C-Cb
                torch.cdist(o_at,  cb_at),   # O-Cb
                torch.cdist(c_at,  o_at),    # C-O
            ])
            ll_rbf_stack = torch.cat(
                [self.rbf(d).to(node_repr.dtype) for d in ll_pair_dists], dim=-1,
            )  # [B, L, L, 25*num_rbf]
            ll_multi_rbf = self.rbf_proj_multi_25(ll_rbf_stack)
            if not rbf_cloned:
                rbf = rbf.clone()
                rbf_cloned = True
            rbf[:, :residue_count, :residue_count, :] = ll_multi_rbf
            del atoms

        # v3 phase 1: replace the [L, M] / [M, L] residue-ligand blocks with
        # a 5-atom × ligand multi-atom RBF (N, Cα, C, O, Cβ × ligand atoms).
        # Re-uses rbf_proj_multi so the same learned mapping covers both [L, L]
        # and [L, M] multi-atom signals.
        ligand_count = coords.shape[1] - residue_count
        if self.pair_distance_atoms_ligand == "backbone_full" and ligand_count > 0:
            lig_xyz = coords[:, residue_count:, :].float()  # [B, M, 3]
            lm_pair_dists = _maybe_thermal_noise([
                torch.cdist(n_at,  lig_xyz),   # D_N_Y
                torch.cdist(ca_at, lig_xyz),   # D_Ca_Y
                torch.cdist(c_at,  lig_xyz),   # D_C_Y
                torch.cdist(o_at,  lig_xyz),   # D_O_Y
                torch.cdist(cb_at, lig_xyz),   # D_Cb_Y
            ])
            lm_rbf_stack = torch.cat(
                [self.rbf(d).to(node_repr.dtype) for d in lm_pair_dists], dim=-1,
            )  # [B, L, M, 5*num_rbf]
            lm_multi_rbf = self.rbf_proj_multi(lm_rbf_stack)  # [B, L, M, pair_dim]
            if not rbf_cloned:
                rbf = rbf.clone()
                rbf_cloned = True
            rbf[:, :residue_count, residue_count:, :] = lm_multi_rbf
            rbf[:, residue_count:, :residue_count, :] = lm_multi_rbf.transpose(1, 2)

        # v3 phase 2: frame-relative angle features added (not replaced) into
        # the [L, M] / [M, L] block. Encodes the ligand atom's bearing in each
        # residue's local N-Cα-C frame as sin/cos of two angles.
        if (
            self.frame_relative_angles
            and residue_ligand_frame_angles is not None
            and ligand_count > 0
        ):
            # [B, L, M, 4] → [B, L, M, pair_dim]
            frame_proj = self.frame_angle_proj(
                residue_ligand_frame_angles.to(node_repr.dtype)
            )
            if not rbf_cloned:
                rbf = rbf.clone()
                rbf_cloned = True
            rbf[:, :residue_count, residue_count:, :] = (
                rbf[:, :residue_count, residue_count:, :] + frame_proj
            )
            rbf[:, residue_count:, :residue_count, :] = (
                rbf[:, residue_count:, :residue_count, :] + frame_proj.transpose(1, 2)
            )

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
        sequence: Tensor | None,
        residue_mask: Tensor,
        decoding_order: Tensor | None = None,
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

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        residue_coords = batch["residue_coords"]
        residue_features = batch["residue_features"]
        residue_mask = batch["residue_mask"].bool()

        ligand_coords = batch["ligand_coords"]
        ligand_mask = batch["ligand_mask"].bool()
        bb_in = batch.get("residue_backbone_coords")

        # v3 phase 4 — raw-coord noise (training only). Mirrors LigandMPNN
        # (model_utils.py:1189-1191): noise is added once to the backbone
        # tensor X and the ligand tensor Y, and the residue anchor (Cβ when
        # data.residue_anchor='cb') is recomputed from the *noisy* X. This
        # keeps all derived distances (Cα-Y, Cβ-Y, frame angles, etc.)
        # geometrically consistent — independent noise on each tensor would
        # have broken the Cβ = f(N, Cα, C) relationship LigandMPNN preserves.
        if self.training and self.coord_noise_std > 0.0:
            ligand_coords = ligand_coords + torch.randn_like(ligand_coords) * self.coord_noise_std
            if bb_in is not None:
                bb_noisy = bb_in + torch.randn_like(bb_in) * self.coord_noise_std
                if self.residue_anchor == "cb":
                    # Same formula as ligandmpnn_bridge._construct_virtual_cb,
                    # inlined and vectorised over [B, L] so the noisy-Cβ
                    # derivation stays in fp32 in-graph (no host roundtrip).
                    n  = bb_noisy[..., 0, :]
                    ca = bb_noisy[..., 1, :]
                    c  = bb_noisy[..., 2, :]
                    b = ca - n
                    c_vec = c - ca
                    a = torch.linalg.cross(b, c_vec, dim=-1)
                    residue_coords = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_vec + ca
                else:  # 'ca'
                    residue_coords = bb_noisy[..., 1, :]
                residue_backbone_noisy = bb_noisy
            else:
                # No backbone tensor available — fall back to noising the
                # anchor independently. Only relevant for code paths that
                # don't request return_backbone_coords (i.e. v2 anchor-only).
                residue_coords = residue_coords + torch.randn_like(residue_coords) * self.coord_noise_std
                residue_backbone_noisy = None
        else:
            residue_backbone_noisy = bb_in

        residue_count = residue_coords.shape[1]

        residue_repr = self.residue_in(residue_features)
        if self.ligand_featurizer == "onehot6":
            ligand_repr = self.ligand_in(batch["ligand_features"])
        else:  # "atomic_number_embedding" or "ligandmpnn_atomic"
            ligand_repr = self.ligand_in(batch["ligand_atomic_numbers"])

        # v3 phase 3 — additive node-level enrichment of ligand embeddings
        # with K-NN intra-ligand distance signature.
        if self.intra_ligand_multidist:
            ligand_repr = ligand_repr + self.ligand_local_env(ligand_coords, ligand_mask)

        node_repr = torch.cat([residue_repr, ligand_repr], dim=1)
        node_repr = self.node_norm(node_repr)
        coords = torch.cat([residue_coords, ligand_coords], dim=1)
        node_mask = torch.cat([residue_mask, ligand_mask], dim=1)

        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        needs_backbone = (
            self.pair_distance_atoms in ("backbone_full", "backbone_full_25")
            or self.pair_distance_atoms_ligand == "backbone_full"
        )
        residue_backbone_coords = residue_backbone_noisy if needs_backbone else None
        residue_ligand_frame_angles = (
            batch.get("residue_ligand_frame_angles") if self.frame_relative_angles else None
        )
        z = self._init_pair(
            node_repr=node_repr,
            coords=coords,
            pair_mask=pair_mask,
            residue_count=residue_count,
            residue_backbone_coords=residue_backbone_coords,
            residue_ligand_frame_angles=residue_ligand_frame_angles,
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
