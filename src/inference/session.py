"""Inference session: owns the model checkpoint and encoded structure state.

A single :class:`InferenceSession` loads a checkpoint once and can then
process many PDB structures. For each structure :meth:`load_structure`
runs the encoder a single time and returns a :class:`StructureContext`
that holds every tensor the downstream decoder needs — so sampling N
sequences from the same PDB does *not* recompute the pair tensor.

This separation is the main reason the package is library-code + thin
CLI rather than a script: the encoder-once, decode-many pattern is
essential for performance and can only be expressed cleanly when the
structure context is a first-class object.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from src.data.ligandmpnn_bridge import load_example_from_pdb
from src.models.uma_inverse import UMAInverse

logger = logging.getLogger(__name__)

PathLike = str | Path


# ─── Data class for encoded structure ─────────────────────────────────────────


@dataclass
class StructureContext:
    """Encoder outputs for a single PDB, ready to feed the autoregressive decoder.

    Populated by :meth:`InferenceSession.load_structure`. Treat all tensors
    as immutable — the decoding layer builds samples by mutating its own
    ``sequence`` tensor, never this object.

    Attributes:
        residue_ids: Per-residue identifiers (``"A23"``, ``"B42C"``), aligned
            with ``native_sequence`` and the first dimension of ``node_repr_res``.
        residue_id_to_index: Inverse lookup for O(1) residue resolution.
        chain_ids: Per-residue chain letter (duplicated from residue_ids for
            fast chain-wide constraint application).
        native_sequence: ``[L]`` int64 token indices parsed from the input PDB.
        design_mask: ``[L]`` bool — True for residues the parser flagged as
            designable (all True by default; LigandMPNN's ``chain_mask``).
        z: ``[1, N, N, pair_dim]`` encoded pair tensor (N = L + M).
        node_repr_res: ``[1, L, node_dim]`` per-residue node representation
            after the attention-based pair readout.
        node_repr: ``[1, N, node_dim]`` pre-readout node representation used
            by :meth:`UMAInverse._autoregressive_context`.
        lig_ctx: ``[1, L, pair_dim]`` residue→ligand context projection.
        pair_mask: ``[1, N, N]`` bool — broadcast of node masks.
        residue_mask: ``[1, L]`` bool — True for every parsed residue (never
            False in the inference path; kept for interface parity).
        residue_count: L (cached to avoid shape inspection).
        residue_backbone_coords: ``[1, L, 4, 3]`` N/Cα/C/O coords used by
            the multi-atom pair-distance path (v2 phase 3). ``None`` when
            the session is configured for ``pair_distance_atoms="anchor_only"``.
        pdb_path: Original path the structure was loaded from.
    """

    residue_ids: list[str]
    residue_id_to_index: dict[str, int]
    chain_ids: list[str]
    native_sequence: torch.Tensor
    design_mask: torch.Tensor
    z: torch.Tensor
    node_repr_res: torch.Tensor
    node_repr: torch.Tensor
    lig_ctx: torch.Tensor
    pair_mask: torch.Tensor
    residue_mask: torch.Tensor
    residue_count: int
    pdb_path: str
    residue_backbone_coords: torch.Tensor | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def length(self) -> int:
        """Number of residues (``L``)."""
        return self.residue_count


# ─── Session ──────────────────────────────────────────────────────────────────


class InferenceSession:
    """Hold a loaded model and encode one or more PDB structures.

    Typical usage::

        session = InferenceSession.from_checkpoint(
            config_path="configs/config.yaml",
            checkpoint="checkpoints/last.ckpt",
            device="cuda",
        )
        ctx = session.load_structure("structure.pdb")
        # ctx is now ready to pass into src.inference.decoding

    The constructor is private — prefer :meth:`from_checkpoint`.
    """

    def __init__(
        self,
        model: UMAInverse,
        config: DictConfig,
        device: torch.device,
        checkpoint_path: str | None = None,
    ) -> None:
        self.model = model.eval()
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        # v2/v3: data layer is the authoritative source; the model
        # reads the same values via OmegaConf interpolation.
        data_section = config.data if "data" in config else {}
        self.ligand_featurizer = str(data_section.get("ligand_featurizer", "onehot6"))
        self.residue_anchor = str(data_section.get("residue_anchor", "ca"))
        self.pair_distance_atoms = str(data_section.get("pair_distance_atoms", "anchor_only"))
        self.pair_distance_atoms_ligand = str(
            data_section.get("pair_distance_atoms_ligand", "anchor_only")
        )
        self.frame_relative_angles = bool(data_section.get("frame_relative_angles", False))
        # Backbone coords are needed for any backbone_full* setting (v2 + v3),
        # for the v3 L-M backbone-pair distances, and to compute v3 frame
        # angles. Centralised so load_structure / _init_pair stay in sync.
        self._needs_backbone_coords = (
            self.pair_distance_atoms.startswith("backbone_full")
            or self.pair_distance_atoms_ligand.startswith("backbone_full")
            or self.frame_relative_angles
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        config_path: PathLike,
        checkpoint: PathLike | None,
        device: str = "auto",
    ) -> InferenceSession:
        """Load a model configuration and (optional) checkpoint.

        Args:
            config_path: Path to the Hydra config YAML (e.g. ``configs/config.yaml``).
                Only the ``model`` and ``data`` sections are read.
            checkpoint: Path to a ``.ckpt`` file produced by training. When
                ``None`` the model is left at random initialisation (useful
                for smoke tests; not for real inference).
            device: ``"cuda"``, ``"cpu"``, or ``"auto"`` (picks CUDA when
                available).

        Returns:
            A configured session ready for :meth:`load_structure`.

        Raises:
            FileNotFoundError: If *config_path* or a provided *checkpoint*
                does not exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"config not found: {config_path}")

        cfg = OmegaConf.load(config_path)
        resolved_device = _resolve_device(device)

        checkpoint_str: str | None = None
        ckpt: dict | None = None
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            checkpoint_str = str(checkpoint_path)
            ckpt = torch.load(checkpoint_str, map_location="cpu", weights_only=False)
            # A checkpoint embeds the architecture it was trained with
            # (Lightning ``hyper_parameters.model_config``). Prefer it over the
            # YAML so a drifting ``configs/config.yaml`` can't desync the model
            # from its weights — a mismatch loads silently via ``strict=False``
            # and yields a degenerate model. Falls back to the YAML when absent.
            cfg = _apply_embedded_model_config(cfg, ckpt)

        model_config = OmegaConf.to_container(cfg.model, resolve=True)
        model = UMAInverse(model_config)  # type: ignore[arg-type]

        if ckpt is not None:
            _load_weights_from_ckpt(model, ckpt, checkpoint_str)

        model = model.to(resolved_device)
        return cls(model=model, config=cfg, device=resolved_device, checkpoint_path=checkpoint_str)

    # ── Structure encoding ────────────────────────────────────────────────────

    @torch.no_grad()
    def load_structure(
        self,
        pdb_path: PathLike,
        parse_chains: list[str] | None = None,
        include_zero_occupancy: bool = False,
        ligand_cutoff: float | None = None,
        mask_ligand: bool = False,
        max_total_nodes: int | None = None,
    ) -> StructureContext:
        """Featurize a PDB and run the structure encoder.

        Args:
            pdb_path: Path to the input PDB file.
            parse_chains: When given, only residues from these chain IDs are
                parsed (ligand atoms are still drawn from every chain).
            include_zero_occupancy: Keep atoms whose occupancy is exactly
                zero (mirrors LigandMPNN's ``parse_atoms_with_zero_occupancy``).
            ligand_cutoff: Å distance threshold for marking ligand atoms near
                Cα. Falls back to ``cfg.data.cutoff_for_score`` when omitted.
            mask_ligand: Replace all ligand atom features with zero — the
                ablation LigandMPNN exposes as ``--ligand_mpnn_use_atom_context 0``.
                Also zeroes the ligand mask so pair-tensor entries involving
                ligand atoms contribute nothing.
            max_total_nodes: Override the configured residue-crop budget.
                Use a very large number (or ``None`` default) when running
                inference to avoid silently dropping user-referenced
                residues. When the parser must crop, chain letters and
                residue numbers are threaded through so constraints remain
                well-defined on the kept subset.

        Returns:
            A :class:`StructureContext` with all tensors already on device.

        Raises:
            FileNotFoundError: If *pdb_path* does not exist.
            ValueError: If the PDB contains no parseable residues.
        """
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        data_cfg = self.config.data
        cutoff = float(ligand_cutoff) if ligand_cutoff is not None else float(data_cfg.cutoff_for_score)
        nodes_budget = int(max_total_nodes) if max_total_nodes is not None else int(data_cfg.max_total_nodes)

        example = load_example_from_pdb(
            pdb_path=str(pdb_path),
            ligand_context_atoms=int(data_cfg.ligand_context_atoms),
            cutoff_for_score=cutoff,
            max_total_nodes=nodes_budget,
            device="cpu",
            parse_chains=parse_chains,
            include_zero_occupancy=include_zero_occupancy,
            return_residue_ids=True,
            ligand_featurizer=self.ligand_featurizer,
            residue_anchor=self.residue_anchor,
            return_backbone_coords=self._needs_backbone_coords,
            return_frame_relative_angles=self.frame_relative_angles,
        )

        residue_ids: list[str] = example["residue_ids"]  # type: ignore[assignment]
        chain_ids: list[str] = [rid[0] for rid in residue_ids]
        residue_id_to_index: dict[str, int] = {rid: i for i, rid in enumerate(residue_ids)}
        if len(residue_id_to_index) != len(residue_ids):
            # PDB files occasionally have duplicate residue identifiers (alt
            # conformations expressed as same chain+resnum). The parser keeps
            # the first occurrence via BioPython default; flag loudly so a
            # downstream constraint referencing the id doesn't silently bind
            # to an unintended copy.
            seen: dict[str, int] = {}
            dups: list[str] = []
            for rid in residue_ids:
                if rid in seen:
                    dups.append(rid)
                seen[rid] = seen.get(rid, 0) + 1
            logger.warning(
                "%s: duplicate residue ids detected (first 5: %s) — "
                "constraints will bind to the first occurrence",
                pdb_path,
                dups[:5],
            )

        device = self.device
        residue_coords = example["residue_coords"].to(device).unsqueeze(0)
        residue_features = example["residue_features"].to(device).unsqueeze(0)
        residue_mask = example["residue_mask"].bool().to(device).unsqueeze(0)
        ligand_coords = example["ligand_coords"].to(device).unsqueeze(0)
        ligand_mask = example["ligand_mask"].bool().to(device).unsqueeze(0)
        native_sequence = example["sequence"].to(device).long()
        design_mask = example["design_mask"].bool().to(device)
        residue_backbone_coords: torch.Tensor | None = None
        if self._needs_backbone_coords:
            residue_backbone_coords = (
                example["residue_backbone_coords"].to(device).unsqueeze(0)
            )
        residue_ligand_frame_angles: torch.Tensor | None = None
        if self.frame_relative_angles and "residue_ligand_frame_angles" in example:
            residue_ligand_frame_angles = (
                example["residue_ligand_frame_angles"].to(device).unsqueeze(0)
            )

        # Featurizer-specific ligand tensor. For mask_ligand:
        #   onehot6                  → zero the [M, 6] feature vector
        #   atomic_number_embedding  → zero atomic numbers; nn.Embedding maps
        #                              index 0 to padding_idx → zero vector
        #   ligandmpnn_atomic        → zero atomic numbers; the LigandMPNN
        #                              featurizer's group/period one-hots fire
        #                              at slot 0 ("no group", "no period"),
        #                              which is the trained "no atom" row
        if self.ligand_featurizer == "onehot6":
            ligand_features = example["ligand_features"].to(device).unsqueeze(0)
            if mask_ligand:
                ligand_features = torch.zeros_like(ligand_features)
        else:  # "atomic_number_embedding" or "ligandmpnn_atomic"
            ligand_atomic_numbers = example["ligand_atomic_numbers"].to(device).unsqueeze(0)
            if mask_ligand:
                ligand_atomic_numbers = torch.zeros_like(ligand_atomic_numbers)

        if mask_ligand:
            ligand_mask = torch.zeros_like(ligand_mask)

        residue_count = residue_coords.shape[1]
        model = self.model

        residue_repr = model.residue_in(residue_features)
        if self.ligand_featurizer == "onehot6":
            ligand_repr = model.ligand_in(ligand_features)
        else:  # "atomic_number_embedding" or "ligandmpnn_atomic"
            ligand_repr = model.ligand_in(ligand_atomic_numbers)
        node_repr = model.node_norm(torch.cat([residue_repr, ligand_repr], dim=1))
        coords_all = torch.cat([residue_coords, ligand_coords], dim=1)
        node_mask = torch.cat([residue_mask, ligand_mask], dim=1)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]

        z = model._init_pair(
            node_repr=node_repr,
            coords=coords_all,
            pair_mask=pair_mask,
            residue_count=residue_count,
            residue_backbone_coords=residue_backbone_coords,
            residue_ligand_frame_angles=residue_ligand_frame_angles,
        )
        z = model.encoder(z, pair_mask.to(dtype=z.dtype))

        # Attention-based pair-to-node readout reproducing UMAInverse.forward
        z_res = z[:, :residue_count, :, :]
        readout_logits = model.pair_readout_attn(z_res).squeeze(-1)
        readout_logits = readout_logits.masked_fill(~node_mask.unsqueeze(1).bool(), -1e4)
        readout_w = torch.softmax(readout_logits, dim=-1)
        z_pooled = torch.einsum("bln,blnd->bld", readout_w, z_res)
        node_repr_res = node_repr[:, :residue_count, :] + model.node_readout(z_pooled)

        lig_ctx = model._ligand_aware_context(z, pair_mask, residue_count=residue_count)

        return StructureContext(
            residue_ids=residue_ids,
            residue_id_to_index=residue_id_to_index,
            chain_ids=chain_ids,
            native_sequence=native_sequence,
            design_mask=design_mask,
            z=z,
            node_repr_res=node_repr_res,
            node_repr=node_repr,
            lig_ctx=lig_ctx,
            pair_mask=pair_mask,
            residue_mask=residue_mask,
            residue_count=residue_count,
            pdb_path=str(pdb_path),
            residue_backbone_coords=residue_backbone_coords,
            device=device,
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_device(device: str) -> torch.device:
    """Map a user-facing device string to a ``torch.device``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return torch.device("cpu")
    return torch.device(device)


# Featurization flags that must match the checkpoint's training config or the
# model receives wrong inputs (these live in cfg.data and drive load_structure).
_FEATURIZER_FLAGS = (
    "ligand_featurizer",
    "residue_anchor",
    "pair_distance_atoms",
    "pair_distance_atoms_ligand",
    "frame_relative_angles",
)


def _apply_embedded_model_config(cfg: DictConfig, ckpt: object) -> DictConfig:
    """Override ``cfg.model`` (and mirror featurizer flags into ``cfg.data``) with
    the checkpoint's embedded ``model_config``, when present.

    Lightning checkpoints store the exact ``model_config`` they were trained with
    under ``hyper_parameters``. Using it makes the loaded model self-describing,
    so the architecture (and the featurization the data layer performs) always
    matches the weights regardless of how ``configs/config.yaml`` has drifted.
    """
    model_config = None
    if isinstance(ckpt, dict):
        model_config = (ckpt.get("hyper_parameters") or {}).get("model_config")
    if not model_config:
        return cfg

    mc = OmegaConf.create(dict(model_config))
    cfg.model = mc
    if "data" not in cfg:
        cfg.data = {}
    for flag in _FEATURIZER_FLAGS:
        if flag in mc:
            cfg.data[flag] = mc[flag]
    logger.info(
        "using checkpoint-embedded architecture (ligand_featurizer=%s, "
        "pair_distance_atoms=%s, frame_relative_angles=%s)",
        mc.get("ligand_featurizer"),
        mc.get("pair_distance_atoms"),
        mc.get("frame_relative_angles"),
    )
    return cfg


def _load_weights_from_ckpt(model: UMAInverse, ckpt: object, checkpoint_path: str | None) -> None:
    """Load an already-loaded Lightning checkpoint dict into a bare ``UMAInverse``.

    Training saves state under ``model.<param>`` prefixes because the Lightning
    module wraps the pure torch model. This strips the prefix so the loaded state
    maps 1:1 onto the module's parameters.
    """
    if not isinstance(ckpt, dict):
        raise ValueError(f"checkpoint is not a dict: {checkpoint_path}")
    state_dict = ckpt.get("state_dict", ckpt)
    cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("checkpoint missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("checkpoint unexpected keys (%d): %s", len(unexpected), unexpected[:5])
    logger.info("loaded weights from %s", os.path.basename(checkpoint_path or "?"))


def _load_weights(model: UMAInverse, checkpoint_path: str) -> None:
    """Load a Lightning ``.ckpt`` file into a bare ``UMAInverse`` module."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _load_weights_from_ckpt(model, ckpt, checkpoint_path)
