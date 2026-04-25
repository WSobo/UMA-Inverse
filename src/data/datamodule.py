"""PyTorch Lightning DataModule for UMA-Inverse.

Wraps UMAInverseDataset (PDB → featurised tensors) with proper error handling:
- Recursion depth guard when samples fail to load (avoids infinite retry loops)
- Failed PDB IDs are logged to ``logs/failed_pdbs.txt`` for post-hoc audit
- Specific exception handling instead of bare ``except Exception``
"""
import logging
import os
import random
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src import PROJECT_ROOT
from .ligandmpnn_bridge import load_example_from_pdb, load_json_ids, resolve_pdb_path

logger = logging.getLogger(__name__)

_FAILED_PDB_LOG = os.path.join(PROJECT_ROOT, "logs", "failed_pdbs.txt")
_MAX_RETRY_DEPTH = 5   # max recursive fallback attempts per sample


def _log_failed_pdb(pdb_id: str, reason: str) -> None:
    """Append a failed PDB ID + reason to the failure log."""
    try:
        os.makedirs(os.path.dirname(_FAILED_PDB_LOG), exist_ok=True)
        with open(_FAILED_PDB_LOG, "a") as f:
            f.write(f"{pdb_id}\t{reason}\n")
    except OSError:
        pass  # never crash the training loop over logging


def _apply_runtime_crop(item: Dict[str, torch.Tensor], max_total_nodes: int) -> Dict[str, torch.Tensor]:
    """Re-crop a cached sample to max_total_nodes (residues + ligand atoms).

    The preprocessing cache is built with a large cap (e.g. 1024). When the
    curriculum asks for a tighter crop, we pick the residues closest to the
    ligand centroid — matching the selection logic used at preprocessing time.
    """
    residue_coords = item["residue_coords"]
    ligand_coords  = item["ligand_coords"]
    n_res = residue_coords.shape[0]
    n_lig = ligand_coords.shape[0]

    if n_res + n_lig <= max_total_nodes:
        return item

    max_residues = max(1, max_total_nodes - n_lig)
    if n_res <= max_residues:
        return item

    if n_lig > 0:
        center = ligand_coords.mean(dim=0, keepdim=True)
        dist = torch.linalg.norm(residue_coords - center, dim=-1)
        keep = torch.topk(dist, k=max_residues, largest=False).indices
        keep, _ = torch.sort(keep)
    else:
        keep = torch.arange(max_residues, device=residue_coords.device)

    out = {
        **item,
        "residue_coords":   item["residue_coords"][keep],
        "residue_features": item["residue_features"][keep],
        "residue_mask":     item["residue_mask"][keep],
        "sequence":         item["sequence"][keep],
        "design_mask":      item["design_mask"][keep],
    }
    # Phase 3: crop backbone coords with the same selection so they stay
    # aligned with residue_coords/residue_features.
    if "residue_backbone_coords" in item:
        out["residue_backbone_coords"] = item["residue_backbone_coords"][keep]
    return out


class UMAInverseDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        pdb_dir: str,
        processed_dir: str,
        ligand_context_atoms: int,
        cutoff_for_score: float,
        max_total_nodes: int,
        ligand_featurizer: str = "onehot6",
        residue_anchor: str = "ca",
        return_backbone_coords: bool = False,
    ) -> None:
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir
        self.ligand_context_atoms = ligand_context_atoms
        self.cutoff_for_score = cutoff_for_score
        self.max_total_nodes = max_total_nodes
        self.ligand_featurizer = ligand_featurizer
        self.residue_anchor = residue_anchor
        self.return_backbone_coords = return_backbone_coords

        ids = load_json_ids(json_path)
        self.pdb_ids = [
            pdb_id for pdb_id in ids
            if (
                os.path.exists(os.path.join(processed_dir, f"{pdb_id}.pt"))
                or resolve_pdb_path(pdb_dir, pdb_id) is not None
            )
        ]

        # Phase 5 fix: pre-filter zero-residue cached samples (e.g. DNA-only
        # PDBs like 100d) so overfit_batches=1 deterministically lands on a
        # valid sample at index 0. The scan is cached to
        # <processed_dir>/_zero_residue_ids.txt so subsequent runs are instant.
        self.pdb_ids = self._filter_zero_residue_ids(self.pdb_ids, processed_dir)

        if not self.pdb_ids:
            raise RuntimeError(
                f"No valid PDB entries found for {json_path}. "
                f"Check pdb_dir={pdb_dir} and processed_dir={processed_dir}."
            )
        logger.info("Dataset loaded: %d structures from %s", len(self.pdb_ids), json_path)

    @staticmethod
    def _filter_zero_residue_ids(
        candidate_ids: List[str], processed_dir: str
    ) -> List[str]:
        """Drop cached PDBs with 0 residues (DNA/RNA-only structures).

        Scans the cache once and writes a reusable blacklist under
        ``<processed_dir>/_zero_residue_ids.txt``. The scan is skipped
        when the blacklist already exists; delete it to force a re-scan
        after cache regeneration.
        """
        blacklist_path = os.path.join(processed_dir, "_zero_residue_ids.txt")
        if os.path.exists(blacklist_path):
            with open(blacklist_path) as f:
                bad = {line.strip() for line in f if line.strip()}
            return [p for p in candidate_ids if p not in bad]

        logger.info(
            "Building zero-residue cache blacklist (one-time scan of %d entries)...",
            len(candidate_ids),
        )
        bad: List[str] = []
        for pid in candidate_ids:
            path = os.path.join(processed_dir, f"{pid}.pt")
            if not os.path.exists(path):
                continue
            try:
                item = torch.load(path, map_location="cpu", weights_only=True)
                if item["residue_coords"].shape[0] == 0:
                    bad.append(pid)
            except (RuntimeError, EOFError, OSError, KeyError):
                # Corrupt / unexpected schema — treat as bad too so we
                # don't retry in the hot path for every epoch.
                bad.append(pid)

        try:
            os.makedirs(processed_dir, exist_ok=True)
            with open(blacklist_path, "w") as f:
                f.write("\n".join(bad) + ("\n" if bad else ""))
            logger.info(
                "Zero-residue blacklist: %d entries written to %s",
                len(bad), blacklist_path,
            )
        except OSError as e:
            logger.warning("Could not write blacklist %s: %s", blacklist_path, e)

        bad_set = set(bad)
        return [p for p in candidate_ids if p not in bad_set]

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def _adapt_cached_item(
        self, item: Dict[str, torch.Tensor]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Project a cached item onto the currently-configured feature set.

        The preprocessor writes a "union cache" with every v2 key, so we can
        derive the right feature combination on the fly without re-running
        preprocess every time a data flag flips. Returns ``None`` when the
        cached item can't satisfy the current flags (e.g. pre-Phase 4 caches
        that lack ``residue_backbone_coords`` against a config that needs it).
        """
        out = dict(item)

        # If the user wants virtual Cβ and the cache has backbone coords,
        # derive Cβ and overwrite residue_coords. The cache's residue_coords
        # is the Cα baseline; residue_backbone_coords[:, 1, :] is identical
        # to it by construction.
        if self.residue_anchor == "cb":
            if "residue_backbone_coords" not in out:
                return None
            from .ligandmpnn_bridge import _construct_virtual_cb
            out["residue_coords"] = _construct_virtual_cb(out["residue_backbone_coords"])

        # Required ligand feature key for the configured featurizer.
        needed_lig_key = (
            "ligand_features"
            if self.ligand_featurizer == "onehot6"
            else "ligand_atomic_numbers"
        )
        if needed_lig_key not in out:
            return None
        # Drop the other featurizer key so the batch is clean.
        other_lig_key = (
            "ligand_atomic_numbers"
            if self.ligand_featurizer == "onehot6"
            else "ligand_features"
        )
        out.pop(other_lig_key, None)

        # Backbone coords: keep only when the model will consume them.
        if self.return_backbone_coords:
            if "residue_backbone_coords" not in out:
                return None
        else:
            out.pop("residue_backbone_coords", None)

        return out

    def __getitem__(self, idx: int, _depth: int = 0) -> Dict[str, torch.Tensor]:
        if _depth >= _MAX_RETRY_DEPTH:
            raise RuntimeError(
                f"UMAInverseDataset: exceeded {_MAX_RETRY_DEPTH} retries — "
                "too many consecutive bad samples. Check your data."
            )

        pdb_id = self.pdb_ids[idx]

        # Fast path: load pre-computed cached tensor. Since Phase 4 the
        # preprocessor emits a "union cache" — every file carries both
        # ligand_features and ligand_atomic_numbers, residue_backbone_coords,
        # and residue_anchor_atom="ca". We derive virtual Cβ on-the-fly when
        # the current config asks for it, so one cache serves any flag combo.
        # Legacy v1 caches (ligand_features only, no backbone coords) are
        # still usable when the flags happen to be all-v1.
        processed_path = os.path.join(self.processed_dir, f"{pdb_id}.pt")
        if os.path.exists(processed_path):
            try:
                item = torch.load(processed_path, map_location="cpu", weights_only=True)
                item = self._adapt_cached_item(item)
                # Zero-residue samples (e.g. DNA-only PDBs like 100d that
                # slipped into pre-v2 caches) produce NaN loss under
                # cross-entropy-over-zero-valid-tokens. Skip them.
                if item is not None and item["residue_coords"].shape[0] > 0:
                    item = _apply_runtime_crop(item, self.max_total_nodes)
                    item["pdb_id"] = pdb_id
                    return item
                # Cache can't satisfy the current flags or has 0 residues —
                # fall through to the slow PDB path.
                if item is not None:
                    _log_failed_pdb(pdb_id, "cache_zero_residues")
            except (RuntimeError, EOFError, OSError) as e:
                logger.warning("Corrupted cache for %s (%s) — falling back to PDB", pdb_id, e)
                _log_failed_pdb(pdb_id, f"cache_corrupt:{e}")

        # Slow path: parse PDB on the fly
        pdb_path = resolve_pdb_path(self.pdb_dir, pdb_id)
        if pdb_path is None:
            logger.debug("PDB not found for %s — sampling replacement", pdb_id)
            _log_failed_pdb(pdb_id, "pdb_not_found")
            return self.__getitem__(random.randrange(len(self.pdb_ids)), _depth + 1)

        try:
            item = load_example_from_pdb(
                pdb_path=pdb_path,
                ligand_context_atoms=self.ligand_context_atoms,
                cutoff_for_score=self.cutoff_for_score,
                max_total_nodes=self.max_total_nodes,
                ligand_featurizer=self.ligand_featurizer,
                residue_anchor=self.residue_anchor,
                return_backbone_coords=self.return_backbone_coords,
            )
        except (ValueError, OSError, RuntimeError) as e:
            logger.warning("Failed to featurize %s (%s) — sampling replacement", pdb_id, e)
            _log_failed_pdb(pdb_id, f"featurize_error:{type(e).__name__}:{e}")
            return self.__getitem__(random.randrange(len(self.pdb_ids)), _depth + 1)

        item["pdb_id"] = pdb_id
        return item


# ── Collation helpers ─────────────────────────────────────────────────────────

def _pad_2d(
    items: List[torch.Tensor], max_len: int, feat_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    out = torch.zeros((len(items), max_len, feat_dim), dtype=dtype)
    for i, t in enumerate(items):
        if t.shape[0] > 0:
            out[i, : t.shape[0], :] = t
    return out


def _pad_1d(
    items: List[torch.Tensor],
    max_len: int,
    dtype: torch.dtype,
    fill_value: int = 0,
) -> torch.Tensor:
    out = torch.full((len(items), max_len), fill_value=fill_value, dtype=dtype)
    for i, t in enumerate(items):
        if t.shape[0] > 0:
            out[i, : t.shape[0]] = t
    return out


def _pad_3d(
    items: List[torch.Tensor],
    max_len: int,
    feat_dim_1: int,
    feat_dim_2: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pad a list of [L_i, D1, D2] tensors into [B, max_len, D1, D2].

    Used for the optional residue_backbone_coords [L, 4, 3] tensor that
    phase 3 emits when the model computes multi-atom pair distances.
    """
    out = torch.zeros((len(items), max_len, feat_dim_1, feat_dim_2), dtype=dtype)
    for i, t in enumerate(items):
        if t.shape[0] > 0:
            out[i, : t.shape[0], :, :] = t
    return out


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a list of variable-length samples into a single batch dict.

    Ligand featurizer and backbone-coords presence are detected per-batch
    from the keys present in the first item. All items in a batch must
    agree on the featurizer / backbone flag.
    """
    max_res = max(item["residue_coords"].shape[0] for item in batch)
    max_lig = max(item["ligand_coords"].shape[0] for item in batch)

    uses_embedding = "ligand_atomic_numbers" in batch[0]
    uses_backbone = "residue_backbone_coords" in batch[0]

    out = {
        "residue_coords":   _pad_2d([b["residue_coords"]   for b in batch], max_res, 3, torch.float32),
        "residue_features": _pad_2d([b["residue_features"] for b in batch], max_res, 6, torch.float32),
        "residue_mask":     _pad_1d([b["residue_mask"].to(torch.bool)  for b in batch], max_res, torch.bool),
        "sequence":         _pad_1d([b["sequence"].to(torch.long)      for b in batch], max_res, torch.long, fill_value=20),
        "design_mask":      _pad_1d([b["design_mask"].to(torch.bool)   for b in batch], max_res, torch.bool),
        "ligand_coords":    _pad_2d([b["ligand_coords"]    for b in batch], max_lig, 3, torch.float32),
        "ligand_mask":      _pad_1d([b["ligand_mask"].to(torch.bool)   for b in batch], max_lig, torch.bool),
        "pdb_id":           [b["pdb_id"] for b in batch],
    }
    if uses_embedding:
        # Pad with fill_value=0 so padded slots map to the embedding's
        # reserved padding_idx (their node vectors are then identically zero
        # and are suppressed by ligand_mask anyway).
        out["ligand_atomic_numbers"] = _pad_1d(
            [b["ligand_atomic_numbers"].to(torch.long) for b in batch],
            max_lig, torch.long, fill_value=0,
        )
    else:
        out["ligand_features"] = _pad_2d(
            [b["ligand_features"] for b in batch], max_lig, 6, torch.float32,
        )
    if uses_backbone:
        out["residue_backbone_coords"] = _pad_3d(
            [b["residue_backbone_coords"] for b in batch], max_res, 4, 3, torch.float32,
        )
    return out


# ── DataModule ────────────────────────────────────────────────────────────────

class UMAInverseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_json: str,
        valid_json: str,
        pdb_dir: str,
        batch_size: int = 1,
        num_workers: int = 2,
        pin_memory: bool = True,
        ligand_context_atoms: int = 25,
        cutoff_for_score: float = 8.0,
        max_total_nodes: int = 384,
        processed_dir: Optional[str] = None,
        ligand_featurizer: str = "onehot6",
        residue_anchor: str = "ca",
        return_backbone_coords: bool = False,
    ) -> None:
        super().__init__()
        self.train_json = train_json
        self.valid_json = valid_json
        self.pdb_dir = pdb_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ligand_context_atoms = ligand_context_atoms
        self.cutoff_for_score = cutoff_for_score
        self.max_total_nodes = max_total_nodes
        self.processed_dir = processed_dir or os.path.join(PROJECT_ROOT, "data", "processed")
        self.ligand_featurizer = ligand_featurizer
        self.residue_anchor = residue_anchor
        self.return_backbone_coords = return_backbone_coords

        self.train_dataset: Optional[UMAInverseDataset] = None
        self.valid_dataset: Optional[UMAInverseDataset] = None

    def _make_dataset(self, json_path: str) -> UMAInverseDataset:
        return UMAInverseDataset(
            json_path=json_path,
            pdb_dir=self.pdb_dir,
            processed_dir=self.processed_dir,
            ligand_context_atoms=self.ligand_context_atoms,
            cutoff_for_score=self.cutoff_for_score,
            max_total_nodes=self.max_total_nodes,
            ligand_featurizer=self.ligand_featurizer,
            residue_anchor=self.residue_anchor,
            return_backbone_coords=self.return_backbone_coords,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset(self.train_json)
            self.valid_dataset = self._make_dataset(self.valid_json)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_batch,
        )

    def val_dataloader(self) -> DataLoader:
        if self.valid_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before val_dataloader().")
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_batch,
        )
