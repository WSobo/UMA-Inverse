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

    return {
        **item,
        "residue_coords":   item["residue_coords"][keep],
        "residue_features": item["residue_features"][keep],
        "residue_mask":     item["residue_mask"][keep],
        "sequence":         item["sequence"][keep],
        "design_mask":      item["design_mask"][keep],
    }


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
    ) -> None:
        self.pdb_dir = pdb_dir
        self.processed_dir = processed_dir
        self.ligand_context_atoms = ligand_context_atoms
        self.cutoff_for_score = cutoff_for_score
        self.max_total_nodes = max_total_nodes
        self.ligand_featurizer = ligand_featurizer

        ids = load_json_ids(json_path)
        self.pdb_ids = [
            pdb_id for pdb_id in ids
            if (
                os.path.exists(os.path.join(processed_dir, f"{pdb_id}.pt"))
                or resolve_pdb_path(pdb_dir, pdb_id) is not None
            )
        ]

        if not self.pdb_ids:
            raise RuntimeError(
                f"No valid PDB entries found for {json_path}. "
                f"Check pdb_dir={pdb_dir} and processed_dir={processed_dir}."
            )
        logger.info("Dataset loaded: %d structures from %s", len(self.pdb_ids), json_path)

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int, _depth: int = 0) -> Dict[str, torch.Tensor]:
        if _depth >= _MAX_RETRY_DEPTH:
            raise RuntimeError(
                f"UMAInverseDataset: exceeded {_MAX_RETRY_DEPTH} retries — "
                "too many consecutive bad samples. Check your data."
            )

        pdb_id = self.pdb_ids[idx]

        # Expected ligand feature key for the configured featurizer. Cached
        # .pt files predate v2 and only carry "ligand_features"; when the
        # caller asks for the embedding path, skip the cache entirely and
        # rebuild from PDB so the batch never mixes featurizer conventions.
        cache_key = (
            "ligand_features"
            if self.ligand_featurizer == "onehot6"
            else "ligand_atomic_numbers"
        )

        # Fast path: load pre-computed cached tensor
        processed_path = os.path.join(self.processed_dir, f"{pdb_id}.pt")
        if os.path.exists(processed_path):
            try:
                item = torch.load(processed_path, map_location="cpu", weights_only=True)
                if cache_key in item:
                    item = _apply_runtime_crop(item, self.max_total_nodes)
                    item["pdb_id"] = pdb_id
                    return item
                # Cache exists but was built with a different featurizer — fall
                # through to the slow PDB path rather than silently mismatching.
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


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a list of variable-length samples into a single batch dict.

    Ligand featurizer is detected per-batch from the keys present in the
    first item. All items in a batch must agree on the featurizer.
    """
    max_res = max(item["residue_coords"].shape[0] for item in batch)
    max_lig = max(item["ligand_coords"].shape[0] for item in batch)

    uses_embedding = "ligand_atomic_numbers" in batch[0]

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
