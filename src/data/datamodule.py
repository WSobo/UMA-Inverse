import os
import random
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .ligandmpnn_bridge import load_example_from_pdb, load_json_ids, resolve_pdb_path


class UMAInverseDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        pdb_dir: str,
        ligand_context_atoms: int,
        cutoff_for_score: float,
        max_total_nodes: int,
    ) -> None:
        self.pdb_dir = pdb_dir
        self.ligand_context_atoms = ligand_context_atoms
        self.cutoff_for_score = cutoff_for_score
        self.max_total_nodes = max_total_nodes

        ids = load_json_ids(json_path)
        self.pdb_ids = [pdb_id for pdb_id in ids if resolve_pdb_path(pdb_dir, pdb_id) is not None]

        if not self.pdb_ids:
            raise RuntimeError(f"No valid pdb entries found for {json_path} in {pdb_dir}")

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pdb_id = self.pdb_ids[idx]
        
        # Check if preprocessed .pt file exists (cached via scripts/preprocess.py)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        processed_path = os.path.join(project_root, "data", "processed", f"{pdb_id}.pt")
        
        if os.path.exists(processed_path):
            item = torch.load(processed_path, map_location="cpu", weights_only=True)
            item["pdb_id"] = pdb_id
            return item
            
        pdb_path = resolve_pdb_path(self.pdb_dir, pdb_id)
        if pdb_path is None:
            return self[random.randrange(len(self.pdb_ids))]

        try:
            item = load_example_from_pdb(
                pdb_path=pdb_path,
                ligand_context_atoms=self.ligand_context_atoms,
                cutoff_for_score=self.cutoff_for_score,
                max_total_nodes=self.max_total_nodes,
            )
        except Exception:
            return self[random.randrange(len(self.pdb_ids))]

        item["pdb_id"] = pdb_id
        return item


def _pad_2d(items: List[torch.Tensor], max_len: int, feat_dim: int, dtype: torch.dtype) -> torch.Tensor:
    out = torch.zeros((len(items), max_len, feat_dim), dtype=dtype)
    for i, tensor in enumerate(items):
        if tensor.shape[0] > 0:
            out[i, : tensor.shape[0], :] = tensor
    return out


def _pad_1d(items: List[torch.Tensor], max_len: int, dtype: torch.dtype, fill_value: int = 0) -> torch.Tensor:
    out = torch.full((len(items), max_len), fill_value=fill_value, dtype=dtype)
    for i, tensor in enumerate(items):
        if tensor.shape[0] > 0:
            out[i, : tensor.shape[0]] = tensor
    return out


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_res = max(item["residue_coords"].shape[0] for item in batch)
    max_lig = max(item["ligand_coords"].shape[0] for item in batch)

    residue_coords = _pad_2d([item["residue_coords"] for item in batch], max_res, 3, torch.float32)
    residue_features = _pad_2d([item["residue_features"] for item in batch], max_res, 6, torch.float32)
    residue_mask = _pad_1d([item["residue_mask"].to(torch.bool) for item in batch], max_res, torch.bool)
    sequence = _pad_1d([item["sequence"].to(torch.long) for item in batch], max_res, torch.long, fill_value=20)
    design_mask = _pad_1d([item["design_mask"].to(torch.bool) for item in batch], max_res, torch.bool)

    ligand_coords = _pad_2d([item["ligand_coords"] for item in batch], max_lig, 3, torch.float32)
    ligand_features = _pad_2d([item["ligand_features"] for item in batch], max_lig, 6, torch.float32)
    ligand_mask = _pad_1d([item["ligand_mask"].to(torch.bool) for item in batch], max_lig, torch.bool)

    return {
        "residue_coords": residue_coords,
        "residue_features": residue_features,
        "residue_mask": residue_mask,
        "sequence": sequence,
        "design_mask": design_mask,
        "ligand_coords": ligand_coords,
        "ligand_features": ligand_features,
        "ligand_mask": ligand_mask,
        "pdb_id": [item["pdb_id"] for item in batch],
    }


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

        self.train_dataset: Optional[UMAInverseDataset] = None
        self.valid_dataset: Optional[UMAInverseDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = UMAInverseDataset(
                json_path=self.train_json,
                pdb_dir=self.pdb_dir,
                ligand_context_atoms=self.ligand_context_atoms,
                cutoff_for_score=self.cutoff_for_score,
                max_total_nodes=self.max_total_nodes,
            )
            self.valid_dataset = UMAInverseDataset(
                json_path=self.valid_json,
                pdb_dir=self.pdb_dir,
                ligand_context_atoms=self.ligand_context_atoms,
                cutoff_for_score=self.cutoff_for_score,
                max_total_nodes=self.max_total_nodes,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting train_dataloader().")
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
            raise RuntimeError("DataModule.setup() must be called before requesting val_dataloader().")
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_batch,
        )
