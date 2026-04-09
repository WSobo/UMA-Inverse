import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.ligandmpnn_bridge import load_example_from_pdb
from src.models.uma_inverse import UMAInverse
from src.utils.io import ids_to_sequence, write_fasta


def _to_batch(example):
    batch = {}
    for key, value in example.items():
        if key.endswith("_mask"):
            batch[key] = value.unsqueeze(0).bool()
        else:
            batch[key] = value.unsqueeze(0)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="UMA-Inverse inference")
    parser.add_argument("--pdb", type=str, required=True, help="Path to a PDB file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path")
    parser.add_argument("--out_fasta", type=str, default="outputs/predicted.fasta", help="Output fasta")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--fixed_residues", type=str, default=None, help="Comma-separated list of zero-indexed residue positions to keep native")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    model = UMAInverse(OmegaConf.to_container(cfg.model, resolve=True))

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        cleaned = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    example = load_example_from_pdb(
        pdb_path=args.pdb,
        ligand_context_atoms=int(cfg.data.ligand_context_atoms),
        cutoff_for_score=float(cfg.data.cutoff_for_score),
        max_total_nodes=int(cfg.data.max_total_nodes),
    )
    batch = _to_batch(example)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    with torch.no_grad():
        out = model(batch)
        logits = out["logits"]

        temperature = args.temperature
        if temperature is None:
            temperature = float(cfg.inference.temperature)

        if temperature <= 0.0:
            probs = torch.softmax(logits, dim=-1)
            pred_ids = torch.argmax(probs, dim=-1)[0]
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            pred_ids = torch.multinomial(probs[0], num_samples=1).squeeze(-1)

    residue_mask = batch["residue_mask"][0]
    design_mask = batch["design_mask"][0]
    
    native_classes = batch["sequence"][0]
    
    if args.fixed_residues is not None:
        try:
            fixed_indices = [int(i.strip()) for i in args.fixed_residues.split(",")]
            for idx in fixed_indices:
                if idx < len(pred_ids) and residue_mask[idx]:
                    pred_ids[idx] = native_classes[idx]
                    design_mask[idx] = False # No longer counted as predicted
        except Exception as e:
            print(f"Failed parsing fixed residues: {e}")

    # Calculate Native Sequence Recovery (NSR) over designer residues
    matches = ((pred_ids == native_classes) & residue_mask & design_mask).sum().item()
    total_residues = (residue_mask & design_mask).sum().item()
    nsr_percentage = (matches / total_residues) * 100.0 if total_residues > 0 else 0.0

    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Native Sequence Recovery (NSR): {nsr_percentage:.2f}% ({matches}/{total_residues} designed residues)")
    print("="*50 + "\n")

    pred_ids = pred_ids[residue_mask]
    sequence = ids_to_sequence(pred_ids.tolist())

    out_dir = os.path.dirname(args.out_fasta)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_fasta(args.out_fasta, header=os.path.basename(args.pdb), sequence=sequence)
    print(f"Wrote sequence to {args.out_fasta}")


if __name__ == "__main__":
    main()
