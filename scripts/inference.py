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

    residue_mask = batch["residue_mask"][0]
    design_mask = batch["design_mask"][0]
    native_classes = batch["sequence"][0].clone()
    
    fixed_indices = []
    if args.fixed_residues is not None:
        try:
            fixed_indices = [int(i.strip()) for i in args.fixed_residues.split(",")]
        except Exception as e:
            print(f"Failed parsing fixed residues: {e}")

    # Create decoding order (fixed contexts first, then randomly ordered designed residues)
    # However, for pure inference, we usually just decode left-to-right, so we can keep the default torch.arange or pass an explicit order.
    # To make fixed residues valid context, we can just ensure they are at the beginning of decoding order.
    L = residue_mask.shape[0]
    pred_seq = torch.full_like(batch["sequence"], fill_value=20)
    
    # We will build decoding order such that fixed residues come first
    decoding_order = torch.zeros((1, L), device=device, dtype=torch.long)
    is_fixed = torch.zeros(L, device=device, dtype=torch.bool)
    for idx in fixed_indices:
        if idx < L and residue_mask[idx]:
            is_fixed[idx] = True
            design_mask[idx] = False # No longer counted as predicted
    
    # Non-designed and fixed are "known"
    is_known = (~design_mask) | is_fixed
    
    # place knowns at beginning (0 to num_known-1)
    num_known = is_known.sum().item()
    if num_known > 0:
        decoding_order[0, is_known] = torch.arange(num_known, device=device)
        pred_seq[0, is_known] = native_classes[is_known]

    # place unknowns after
    num_unknown = L - num_known
    if num_unknown > 0:
        unknown_indices = (~is_known).nonzero(as_tuple=True)[0]
        # Sort unknown indices left-to-right for simple autoregressive step
        # Wait, if we want random, we can shuffle them. We use left-to-right to match training baseline if we like, or random.
        # Let's use left-to-right
        decoding_order[0, ~is_known] = torch.arange(num_known, L, device=device)
        
        # Sort so we iterate in the exact decoding order
        decode_steps = unknown_indices[torch.argsort(decoding_order[0, ~is_known])]
    else:
        decode_steps = []

    batch["decoding_order"] = decoding_order

    temperature = args.temperature
    if temperature is None:
        temperature = float(cfg.inference.temperature)

    # Autoregressive decoding loop generator
    # We can pre-compute encoder Z since it doesn't depend on sequence!
    # Wait, the pairmixer encoder only depends on coords and node features. It is invariant to sequence!
    # Thus, we can compute pair_repr ONCE to save time!
    with torch.no_grad():
        residue_coords = batch["residue_coords"]
        residue_features = batch["residue_features"]
        ligand_coords = batch["ligand_coords"]
        ligand_features = batch["ligand_features"]
        ligand_mask = batch["ligand_mask"].bool()
        residue_count = residue_coords.shape[1]

        residue_repr = model.residue_in(residue_features)
        ligand_repr = model.ligand_in(ligand_features)
        node_repr = torch.cat([residue_repr, ligand_repr], dim=1)
        node_repr = model.node_norm(node_repr)
        
        coords = torch.cat([residue_coords, ligand_coords], dim=1)
        node_mask = torch.cat([batch["residue_mask"].bool(), ligand_mask], dim=1)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        
        z = model._init_pair(node_repr=node_repr, coords=coords, pair_mask=pair_mask)
        z = model.encoder(z, pair_mask.to(dtype=z.dtype))
        lig_ctx = model._ligand_aware_context(z, pair_mask, residue_count=residue_count)

        for step_idx, res_idx in enumerate(decode_steps):
            batch["sequence"] = pred_seq
            
            ar_ctx = model._autoregressive_context(
                z=z,
                sequence=batch["sequence"],
                residue_mask=batch["residue_mask"].bool(),
                decoding_order=batch["decoding_order"],
            )
            
            decoder_input = torch.cat([node_repr[:, :residue_count, :] + ar_ctx, lig_ctx], dim=-1)
            logits = model.decoder(decoder_input)
            
            step_logits = logits[0, res_idx]

            if temperature <= 0.0:
                probs = torch.softmax(step_logits, dim=-1)
                pred_id = torch.argmax(probs, dim=-1)
            else:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                pred_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
            pred_seq[0, res_idx] = pred_id

    pred_ids = pred_seq[0]

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
