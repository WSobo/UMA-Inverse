"""UMA-Inverse inference script.

Usage:
    uv run python scripts/inference.py --pdb path/to/structure.pdb --ckpt checkpoints/last.ckpt

Or via Makefile:
    make inference PDB=path/to/structure.pdb
"""
import argparse
import logging
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _to_batch(example: dict) -> dict:
    batch = {}
    for key, value in example.items():
        if key.endswith("_mask"):
            batch[key] = value.unsqueeze(0).bool()
        else:
            batch[key] = value.unsqueeze(0)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="UMA-Inverse inference")
    parser.add_argument("--pdb",      type=str, required=True,
                        help="Path to a PDB file")
    parser.add_argument("--config",   type=str, default="configs/config.yaml",
                        help="Path to config yaml")
    parser.add_argument("--ckpt",     type=str, default=None,
                        help="Optional checkpoint path")
    parser.add_argument("--out_fasta", type=str, default="outputs/predicted.fasta",
                        help="Output FASTA path")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (>0 = stochastic, 0 = argmax)")
    parser.add_argument("--fixed_residues", type=str, default=None,
                        help="Comma-separated zero-indexed residue positions to keep native")
    args = parser.parse_args()

    # ── Input validation ───────────────────────────────────────────────────────
    if not os.path.exists(args.pdb):
        parser.error(f"PDB file not found: {args.pdb}")
    if not os.path.exists(args.config):
        parser.error(f"Config file not found: {args.config}")
    if args.temperature < 0:
        parser.error(f"--temperature must be >= 0, got {args.temperature}")
    if args.ckpt and not os.path.exists(args.ckpt):
        parser.error(f"Checkpoint not found: {args.ckpt}")

    # ── Load model ─────────────────────────────────────────────────────────────
    cfg   = OmegaConf.load(args.config)
    model = UMAInverse(OmegaConf.to_container(cfg.model, resolve=True))

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            parser.error(f"Checkpoint at {args.ckpt} is not a dict — may be corrupted.")
        state_dict = ckpt.get("state_dict", ckpt)
        cleaned    = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning("Missing keys in checkpoint: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running inference on %s", device)
    model.to(device).eval()

    # ── Featurize ──────────────────────────────────────────────────────────────
    example = load_example_from_pdb(
        pdb_path=args.pdb,
        ligand_context_atoms=int(cfg.data.ligand_context_atoms),
        cutoff_for_score=float(cfg.data.cutoff_for_score),
        max_total_nodes=int(cfg.data.max_total_nodes),
    )
    batch = _to_batch(example)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    residue_mask   = batch["residue_mask"][0]
    design_mask    = batch["design_mask"][0].clone()
    native_classes = batch["sequence"][0].clone()
    L = residue_mask.shape[0]

    # ── Fixed residues ─────────────────────────────────────────────────────────
    fixed_indices = []
    if args.fixed_residues:
        try:
            fixed_indices = [int(i.strip()) for i in args.fixed_residues.split(",") if i.strip()]
        except ValueError as e:
            logger.warning("Could not parse --fixed_residues: %s", e)

    is_fixed = torch.zeros(L, dtype=torch.bool, device=device)
    for idx in fixed_indices:
        if 0 <= idx < L and residue_mask[idx]:
            is_fixed[idx] = True
            design_mask[idx] = False

    is_known  = (~design_mask) | is_fixed
    num_known = int(is_known.sum().item())

    # Decoding order: known residues first, then unknown left-to-right
    decoding_order = torch.zeros((1, L), dtype=torch.long, device=device)
    pred_seq       = torch.full_like(batch["sequence"], fill_value=20)

    if num_known > 0:
        decoding_order[0, is_known] = torch.arange(num_known, device=device)
        pred_seq[0, is_known]       = native_classes[is_known]

    num_unknown = L - num_known
    if num_unknown > 0:
        unknown_indices = (~is_known).nonzero(as_tuple=True)[0]
        decoding_order[0, ~is_known] = torch.arange(num_known, L, device=device)
        decode_steps = unknown_indices[torch.argsort(decoding_order[0, ~is_known])]
    else:
        decode_steps = torch.tensor([], dtype=torch.long, device=device)

    batch["decoding_order"] = decoding_order
    temperature = args.temperature

    # ── Autoregressive decoding ────────────────────────────────────────────────
    # Encode structure once (encoder is sequence-independent)
    with torch.no_grad():
        residue_coords   = batch["residue_coords"]
        residue_features = batch["residue_features"]
        ligand_coords    = batch["ligand_coords"]
        ligand_features  = batch["ligand_features"]
        ligand_mask      = batch["ligand_mask"].bool()
        residue_count    = residue_coords.shape[1]

        residue_repr = model.residue_in(residue_features)
        ligand_repr  = model.ligand_in(ligand_features)
        node_repr    = model.node_norm(torch.cat([residue_repr, ligand_repr], dim=1))
        coords       = torch.cat([residue_coords, ligand_coords], dim=1)
        node_mask    = torch.cat([batch["residue_mask"].bool(), ligand_mask], dim=1)
        pair_mask    = node_mask[:, :, None] & node_mask[:, None, :]

        z       = model._init_pair(node_repr=node_repr, coords=coords, pair_mask=pair_mask)
        z       = model.encoder(z, pair_mask.to(dtype=z.dtype))
        lig_ctx = model._ligand_aware_context(z, pair_mask, residue_count=residue_count)

        for res_idx in decode_steps:
            batch["sequence"] = pred_seq
            ar_ctx = model._autoregressive_context(
                z=z,
                sequence=batch["sequence"],
                residue_mask=batch["residue_mask"].bool(),
                decoding_order=batch["decoding_order"],
            )
            decoder_input = torch.cat(
                [node_repr[:, :residue_count, :] + ar_ctx, lig_ctx], dim=-1
            )
            step_logits = model.decoder(decoder_input)[0, res_idx]

            if temperature <= 0.0:
                pred_id = step_logits.argmax(dim=-1)
            else:
                probs   = torch.softmax(step_logits / temperature, dim=-1)
                pred_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pred_seq[0, res_idx] = pred_id

    # ── Results ────────────────────────────────────────────────────────────────
    pred_ids = pred_seq[0]
    matches  = ((pred_ids == native_classes) & residue_mask & design_mask).sum().item()
    total    = (residue_mask & design_mask).sum().item()
    nsr      = (matches / total * 100.0) if total > 0 else 0.0

    print(f"\n{'='*50}")
    print("INFERENCE RESULTS")
    print(f"{'='*50}")
    print(f"Native Sequence Recovery (NSR): {nsr:.2f}% ({matches}/{total} designed residues)")
    print(f"{'='*50}\n")

    sequence = ids_to_sequence(pred_ids[residue_mask].tolist())
    out_dir  = os.path.dirname(args.out_fasta)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_fasta(args.out_fasta, header=os.path.basename(args.pdb), sequence=sequence)
    logger.info("Wrote sequence to %s", args.out_fasta)


if __name__ == "__main__":
    main()
