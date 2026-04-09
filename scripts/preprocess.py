import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.ligandmpnn_bridge import load_example_from_pdb, load_json_ids, resolve_pdb_path

def process_pdb(pdb_id, pdb_dir, target_dir, max_total_nodes=1024, ligand_context_atoms=25, cutoff_for_score=8.0):
    try:
        path = resolve_pdb_path(pdb_dir, pdb_id)
        if path is None:
            return pdb_id, False
            
        out_path = os.path.join(target_dir, f"{pdb_id}.pt")
        if os.path.exists(out_path):
            return pdb_id, True

        example = load_example_from_pdb(
            pdb_path=path,
            ligand_context_atoms=ligand_context_atoms,
            cutoff_for_score=cutoff_for_score,
            max_total_nodes=max_total_nodes,
        )
        torch.save(example, out_path)
        return pdb_id, True
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return pdb_id, False

def main():
    parser = argparse.ArgumentParser(description="Preprocess PDBs for UMA-Inverse")
    parser.add_argument("--json_train", type=str, default="LigandMPNN/training/train.json")
    parser.add_argument("--json_valid", type=str, default="LigandMPNN/training/valid.json")
    parser.add_argument("--pdb_dir", type=str, default="data/raw/pdb_archive")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Fetching IDs from JSONs...")
    all_ids = set()
    for json_path in [args.json_train, args.json_valid]:
        full_path = os.path.join(PROJECT_ROOT, json_path)
        if os.path.exists(full_path):
            ids = load_json_ids(full_path)
            all_ids.update(ids)

    all_ids = list(all_ids)
    print(f"Discovered {len(all_ids)} sequence IDs for preprocessing")

    # Fast CPU-bound process pool
    success = 0
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(
                process_pdb, 
                pid, 
                os.path.join(PROJECT_ROOT, args.pdb_dir), 
                os.path.join(PROJECT_ROOT, args.out_dir)
            ) for pid in all_ids
        ]
        
        for i, future in enumerate(as_completed(futures)):
            pid, ok = future.result()
            if ok:
                success += 1
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(all_ids)}...")

    print(f"Done! Evaluated features successfully for {success}/{len(all_ids)} graphs and cached to {args.out_dir}.")

if __name__ == "__main__":
    main()
