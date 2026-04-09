import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_pdb(pdb_id, out_dir):
    pdb_id = str(pdb_id).lower()
    
    # Use RCSB-style HPC-safe 2-character subdirectories (e.g. 1abc -> ab)
    sub_dir = pdb_id[1:3] if len(pdb_id) >= 4 else "misc"
    target_dir = os.path.join(out_dir, sub_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    out_path = os.path.join(target_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(out_path):
        return True
        
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, out_path)
            return True
        except Exception:
            time.sleep(2 * (attempt + 1))  # Exponential backoff
            
    print(f"Failed to fetch {pdb_id}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Download PDB files listed in LigandMPNN JSON splits")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of structures fetched (useful for pilot runs)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download threads")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(project_root, "../data/raw/pdb_archive")
    os.makedirs(out_dir, exist_ok=True)
    
    json_files = [
        os.path.join(project_root, "../../LigandMPNN/training/train.json"),
        os.path.join(project_root, "../../LigandMPNN/training/valid.json"),
        os.path.join(project_root, "../../LigandMPNN/training/test_small_molecule.json")
    ]
    
    pdb_ids = set()
    for json_file in json_files:
        if os.path.exists(json_file):
            print(f"Parsing IDs from: {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    pdb_ids.update(data)
                elif isinstance(data, dict):
                    pdb_ids.update(data.keys())
    
    pdb_ids = list(pdb_ids)
    if args.limit is not None:
        pdb_ids = pdb_ids[: args.limit]

    print(f"Discovered {len(pdb_ids)} unique PDB structures to download.")
    
    success = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_pdb, pid, out_dir) for pid in pdb_ids]
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                success += 1
                
    print(f"✅ Successfully prepared {success} out of {len(pdb_ids)} PDB files in {out_dir}/")

if __name__ == "__main__":
    main()
