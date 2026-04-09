import argparse

from src.data.ligandmpnn_bridge import load_example_from_pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    args = parser.parse_args()

    example = load_example_from_pdb(args.pdb)
    print("Residues:", int(example["residue_mask"].sum()))
    print("Ligand atoms:", int(example["ligand_mask"].sum()))
