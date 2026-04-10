"""Entry point — delegates to the Hydra training CLI.

Usage (via uv + SLURM):
    uv run python main.py                          # default config
    uv run python main.py ++data.max_total_nodes=64  # Hydra override
"""
from scripts.train import main

if __name__ == "__main__":
    main()
