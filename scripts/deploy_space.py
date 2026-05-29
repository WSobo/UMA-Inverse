"""Deploy the UMA-Inverse serving layer to a Hugging Face Docker Space.

Uploads the Dockerfile, package source (`src/`), and `configs/`. The checkpoint
is NOT uploaded — it is fetched from the Hub (WSobo/Uma-Inverse-1) at container
startup. HF builds the Docker image server-side after upload.

Prerequisites: `hf auth login` with a write-scoped token. Run from the repo root:

    python scripts/deploy_space.py
    # override the target space id:
    UMA_SPACE_ID=youruser/uma-inverse python scripts/deploy_space.py
"""
from __future__ import annotations

import os

from huggingface_hub import HfApi

REPO_ID = os.environ.get("UMA_SPACE_ID", "WSobo/uma-inverse")


def main() -> None:
    api = HfApi()
    print(f"target space: {REPO_ID}")

    # 1) Create the public Docker Space (idempotent).
    api.create_repo(REPO_ID, repo_type="space", space_sdk="docker", exist_ok=True)

    # 2) Set the Space README to the front-matter version (sdk: docker, app_port).
    api.upload_file(
        path_or_fileobj="deploy/hf_space/README.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="deploy: Space README front-matter",
    )

    # 3) Upload only what the image build needs.
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=".",
        allow_patterns=[
            "Dockerfile",
            ".dockerignore",
            "pyproject.toml",
            "src/**",
            "configs/**",
        ],
        commit_message="deploy: UMA-Inverse CPU serving layer",
    )

    print(f"done -> https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
