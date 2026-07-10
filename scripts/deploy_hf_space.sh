#!/usr/bin/env bash
# Sync this repo's serving code into the Hugging Face Space and (optionally) push.
#
# The Space (WSobo/uma-inverse, Docker SDK) is a *projection* of this repo: it
# needs only src/, configs/, Dockerfile, pyproject.toml, .dockerignore, and an
# HF-flavoured README (deploy/hf_space/README.md — it carries the Space
# front-matter). This repo is the source of truth; the Space is a deploy target.
#
# Usage:
#   scripts/deploy_hf_space.sh [SPACE_DIR] [--no-push] [--message "msg"]
#
#   SPACE_DIR    Path to a local clone of the Space (default: ../uma-inverse-space).
#                If it does not exist, it is cloned from the Space git remote.
#   --no-push    Stage and commit in the clone but do not push (review first).
#   --message    Commit message (default: "deploy: sync serving code from <sha>").
#
# Auth: pushing to HF needs write access. Either run `huggingface-cli login`
# first, or let git prompt for your HF username + a write token as the password.
set -euo pipefail

# ── Resolve repo root (this script lives in scripts/) ──────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SPACE_URL="https://huggingface.co/spaces/WSobo/uma-inverse"
SPACE_DIR="../uma-inverse-space"
PUSH=1
MESSAGE=""

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-push) PUSH=0; shift ;;
    --message) MESSAGE="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *) SPACE_DIR="$1"; shift ;;
  esac
done

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
: "${MESSAGE:=deploy: sync serving code from ${SHA}}"

# ── Ensure the Space clone exists ──────────────────────────────────────────────
if [[ ! -d "$SPACE_DIR/.git" ]]; then
  echo "→ Cloning Space into $SPACE_DIR"
  git clone "$SPACE_URL" "$SPACE_DIR"
fi
SPACE_DIR="$(cd "$SPACE_DIR" && pwd)"
echo "→ Space clone: $SPACE_DIR"

# ── Project the serving file set into the clone ────────────────────────────────
# --delete keeps src/ and configs/ from accumulating files removed upstream.
RSYNC_EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache')

echo "→ Syncing src/ and configs/ (with --delete)"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" src/     "$SPACE_DIR/src/"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" configs/ "$SPACE_DIR/configs/"

echo "→ Copying Dockerfile, pyproject.toml, .dockerignore"
cp Dockerfile        "$SPACE_DIR/Dockerfile"
cp pyproject.toml    "$SPACE_DIR/pyproject.toml"
cp .dockerignore     "$SPACE_DIR/.dockerignore"

echo "→ Copying HF README (deploy/hf_space/README.md → README.md)"
cp deploy/hf_space/README.md "$SPACE_DIR/README.md"

# ── Commit & (optionally) push ─────────────────────────────────────────────────
cd "$SPACE_DIR"
git add -A

if git diff --cached --quiet; then
  echo "✓ Space already up to date — nothing to commit."
  exit 0
fi

echo
echo "── Changes to deploy ──────────────────────────────────────────"
git diff --cached --stat
echo "───────────────────────────────────────────────────────────────"

git commit -m "$MESSAGE"

if [[ "$PUSH" -eq 1 ]]; then
  echo "→ Pushing to $SPACE_URL (HF will rebuild the Docker image)"
  git push
  echo "✓ Pushed. Watch the build at ${SPACE_URL}?logs=build"
else
  echo "✓ Committed in $SPACE_DIR (not pushed). Push with: git -C '$SPACE_DIR' push"
fi
