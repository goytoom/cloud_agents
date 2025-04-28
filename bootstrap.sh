#!/usr/bin/env bash
set -euo pipefail

# ── 1) parameters ────────────────────────────────────────────────────────────
VENV_DIR="agents"                                  # where to install your venv
MODEL_ID="avsolatorio/GIST-small-Embedding-v0"     # huggingface model to fetch
MODEL_CACHE="$HOME/workspace/models"                         # where to snapshot_download
EXTRA_PKGS=(flash-attn)                            # any pkgs not in requirements.txt
REQUIREMENTS="requirements.txt"

# ── 2) make sure we have Python & git ───────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 not found"; exit 1; }
command -v git     >/dev/null 2>&1 || { echo "❌ git     not found"; exit 1; }

# ── 3) create & activate venv ───────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "🔧 creating virtualenv in ./$VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 4) upgrade pip, install base deps ───────────────────────────────────────
echo "📦 upgrading pip & installing from $REQUIREMENTS"
pip install --upgrade pip setuptools wheel
pip install -r "$REQUIREMENTS"

# ── 5) install extra packages in the right order ───────────────────────────
if [ "${#EXTRA_PKGS[@]}" -gt 0 ]; then
  echo "➕ installing extra pip packages: ${EXTRA_PKGS[*]}"
  pip install "${EXTRA_PKGS[@]}"
fi

# ── 6) install CLI tools for downstream steps ───────────────────────────────
echo "🔧 installing awscli & huggingface_hub"
pip install --upgrade awscli huggingface_hub

# ── 7) fetch your LLM embedding model once ──────────────────────────────────
echo "📥 snapshot-downloading HuggingFace model $MODEL_ID → $MODEL_CACHE"
python3 - <<EOF
from huggingface_hub import snapshot_download
import os
os.makedirs("$MODEL_CACHE", exist_ok=True)
snapshot_download(
    repo_id="$MODEL_ID",
    cache_dir="$MODEL_CACHE",
    resume_download=True
)
EOF

echo "✅ bootstrap complete!  Activate with  source $VENV_DIR/bin/activate"
