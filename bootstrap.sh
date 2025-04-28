#!/bin/bash

set -euo pipefail

# ─── debug / logging ────────────────────────────────────────
# Append all stdout/stderr to /var/log/bootstrap.log (must be writable)
exec >> >(tee -a /var/log/bootstrap.log) 2>&1
set -x
trap 'echo "❌ Error on line $LINENO (exit code $?)"; exit 1' ERR

# ────────────────────────────────────────────────────────────────
# PREP: install system tools (git, venv & pip)
# ────────────────────────────────────────────────────────────────
apt update && apt install -y --no-install-recommends \
    git \
    python3-venv \
    python3-pip \
    micro
rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────
# 1) Clone your repo
# ────────────────────────────────────────────────────────────────
git clone https://github.com/goytoom/cloud_agents.git
cd cloud_agents

# ────────────────────────────────────────────────────────────────
# 2) Create & activate the `agents` venv
# ────────────────────────────────────────────────────────────────
python3 -m venv ../agents
# If running under bash, source to keep the venv active in this shell
# (scripts run in a fresh shell by default)
source ../agents/bin/activate

# Upgrade pip
pip install --upgrade pip

# ────────────────────────────────────────────────────────────────
# 3) Install your Python deps
# ────────────────────────────────────────────────────────────────
pip install -r requirements.txt
pip install awscli huggingface-hub
pip install --no-build-isolation flash-attn

# ────────────────────────────────────────────────────────────────
# 4) Download your LLM from Hugging Face
# ────────────────────────────────────────────────────────────────
export MODEL_DIR="../models/Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2"
mkdir -p "$MODEL_DIR"
huggingface-cli download \
  matatonic/Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2 \
  --repo-type model \
  --local-dir "$MODEL_DIR" \
  || { echo "❌ HF download failed (exit $?)"; exit 1; }

echo "✅ bootstrap complete!"
