#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# 1) clone the code
# ────────────────────────────────────────────────────────────────
# If your repo is PUBLIC you can omit credentials.
git clone https://github.com/goytoom/cloud_agents.git
cd cloud_agents

# ────────────────────────────────────────────────────────────────
# 2) create & activate a venv, install core deps
# ────────────────────────────────────────────────────────────────
python3 -m venv ../agents
source ../agents/bin/activate
pip install --upgrade pip

# install everything in requirements.txt
pip install -r requirements.txt

# ────────────────────────────────────────────────────────────────
# 3) install extra packages in order
# ────────────────────────────────────────────────────────────────
# flash-attn sometimes needs to come *after* your main deps:
pip install flash-attn
pip install awscli

# ────────────────────────────────────────────────────────────────
# 4) download your LLM with the HF CLI
# ────────────────────────────────────────────────────────────────
# requires that you have done: `pip install huggingface_hub`
# and have set HF_TOKEN in environment, or run `huggingface-cli login`
export MODEL_DIR="../models/Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2"
mkdir -p "$MODEL_DIR"
huggingface-cli download \
  matatonic/Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2 \
  --repo-type model \
  --local-dir "$MODEL_DIR"

echo "✅ bootstrap complete!"
