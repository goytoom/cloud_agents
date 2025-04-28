#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Helper: detect if we have a Blackwell GPU present
# ────────────────────────────────────────────────────────────────
detect_blackwell() {
  # must have nvidia-smi
  command -v nvidia-smi &>/dev/null || return 1
  # and its architecture must include “Blackwell”
  nvidia-smi -q | grep -iq "Blackwell"
}

IS_BLACKWELL=false
if detect_blackwell; then
  IS_BLACKWELL=true
  echo "⚙️  Blackwell GPU detected—will apply NCCL & PyTorch fixes"
fi

# ────────────────────────────────────────────────────────────────
# If Blackwell & not already CUDA 12.8, update system NCCL
# ────────────────────────────────────────────────────────────────
if $IS_BLACKWELL; then
  # skip if image is already CUDA 12.8+
  if [[ "${CUDA_VERSION:-}" != 12.8* ]]; then
    echo "→ Applying system-level NCCL update for Blackwell"
    # grab the installed NCCL version (or “0.0.0” if none)
    NCCL_VERSION=$(dpkg-query -W -f='${Version}' libnccl2 2>/dev/null \
                   | cut -d'-' -f1 || echo "0.0.0")
    if dpkg --compare-versions "$NCCL_VERSION" lt "2.26.2"; then
      apt-get update
      apt-get install -y --allow-change-held-packages \
        libnccl2=2.26.2-1+cuda12.8 \
        libnccl-dev=2.26.2-1+cuda12.8
      echo "   • NCCL upgraded to 2.26.2-1+cuda12.8"
    else
      echo "   • System NCCL already ≥2.26.2 (found $NCCL_VERSION)"
    fi
  else
    echo "   • CUDA 12.8 image—no system NCCL update needed"
  fi
  echo "✅ System-level NVIDIA/Blackwell fixes applied"
fi

# ────────────────────────────────────────────────────────────────
# PREP: make sure git, venv & pip exist
# ────────────────────────────────────────────────────────────────
apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-venv \
    python3-pip
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
source ../agents/bin/activate
pip install --upgrade pip

# ────────────────────────────────────────────────────────────────
# 3) Install your Python deps
# ────────────────────────────────────────────────────────────────
pip install -r requirements.txt
pip install flash-attn awscli

# ────────────────────────────────────────────────────────────────
# 4) Download your LLM from HF
# ────────────────────────────────────────────────────────────────
export MODEL_DIR="../models/Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2"
mkdir -p "$MODEL_DIR"
huggingface-cli download \
  matatonic/Mistral-Small-24B-Instruct-2501-6.5bpw-h8-exl2 \
  --repo-type model \
  --local-dir "$MODEL_DIR"

# ────────────────────────────────────────────────────────────────
# 5) Inside venv, apply PyTorch Blackwell fix if needed
# ────────────────────────────────────────────────────────────────
if $IS_BLACKWELL && python - <<<'import importlib; exit(0 if importlib.util.find_spec("torch") else 1)'; then
  echo "→ Applying PyTorch Blackwell fix in agents venv…"

  pip uninstall -y torch torchvision torchaudio xformers || true

  # try nightly first, fallback to stable if it fails
  if ! pip install --no-cache-dir --pre \
        torch==2.7.0.dev20250312+cu128 \
        torchvision==0.22.0.dev20250312+cu128 \
        torchaudio==2.6.0.dev20250312+cu128 \
        --upgrade-strategy only-if-needed \
        --index-url https://download.pytorch.org/whl/nightly/cu128; then
    pip install --no-cache-dir --pre \
        torch==2.7.0 torchvision torchaudio \
        --upgrade-strategy only-if-needed \
        --index-url https://download.pytorch.org/whl/cu128
  fi

  # copy system NCCL into the venv’s site-packages
  PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  DEST="$VIRTUAL_ENV/lib/python$PYVER/site-packages/nvidia/nccl/lib"
  mkdir -p "$DEST"
  cp /usr/lib/x86_64-linux-gnu/libnccl.so.2 "$DEST"

  echo "✅ PyTorch Blackwell fix applied"
fi

echo "✅ bootstrap complete!"
