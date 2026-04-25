#!/usr/bin/env bash
# Entrypoint executed inside the HuggingFace Jobs A100 container.
#
# Required environment variables (provided by `hf jobs run --env / --secrets`):
#   REPO_URL         : git clone URL of the public CrisisOps repo
#   REPO_REF         : (optional) git ref to check out (default: main)
#   HF_OUTPUT_REPO   : HF Hub model repo to push artifacts to
#   HF_TOKEN         : (secret) HF write token
#   WANDB_API_KEY    : (optional secret) Weights & Biases API key
#
# Flow:
#   1. Clone the repo at REPO_REF.
#   2. Install Python dependencies (Unsloth, vLLM, TRL, etc.).
#   3. Install the local crisisops_env package.
#   4. Launch scripts/train_crisisops_grpo.py.
#
# Anything else is logged to stdout/stderr so `hf jobs logs <id>` can stream it.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Vk2245/CrisisOps-Multi-Agent-SRE-Training-via-OpenEnv.git}"
REPO_REF="${REPO_REF:-main}"
WORKDIR="/workspace/repo"

echo "[entrypoint] CUDA visible:"
nvidia-smi || true

echo "[entrypoint] Python:"
python --version
pip --version

echo "[entrypoint] Cloning ${REPO_URL} @ ${REPO_REF}"
rm -rf "${WORKDIR}"
git clone --depth=1 --branch "${REPO_REF}" "${REPO_URL}" "${WORKDIR}"
cd "${WORKDIR}"

echo "[entrypoint] Upgrading pip"
pip install -q -U pip wheel setuptools

echo "[entrypoint] Installing GRPO training stack (this can take 5-10 min)"
# Pin torch first so Unsloth picks the right wheels.
pip install -q -U \
    "unsloth[colab-new]" \
    "vllm>=0.6.0" \
    "trl>=0.13.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.44.0" \
    "datasets>=3.0.0" \
    "wandb" \
    "pandas" \
    "matplotlib" \
    "seaborn" \
    "huggingface_hub>=0.26.0"

echo "[entrypoint] Installing crisisops_env in editable mode"
pip install -q -e ./crisisops_env

echo "[entrypoint] Pip freeze (truncated):"
pip freeze | head -n 40 || true

echo "[entrypoint] Launching scripts/train_crisisops_grpo.py"
python scripts/train_crisisops_grpo.py

echo "[entrypoint] Done."
