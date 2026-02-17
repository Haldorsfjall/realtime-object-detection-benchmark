#!/usr/bin/env bash
set -euo pipefail

# Creates .venv and installs dependencies.
# Strategy:
#  1) Try to install CUDA-enabled PyTorch (cu128) first.
#  2) If that fails, install CPU build of PyTorch.
#  3) Install the rest from requirements-base.txt
#
# Optional env vars:
#   PYTHON=python3
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128   (default)
#   FORCE_CPU=1  (skip CUDA attempt)

PYTHON="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FORCE_CPU="${FORCE_CPU:-0}"

say() { echo -e "\033[1;32m[setup]\033[0m $*"; }
warn() { echo -e "\033[1;33m[setup]\033[0m $*" >&2; }

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON not found." >&2
  exit 2
fi

if [[ ! -d ".venv" ]]; then
  say "Creating venv: .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

say "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

install_torch_cuda() {
  say "Installing PyTorch (CUDA) from: $TORCH_INDEX_URL"
  python -m pip install --no-cache-dir torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
}

install_torch_cpu() {
  say "Installing PyTorch (CPU)"
  python -m pip install --no-cache-dir torch torchvision torchaudio
}

if [[ "$FORCE_CPU" == "1" ]]; then
  warn "FORCE_CPU=1 set: skipping CUDA torch install."
  install_torch_cpu
else
  if install_torch_cuda; then
    say "CUDA torch installed."
  else
    warn "CUDA torch install failed. Falling back to CPU torch."
    install_torch_cpu
  fi
fi

say "Installing project deps (requirements-base.txt)"
python -m pip install --no-cache-dir -r requirements-base.txt

say "Writing environment manifest: runs/manifest.txt"
mkdir -p runs
{
  echo "== date =="
  date
  echo
  echo "== python =="
  python --version
  echo
  echo "== pip freeze =="
  python -m pip freeze
  echo
  echo "== torch cuda =="
  python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
} > runs/manifest.txt

say "Done. Activate with: source .venv/bin/activate"
