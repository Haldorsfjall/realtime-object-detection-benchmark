#!/usr/bin/env bash
set -euo pipefail

say() { echo -e "\033[1;36m[check]\033[0m $*"; }
warn() { echo -e "\033[1;33m[warn]\033[0m $*" >&2; }
err() { echo -e "\033[1;31m[err ]\033[0m $*" >&2; }

say "OS:"
cat /etc/os-release | sed -n '1,6p' || true
echo

say "Disk space:"
df -h / | tail -n 1 || true
echo

say "Python:"
if command -v python3 >/dev/null 2>&1; then
  python3 --version
else
  err "python3 not found. Install Python 3."
fi
echo

say "NVIDIA driver (nvidia-smi):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  warn "nvidia-smi not found. If you want GPU, install NVIDIA driver."
fi
echo

say "ffmpeg:"
if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -version | head -n 1
else
  warn "ffmpeg not found. Video decode may fail. Install ffmpeg."
fi
echo

if [[ -d ".venv" ]]; then
  say "Virtualenv detected: .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  say "Pip:"
  python -m pip --version || true
  say "Torch:"
  python - <<'PY' || true
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
else
  warn "No .venv found yet (this is OK). Run scripts/setup_venv.sh"
fi

say "Done."
