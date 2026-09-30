#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run this script inside WSL2. Native Windows startup requires the slower baseline path." >&2
  exit 1
fi
case "$project_root" in
  /mnt/*) echo "Run the API from a copy under your Linux home directory, not /mnt/c." >&2; exit 1 ;;
esac

cd "$project_root"
[[ -x .venv/bin/uvicorn ]] || { echo "Run scripts/setup-wsl-flashinfer.sh first." >&2; exit 1; }
uv run --no-sync python -c 'import torch, flashinfer; from omnivoice.models.omnivoice_flashinfer import apply_flashinfer; assert torch.cuda.is_available(), "WSL CUDA GPU is unavailable"'
exec env OMNIVOICE_ENABLE_FLASHINFER=true uv run --no-sync uvicorn main:app --log-config scripts/rich-logging.json --host 127.0.0.1 --port 8000
