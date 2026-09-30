#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "Run this script inside a WSL2 Linux distribution." >&2
  exit 1
fi

case "$project_root" in
  /mnt/*)
    echo "Move the project under the Linux home filesystem (for example ~/projects/omnivoice-fastapi)." >&2
    echo "Running model inference from /mnt/c is slower and can make dependency setup much slower." >&2
    exit 1
    ;;
esac

command -v uv >/dev/null || { echo "Install uv in WSL before continuing." >&2; exit 1; }
if ! command -v nvidia-smi >/dev/null && [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  export PATH="/usr/lib/wsl/lib:$PATH"
fi
command -v nvidia-smi >/dev/null || { echo "WSL cannot find nvidia-smi; install/update the Windows NVIDIA driver and WSL2." >&2; exit 1; }
nvidia-smi

cd "$project_root"
[[ -f .env ]] || cp .env.example .env
uv sync --no-install-project
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot see the WSL CUDA GPU"; print("CUDA device:", torch.cuda.get_device_name(0))'

flashinfer_wheel_name='flashinfer_jit_cache-0.6.15.post1+cu128-cp39-abi3-manylinux_2_28_x86_64.whl'
flashinfer_wheel_sha256='1af314fdf5a879b187a3e85c48c71a1f32c4d982ce520666b0d2795060b3e62b'
flashinfer_wheel_url="https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.15.post1/${flashinfer_wheel_name}"
flashinfer_wheel_cache="$HOME/.cache/omnivoice-fastapi/wheels"
flashinfer_wheel_path="$flashinfer_wheel_cache/$flashinfer_wheel_name"

verify_flashinfer_wheel() {
  local wheel_path="$1"
  [[ -f "$wheel_path" ]] || return 1
  printf '%s  %s\n' "$flashinfer_wheel_sha256" "$wheel_path" | sha256sum --check --status
}

install_flashinfer_from_index() {
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}" uv pip install \
    flashinfer-python==0.6.15.post1 \
    'flashinfer-jit-cache==0.6.15.post1+cu128' \
    --extra-index-url https://flashinfer.ai/whl/cu128/
}

if ! install_flashinfer_from_index; then
  echo "The FlashInfer index install did not complete; using a resumable local wheel fallback." >&2
  mkdir -p "$flashinfer_wheel_cache"

  if verify_flashinfer_wheel "$flashinfer_wheel_path"; then
    echo "Using verified cached wheel: $flashinfer_wheel_path"
  elif verify_flashinfer_wheel "$HOME/omnivoice-test/$flashinfer_wheel_name"; then
    flashinfer_wheel_path="$HOME/omnivoice-test/$flashinfer_wheel_name"
    echo "Using verified wheel from omnivoice-test: $flashinfer_wheel_path"
  else
    command -v curl >/dev/null || { echo "Install curl in WSL to download the FlashInfer wheel fallback." >&2; exit 1; }
    curl --fail --location --continue-at - \
      --retry 15 --retry-all-errors --retry-delay 10 --retry-max-time 7200 \
      --connect-timeout 60 --progress-bar \
      --output "$flashinfer_wheel_path" "$flashinfer_wheel_url"
  fi

  if ! verify_flashinfer_wheel "$flashinfer_wheel_path"; then
    echo "FlashInfer JIT cache wheel SHA-256 verification failed." >&2
    exit 1
  fi
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" uv pip install \
    flashinfer-python==0.6.15.post1 "$flashinfer_wheel_path" \
    --extra-index-url https://flashinfer.ai/whl/cu128/
fi

uv run --no-sync python -c 'import torch, flashinfer; from omnivoice.models.omnivoice_flashinfer import apply_flashinfer; assert torch.cuda.is_available(); print("FlashInfer and OmniVoice integration are importable")'
echo "Setup complete. Start the API with: bash scripts/run-wsl.sh"
