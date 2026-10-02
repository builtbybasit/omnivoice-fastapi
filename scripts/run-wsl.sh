#!/usr/bin/env bash
# Set up and start the API in WSL2 on an NVIDIA GPU, with FlashInfer (Linux-only; about 2x faster).
#
#   bash scripts/run-wsl.sh               set up whatever is missing, then start the server
#   bash scripts/run-wsl.sh --setup-only  set up without starting: packages, FlashInfer, model
#
# Steps that are already done are skipped, so after the first run this starts in seconds.
# HOST and PORT choose where the server listens (default 127.0.0.1:8000). A FlashInfer JIT-cache
# wheel downloaded some other way can be used with FLASHINFER_JIT_CACHE_WHEEL=/path/to/wheel.
set -euo pipefail

setup_only=false
case "${1:-}" in
  "") ;;
  --setup-only) setup_only=true ;;
  *) echo "Usage: bash scripts/run-wsl.sh [--setup-only]" >&2; exit 2 ;;
esac

fail() { echo "$*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ -n "${WSL_DISTRO_NAME:-}" ]] ||
  fail "Run this inside a WSL2 Linux distribution: FlashInfer, which roughly doubles generation speed, is Linux-only."
case "$project_root" in
  /mnt/*) fail "Clone the project under your Linux home directory (e.g. ~/projects/omnivoice-fastapi), not /mnt/c; models and packages load much more slowly from there." ;;
esac
command -v uv >/dev/null || fail "Install uv in WSL first: curl -LsSf https://astral.sh/uv/install.sh | sh"

step "Checking the NVIDIA driver"
# --- NVIDIA driver. WSL gets CUDA from the Windows driver; do not install a Linux driver in WSL.
[[ -x /usr/lib/wsl/lib/nvidia-smi ]] && PATH="/usr/lib/wsl/lib:$PATH"
command -v nvidia-smi >/dev/null ||
  fail "WSL cannot find nvidia-smi: install or update the NVIDIA driver on Windows, then run 'wsl --update' in PowerShell."
cuda_version="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -n 1)"
[[ -n "$cuda_version" ]] || fail "nvidia-smi reported no CUDA version; is the GPU visible to WSL?"
if [[ "$(printf '%s\n' 12.8 "$cuda_version" | sort -V | head -n 1)" != 12.8 ]]; then
  fail "The Windows NVIDIA driver supports CUDA $cuda_version, but PyTorch here needs 12.8 or newer (driver R570+). Update the driver on Windows."
fi

cd "$project_root"
[[ -f .env ]] || cp .env.example .env

# --- Python packages, flashinfer-python included. --inexact keeps the JIT cache, which uv.lock
# does not list; a plain `uv sync` removes it, and the next step puts it back from the local wheel.
step "Installing Python packages (the first run downloads several GB)"
uv sync --extra cuda --inexact

# --- FlashInfer JIT cache: precompiled kernels, a 1.3 GB wheel. Downloads through pip/uv stalled,
# so it is fetched with curl (resumable, restarts stalled transfers) and checked before installing.
flashinfer_version='0.6.15.post1'
flashinfer_wheel_name="flashinfer_jit_cache-${flashinfer_version}+cu128-cp39-abi3-manylinux_2_28_x86_64.whl"
flashinfer_wheel_sha256='1af314fdf5a879b187a3e85c48c71a1f32c4d982ce520666b0d2795060b3e62b'
flashinfer_wheel_url="https://github.com/flashinfer-ai/flashinfer/releases/download/v${flashinfer_version}/${flashinfer_wheel_name/+/%2B}"
flashinfer_wheel_cache="$HOME/.cache/omnivoice-fastapi/wheels"

verify_flashinfer_wheel() {
  [[ -f "$1" ]] && printf '%s  %s\n' "$flashinfer_wheel_sha256" "$1" | sha256sum --check --status
}

download_flashinfer_wheel() {
  local target="$1" partial="$1.part"
  command -v curl >/dev/null || fail "Install curl in WSL (sudo apt install curl) to download the FlashInfer wheel."
  echo "Downloading $flashinfer_wheel_name (1.3 GB); rerunning this script resumes an interrupted download."
  # --speed-limit/--speed-time abandon a stalled transfer; --retry with --continue-at resumes it.
  curl --fail --location --continue-at - \
    --retry 30 --retry-all-errors --retry-delay 10 --retry-max-time 14400 \
    --speed-limit 10240 --speed-time 60 --connect-timeout 60 --progress-bar \
    --output "$partial" "$flashinfer_wheel_url"
  if ! verify_flashinfer_wheel "$partial"; then
    rm -f "$partial"
    fail "The downloaded FlashInfer wheel failed SHA-256 verification and was deleted; rerun to download it again."
  fi
  mv "$partial" "$target"
}

flashinfer_wheel() {
  local wheel="${FLASHINFER_JIT_CACHE_WHEEL:-$flashinfer_wheel_cache/$flashinfer_wheel_name}"
  if verify_flashinfer_wheel "$wheel"; then
    :
  elif [[ -n "${FLASHINFER_JIT_CACHE_WHEEL:-}" ]]; then
    fail "FLASHINFER_JIT_CACHE_WHEEL is missing or fails SHA-256 verification: $wheel"
  elif verify_flashinfer_wheel "$HOME/omnivoice-test/$flashinfer_wheel_name"; then
    wheel="$HOME/omnivoice-test/$flashinfer_wheel_name"
  else
    mkdir -p "$flashinfer_wheel_cache"
    download_flashinfer_wheel "$wheel" >&2
  fi
  echo "$wheel"
}

step "Checking the FlashInfer JIT cache"
installed_jit_cache="$(uv pip show flashinfer-jit-cache 2>/dev/null | sed -n 's/^Version: //p')"
if [[ "$installed_jit_cache" != "$flashinfer_version"* ]]; then
  wheel="$(flashinfer_wheel)"
  echo "Installing FlashInfer JIT cache from $wheel"
  uv pip install --no-deps "$wheel"
fi

# --- The model (3.3 GB on first run). Downloaded here so a stall shows in this terminal and
# resumes on the next run, instead of leaving a started server stuck at "loading".
step "Checking the model"
export OMNIVOICE_BACKEND=torch
uv run --no-sync python -m omnivoice_api.download

if [[ "$setup_only" == true ]]; then
  echo "Setup complete. Start the API with: bash scripts/run-wsl.sh"
  exit 0
fi

step "Starting the server: loading the model takes a minute; it is ready when 'Uvicorn running on' appears"
# FlashInfer is required: the server refuses to start without it rather than run 2x slower.
# Set OMNIVOICE_ENABLE_FLASHINFER=false in the environment to run the baseline path on purpose.
exec env OMNIVOICE_ENABLE_FLASHINFER="${OMNIVOICE_ENABLE_FLASHINFER:-true}" \
  uv run --no-sync uvicorn main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
