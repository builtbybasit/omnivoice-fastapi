#!/usr/bin/env bash
# Start the API on Apple Silicon with the MLX backend. The first run downloads the model (1.6 GB);
# rerunning resumes an interrupted download. HOST and PORT default to 127.0.0.1:8000.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$(uname -s)/$(uname -m)" == "Darwin/arm64" ]] || { echo "This script is for Apple Silicon Macs." >&2; exit 1; }
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
[[ -f .env ]] || cp .env.example .env
uv sync --extra mlx
export OMNIVOICE_BACKEND=mlx
uv run --no-sync python -m omnivoice_api.download
exec uv run --no-sync uvicorn main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
