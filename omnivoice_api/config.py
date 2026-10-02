"""Server settings, read from OMNIVOICE_* environment variables and the project's .env file."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {
    "torch": "k2-fsa/OmniVoice",
    "mlx": "mlx-community/OmniVoice-bf16",
    "fake": "fake",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OMNIVOICE_", env_file=PROJECT_ROOT / ".env", extra="ignore"
    )

    backend: Literal["auto", "torch", "mlx", "fake"] = "auto"
    model: str = ""
    api_model: str = "omnivoice"
    device: str = "auto"
    enable_flashinfer: Literal["auto", "true", "false"] = "auto"
    transcribe_model: str = "mlx-community/Qwen3-ASR-0.6B-8bit"
    mlx_cache_gb: float = Field(1, ge=0)  # freed MLX memory kept for reuse
    api_key: str = ""
    voices_dir: Path = Path("voices")
    prompts_dir: Path = Path("prompts")
    max_batch_items: int = Field(16, ge=1)
    engine_batch_size: int = Field(0, ge=0)  # lines per model call; 0 = a whole group
    max_batch_chars: int = Field(12000, ge=1)
    max_item_chars: int = Field(1500, ge=1)
    max_upload_mb: int = Field(50, ge=1)
    default_num_steps: int = Field(32, ge=1)
    ping_seconds: float = Field(10, gt=0)
    log_level: str = "INFO"

    @property
    def resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if sys.platform == "darwin" and platform.machine() == "arm64":
            return "mlx"
        return "torch"

    @property
    def resolved_model(self) -> str:
        return self.model or DEFAULT_MODELS[self.resolved_backend]
