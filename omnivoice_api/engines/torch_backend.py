"""k2-fsa/OmniVoice on PyTorch: CUDA (optionally with FlashInfer) on Linux/WSL, CPU elsewhere."""

from __future__ import annotations

import gc
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import Settings
from . import Line, Option, sampling_options

LOG = logging.getLogger("omnivoice_api")


def _device(configured: str) -> str:
    if configured and configured.lower() != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _running_in_wsl() -> bool:
    if sys.platform != "linux":
        return False
    if os.getenv("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


class TorchEngine:
    prompt_suffix = ".pt"  # upstream's VoiceClonePrompt file

    def __init__(self, settings: Settings):
        from omnivoice import OmniVoice

        device = _device(settings.device)
        flashinfer = settings.enable_flashinfer == "true" or (
            settings.enable_flashinfer == "auto" and _running_in_wsl() and device.startswith("cuda")
        )
        if flashinfer and (sys.platform != "linux" or not device.startswith("cuda")):
            raise RuntimeError(
                "FlashInfer needs the Linux CUDA runtime. Run inside WSL2 with an NVIDIA GPU, "
                "or set OMNIVOICE_ENABLE_FLASHINFER=false."
            )
        dtype = torch.float32 if device == "cpu" else torch.float16
        LOG.info("Loading %s on %s (%s)", settings.resolved_model, device, dtype)
        # Whisper, loaded to transcribe a clone recording that has no transcript, stays loaded:
        # on the CPU it costs RAM instead of 1.6 GB of the GPU memory batches need.
        model = OmniVoice.from_pretrained(
            settings.resolved_model, device_map=device, dtype=dtype, asr_device="cpu"
        )
        if flashinfer:
            try:
                from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
            except ImportError as exc:
                raise RuntimeError(
                    "FlashInfer is enabled but not installed; run scripts/setup-wsl-flashinfer.sh."
                ) from exc
            model = apply_flashinfer(model)
            LOG.info("FlashInfer acceleration enabled")
        self.model = model
        self.sample_rate = int(model.sampling_rate)
        self.options: dict[str, Option] = {
            **sampling_options(settings.default_num_steps),
            "denoise": Option("boolean", True, "Add the denoise token to clone prompts"),
            "postprocess_output": Option("boolean", True, "Trim silence from the output"),
        }

    def generate(self, lines: Sequence[Line], options: Mapping[str, Any]) -> list[np.ndarray]:
        kwargs: dict[str, Any] = {
            "text": [line.text for line in lines],
            "language": [line.language for line in lines],
            "instruct": [line.instruct for line in lines],
            "speed": [line.speed for line in lines],
            **options,
        }
        if lines[0].prompt is not None:
            kwargs["voice_clone_prompt"] = [line.prompt for line in lines]
        try:
            with torch.inference_mode():
                audios = self.model.generate(**kwargs)
        except Exception as exc:
            # The traceback holds the failed call's frames and their GPU tensors. Keep only the
            # message, so they are freed before the cache is emptied and the caller retries.
            error = f"{type(exc).__name__}: {exc}"
        else:
            return [np.asarray(audio, dtype=np.float32).reshape(-1) for audio in audios]
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(error)

    def encode_prompt(self, recording: Path, transcript: str | None) -> tuple[Any, str]:
        # With no transcript, upstream loads Whisper and transcribes the trimmed recording.
        prompt = self.model.create_voice_clone_prompt(ref_audio=str(recording), ref_text=transcript)
        return prompt, prompt.ref_text

    def save_prompt(self, prompt: Any, path: Path, source: str) -> None:
        """Upstream's ``VoiceClonePrompt.save()`` dict plus ``source``, which upstream's
        ``VoiceClonePrompt.load()`` ignores, so the file stays loadable there."""
        temp = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "format_version": 1,
                "ref_audio_tokens": prompt.ref_audio_tokens.detach().cpu(),
                "ref_text": prompt.ref_text,
                "ref_rms": float(prompt.ref_rms),
                "source": source,
            },
            temp,
        )
        temp.replace(path)

    def load_prompt(self, path: Path) -> tuple[Any, str | None]:
        from omnivoice import VoiceClonePrompt

        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError(f"{path.name} is not a VoiceClonePrompt (format version 1)")
        prompt = VoiceClonePrompt(data["ref_audio_tokens"], data["ref_text"], data["ref_rms"])
        return prompt, data.get("source")
