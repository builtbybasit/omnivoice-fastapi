"""The seam between the HTTP API and a speech model.

An engine renders lines and makes voice-clone prompts; nothing else in the server imports torch or
MLX. ``torch`` runs k2-fsa/OmniVoice on CUDA (optionally with FlashInfer), ``mlx`` runs the
mlx-audio port on Apple Silicon, and ``fake`` renders tones so the API can be tested anywhere.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from ..config import Settings

T = TypeVar("T")

@dataclass(frozen=True)
class Line:
    text: str
    language: str | None
    instruct: str | None
    speed: float
    prompt: Any = None  # the engine's clone prompt; None renders a designed (or model-picked) voice


@dataclass(frozen=True)
class Option:
    type: str  # "integer", "number" or "boolean", as JSON names them
    default: Any
    description: str
    minimum: float | None = None
    maximum: float | None = None

    def public(self) -> dict[str, Any]:
        result = {"type": self.type, "default": self.default, "description": self.description}
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        return result

    def check(self, name: str, value: Any) -> Any:
        if self.type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"extra.{name} must be a boolean")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"extra.{name} must be a number")
        if self.type == "integer":
            if value != int(value):
                raise ValueError(f"extra.{name} must be a whole number")
            value = int(value)
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"extra.{name} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"extra.{name} must be at most {self.maximum}")
        return value


def sampling_options(default_num_steps: int) -> dict[str, Option]:
    """The generation options both OmniVoice backends take, under upstream's names."""
    return {
        "num_step": Option("integer", default_num_steps, "Diffusion decoding steps", 1, 200),
        "guidance_scale": Option("number", 2.0, "Classifier-free guidance scale", 0),
        "t_shift": Option("number", 0.1, "Diffusion time shift", 0),
        "layer_penalty_factor": Option("number", 5.0, "Layer-wise sampling penalty", 0),
        "position_temperature": Option("number", 5.0, "Position selection temperature", 0),
        "class_temperature": Option("number", 0.0, "Class token temperature", 0),
    }


def resolve_options(
    spec: Mapping[str, Option], *layers: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Defaults, then each layer of ``extra`` over them; unknown keys are ignored per the spec."""
    options = {name: option.default for name, option in spec.items()}
    for layer in layers:
        for name, value in (layer or {}).items():
            if name in spec:
                options[name] = spec[name].check(name, value)
    return options


class Engine(Protocol):
    sample_rate: int
    options: dict[str, Option]

    def generate(self, lines: Sequence[Line], options: Mapping[str, Any]) -> list[np.ndarray]:
        """One float32 array per line, in order. Lines share one kind: all cloned or none."""

    # A voice-clone prompt: the reference recording encoded once, then saved under the prompts
    # directory as ``<voice id><prompt_suffix>`` and reused until the recording or transcript
    # changes (``source`` fingerprints them).
    prompt_suffix: str

    def encode_prompt(self, recording: Path, transcript: str | None) -> tuple[Any, str]:
        """Encode a reference recording; returns the prompt and its transcript. With no
        transcript the engine transcribes the recording itself."""

    def save_prompt(self, prompt: Any, path: Path, source: str) -> None: ...

    def load_prompt(self, path: Path) -> tuple[Any, str | None]:
        """The saved prompt and the ``source`` it was saved with (None if it has none)."""


def load_engine(settings: Settings) -> Engine:
    backend = settings.resolved_backend
    if backend == "mlx":
        from .mlx_backend import MlxEngine

        return MlxEngine(settings)
    if backend == "torch":
        from .torch_backend import TorchEngine

        return TorchEngine(settings)
    from .fake import FakeEngine

    return FakeEngine(default_num_steps=settings.default_num_steps)


class ModelThread:
    """Runs every model call on one dedicated thread: serialises GPU work and keeps MLX's
    per-thread streams and CUDA state in one place."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")

    async def run(self, fn: Callable[..., T], *args: Any) -> T:
        return await asyncio.wrap_future(self._pool.submit(fn, *args))

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
