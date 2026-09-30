"""A stand-in engine that renders a tone per line, for tests and for working on the API anywhere."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import Line, Option, sampling_options


class FakeEngine:
    sample_rate = 24_000
    prompt_suffix = ".fake.json"

    def __init__(self, default_num_steps: int = 32, delay: float = 0.0):
        self.options: dict[str, Option] = sampling_options(default_num_steps)
        self.delay = delay
        self.calls: list[tuple[list[Line], dict[str, Any]]] = []
        self.encodes = 0

    def generate(self, lines: Sequence[Line], options: Mapping[str, Any]) -> list[np.ndarray]:
        self.calls.append((list(lines), dict(options)))
        if self.delay:
            time.sleep(self.delay)
        audios = []
        for line in lines:
            seconds = max(0.2, 0.06 * len(line.text) / line.speed)
            t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
            pitch = 180 + (hash(line.prompt or line.instruct) % 200)
            audios.append((0.2 * np.sin(2 * np.pi * pitch * t)).astype(np.float32))
        return audios

    def encode_prompt(self, recording: Path, transcript: str | None) -> tuple[Any, str]:
        self.encodes += 1
        return f"prompt:{recording.stem}", transcript or "(transcribed by the fake engine)"

    def save_prompt(self, prompt: Any, path: Path, source: str) -> None:
        path.write_text(json.dumps({"prompt": prompt, "source": source}))

    def load_prompt(self, path: Path) -> tuple[Any, str | None]:
        data = json.loads(path.read_text())
        return data["prompt"], data.get("source")
