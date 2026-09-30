from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from omnivoice_api.app import create_app
from omnivoice_api.config import Settings
from omnivoice_api.engines.fake import FakeEngine


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        backend="fake",
        voices_dir=tmp_path / "voices",
        prompts_dir=tmp_path / "prompts",
        max_batch_items=4,
        max_batch_chars=200,
        max_item_chars=60,
        ping_seconds=0.05,
    )


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(settings, engine):
    with TestClient(create_app(settings, engine)) as client:
        yield client


def wav_bytes(seconds: float = 1.0, rate: int = 16_000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    output = io.BytesIO()
    sf.write(output, 0.3 * np.sin(2 * np.pi * 220 * t), rate, format="WAV")
    return output.getvalue()


def make_voice(client, name="Mara", transcript: str | None = "Hello there.", **form):
    files = {"samples": ("mara.wav", wav_bytes(), "audio/wav")} if transcript is not None else None
    data = {"name": name, **({"transcript": transcript} if transcript else {}), **form}
    response = client.post("/v1/audio/voices", data=data, files=files)
    assert response.status_code == 201, response.text
    return response.json()


def lines(response) -> list[dict]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in response.text.splitlines()]


def decode_wav(line: dict) -> tuple[np.ndarray, int]:
    return sf.read(io.BytesIO(base64.b64decode(line["audio"])))
