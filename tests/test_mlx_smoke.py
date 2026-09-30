"""The real MLX model end to end; slow, Apple Silicon only: OMNIVOICE_TEST_MLX=1 pytest -m mlx"""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import decode_wav, lines, wav_bytes
from fastapi.testclient import TestClient

from omnivoice_api.app import create_app
from omnivoice_api.config import Settings

pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(not os.getenv("OMNIVOICE_TEST_MLX"), reason="set OMNIVOICE_TEST_MLX=1"),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(
        _env_file=None,
        backend="mlx",
        default_num_steps=16,
        voices_dir=tmp_path_factory.mktemp("voices"),
        prompts_dir=tmp_path_factory.mktemp("prompts"),
    )
    with TestClient(create_app(settings)) as client:
        yield client


def test_designed_voice_batch(client):
    response = client.post(
        "/v1/audio/voices", data={"name": "Narrator", "description": "female, british accent"}
    )
    assert response.status_code == 201, response.text
    stream = lines(
        client.post(
            "/v1/audio/speech/batch",
            json={
                "model": "omnivoice",
                "items": [
                    {
                        "id": "a",
                        "input": "We are short again.",
                        "voice": "narrator",
                        "instructions": "Tired, flat, whisper",
                    },
                    {
                        "id": "b",
                        "input": "Then we count it twice.",
                        "voice": "narrator",
                        "speed": 1.5,
                        "language": "en",
                    },
                ],
            },
        )
    )
    assert stream[-1]["items"] == {"done": 2, "failed": 0}, stream
    for line in stream:
        if line["type"] == "item":
            samples, rate = decode_wav(line)
            assert rate == 24_000 and 0.5 < len(samples) / rate < 10
            assert np.sqrt(np.mean(samples**2)) > 0.01


def test_cloned_voice(client):
    # A tone is a poor voice, but it exercises the whole clone path: tokenize, cache, render.
    response = client.post(
        "/v1/audio/voices",
        data={"name": "Tone", "transcript": "Hello there."},
        files={"samples": ("tone.wav", wav_bytes(3.0), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    stream = lines(
        client.post(
            "/v1/audio/speech/batch",
            json={
                "model": "omnivoice",
                "items": [{"id": "a", "input": "Testing the cloned voice.", "voice": "tone"}],
            },
        )
    )
    assert stream[-1]["items"] == {"done": 1, "failed": 0}, stream
