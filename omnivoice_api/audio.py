"""Decoding uploaded recordings and encoding rendered audio.

All of it blocks: run it off the event loop.
"""

from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import soundfile as sf

CONTENT_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg; codecs=opus",
}


def available_formats() -> list[str]:
    formats = ["wav", "flac", "pcm"]
    if shutil.which("ffmpeg"):
        formats += ["mp3", "opus"]
    return formats


def _read(data: bytes) -> tuple[np.ndarray, int]:
    samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return samples.mean(axis=1).astype(np.float32), int(rate)


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float32 samples and their rate; ffmpeg handles what libsndfile cannot (m4a, …)."""
    try:
        return _read(data)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise
        wav = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
            input=data,
            capture_output=True,
            check=True,
        ).stdout
        return _read(wav)


def encode(audio: np.ndarray, audio_format: str, sample_rate: int) -> bytes:
    audio = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if audio_format == "pcm":
        return (audio * 32767).astype("<i2").tobytes()
    if audio_format in ("wav", "flac"):
        output = io.BytesIO()
        sf.write(output, audio, sample_rate, format=audio_format.upper(), subtype="PCM_16")
        return output.getvalue()
    codec, container = ("libmp3lame", "mp3") if audio_format == "mp3" else ("libopus", "ogg")
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", container, "-acodec", codec, "pipe:1"],
        input=encode(audio, "wav", sample_rate),
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[:500] or "ffmpeg failed")
    return result.stdout
