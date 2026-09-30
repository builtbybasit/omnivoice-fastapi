"""OpenAI-compatible single and batch speech API backed by OmniVoice."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from rich.logging import RichHandler
from starlette.concurrency import run_in_threadpool

def _load_local_env() -> None:
    """Load simple KEY=value entries from this project's .env without extra deps."""
    env_file = Path(__file__).with_name(".env")
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


_load_local_env()

LOG = logging.getLogger("omnivoice_api")
_log_level = os.getenv("OMNIVOICE_LOG_LEVEL", "INFO").upper()
if not LOG.handlers:
    LOG.addHandler(
        RichHandler(
            show_path=False,
            rich_tracebacks=True,
            markup=False,
        )
    )
LOG.setLevel(_log_level)
LOG.propagate = False
SAMPLE_RATE = 24_000
MODEL_ID = os.getenv("OMNIVOICE_API_MODEL", "omnivoice")
MODEL_CHECKPOINT = os.getenv("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
API_KEY = os.getenv("OMNIVOICE_API_KEY", "").strip()
VOICES_DIR = Path(os.getenv("OMNIVOICE_VOICES_DIR", "voices")).resolve()
MAX_BATCH_ITEMS = int(os.getenv("OMNIVOICE_MAX_BATCH_ITEMS", "16"))
MAX_BATCH_CHARS = int(os.getenv("OMNIVOICE_MAX_BATCH_CHARS", "12000"))
MAX_ITEM_CHARS = int(os.getenv("OMNIVOICE_MAX_ITEM_CHARS", "1500"))
DEFAULT_NUM_STEPS = int(os.getenv("OMNIVOICE_DEFAULT_NUM_STEPS", "32"))
PING_SECONDS = max(1, int(os.getenv("OMNIVOICE_PING_SECONDS", "10")))
FLASHINFER_SETTING = os.getenv("OMNIVOICE_ENABLE_FLASHINFER", "true").strip().lower()

_model: Any = None
_voice_lock = threading.RLock()
_generation_lock = threading.Lock()
_voice_cache: dict[str, Voice] = {}


@dataclass
class Voice:
    id: str
    name: str
    directory: Path
    prompt: Any = None
    instructions: str | None = None
    gender: str | None = None
    language: str | None = None
    description: str | None = None

    def as_public_dict(self) -> dict[str, str]:
        result = {"id": self.id, "name": self.name}
        for key in ("gender", "language", "description"):
            value = getattr(self, key)
            if value:
                result[key] = value
        return result


def _best_device() -> str:
    configured = os.getenv("OMNIVOICE_DEVICE", "auto").strip()
    if configured and configured.lower() != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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


def _flashinfer_enabled() -> bool:
    if FLASHINFER_SETTING in {"1", "true", "yes", "on"}:
        return True
    if FLASHINFER_SETTING in {"0", "false", "no", "off"}:
        return False
    if FLASHINFER_SETTING == "auto":
        return _running_in_wsl()
    raise RuntimeError(
        "OMNIVOICE_ENABLE_FLASHINFER must be auto, true, or false; "
        f"got {FLASHINFER_SETTING!r}"
    )


def _load_model() -> Any:
    from omnivoice import OmniVoice

    device = _best_device()
    enable_flashinfer = _flashinfer_enabled()
    if enable_flashinfer and (sys.platform != "linux" or not device.startswith("cuda")):
        raise RuntimeError(
            "FlashInfer requires the Linux CUDA runtime. Run this API inside WSL2 with "
            "NVIDIA GPU support, or set OMNIVOICE_ENABLE_FLASHINFER=false for a baseline run."
        )

    LOG.info("Loading OmniVoice checkpoint %s on %s", MODEL_CHECKPOINT, device)
    model = OmniVoice.from_pretrained(MODEL_CHECKPOINT, device_map=device, dtype=torch.float16)
    if enable_flashinfer:
        try:
            from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
        except ImportError as exc:
            raise RuntimeError(
                "FlashInfer is enabled but not installed in this WSL environment. "
                "Run scripts/setup-wsl-flashinfer.sh before starting the API."
            ) from exc
        model = apply_flashinfer(model)
        LOG.info("FlashInfer acceleration enabled")
    else:
        LOG.info("FlashInfer disabled; using the standard OmniVoice inference path")
    return model


def _safe_voice_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]
    return slug or "voice"


def _load_voice(voice_id: str) -> Voice | None:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", voice_id) or len(voice_id) > 48:
        return None
    with _voice_lock:
        cached = _voice_cache.get(voice_id)
        if cached is not None:
            return cached
    directory = VOICES_DIR / voice_id
    metadata_path = directory / "voice.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        prompt_path = directory / "voice.pt"
        prompt = None
        if prompt_path.is_file():
            from omnivoice import VoiceClonePrompt

            prompt = VoiceClonePrompt.load(str(prompt_path))
        voice = Voice(
            id=voice_id,
            name=str(metadata.get("name", voice_id)),
            directory=directory,
            prompt=prompt,
            instructions=metadata.get("instructions"),
            gender=metadata.get("gender"),
            language=metadata.get("language"),
            description=metadata.get("description"),
        )
        with _voice_lock:
            return _voice_cache.setdefault(voice_id, voice)
    except Exception:
        LOG.exception("Could not load voice %s", voice_id)
        return None


def _all_voices() -> list[Voice]:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    with _voice_lock:
        voices = [_load_voice(path.name) for path in sorted(VOICES_DIR.iterdir()) if path.is_dir()]
    return [voice for voice in voices if voice is not None]


def _voice_or_error(voice_id: str) -> Voice:
    voice = _load_voice(voice_id)
    if voice is None:
        raise ItemFailure("voice_not_found", f"No voice '{voice_id}' on this server", False)
    return voice


def _error_body(message: str, code: str, param: str | None = None) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": code,
            **({"param": param} if param else {}),
        }
    }


class ItemFailure(Exception):
    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


def _formats() -> list[str]:
    formats = ["wav", "flac", "pcm"]
    if shutil.which("ffmpeg"):
        formats.extend(["mp3", "opus"])
    return formats


def _capabilities() -> dict[str, Any]:
    return {
        "object": "speech.capabilities",
        "version": 1,
        "models": [
            {
                "id": MODEL_ID,
                "batch": {"max_items": MAX_BATCH_ITEMS, "max_input_chars": MAX_BATCH_CHARS},
                "max_item_chars": MAX_ITEM_CHARS,
                "response_formats": _formats(),
                "sample_rates": [SAMPLE_RATE],
                "instructions": True,
                "speed": {"min": 0.5, "max": 2.0},
                "languages": None,
                "tags": {"open": "[", "close": "]", "known": ["laughter", "sigh"]},
                "extra": {
                    "num_step": {"type": "integer", "default": DEFAULT_NUM_STEPS,
                                 "description": "Number of diffusion decoding steps"},
                    "guidance_scale": {"type": "number", "default": 2.0,
                                       "description": "Classifier-free guidance scale"},
                    "t_shift": {"type": "number", "default": 0.1,
                                "description": "Diffusion time shift"},
                    "denoise": {"type": "boolean", "default": True,
                                "description": "Add the denoise token to clone prompts"},
                    "postprocess_output": {"type": "boolean", "default": True,
                                           "description": "Remove output silence"},
                    "layer_penalty_factor": {"type": "number", "default": 5.0,
                                              "description": "Layer-wise sampling penalty"},
                    "position_temperature": {"type": "number", "default": 5.0,
                                             "description": "Position selection temperature"},
                    "class_temperature": {"type": "number", "default": 0.0,
                                          "description": "Class token temperature"},
                },
            }
        ],
    }


async def require_api_key(authorization: str | None = Header(default=None)) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail=_error_body("Invalid API key", "unauthorized"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    _model = await run_in_threadpool(_load_model)
    yield
    _model = None


app = FastAPI(title="OmniVoice Speech API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def api_http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    body = exc.detail if isinstance(exc.detail, dict) and "error" in exc.detail else {"detail": exc.detail}
    return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok" if _model is not None else "loading", "model": MODEL_ID}


@app.get("/v1/audio/speech/capabilities", dependencies=[Depends(require_api_key)])
async def capabilities() -> dict[str, Any]:
    return _capabilities()


@app.get("/v1/audio/voices", dependencies=[Depends(require_api_key)])
async def list_voices() -> dict[str, list[dict[str, str]]]:
    return {"voices": [voice.as_public_dict() for voice in _all_voices()]}


def _decode_upload(data: bytes, filename: str) -> tuple[np.ndarray, int]:
    try:
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        return np.asarray(samples, dtype=np.float32), int(rate)
    except Exception as first_error:
        if shutil.which("ffmpeg"):
            try:
                decoded = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                    input=data, capture_output=True, check=True,
                ).stdout
                samples, rate = sf.read(io.BytesIO(decoded), dtype="float32", always_2d=False)
                if samples.ndim == 2:
                    samples = samples.mean(axis=1)
                return np.asarray(samples, dtype=np.float32), int(rate)
            except Exception:
                pass
        raise HTTPException(400, detail=_error_body(f"Could not read audio file '{filename}'", "invalid_request")) from first_error


@app.post("/v1/audio/voices", status_code=201, dependencies=[Depends(require_api_key)])
async def create_voice(
    name: str = Form(...),
    samples: list[UploadFile] = File(default=[]),
    transcript: str | None = Form(default=None),
    description: str | None = Form(default=None),
) -> dict[str, Any]:
    name = name.strip()
    if not name:
        raise HTTPException(400, detail=_error_body("name must not be empty", "invalid_request", "name"))
    voice_id = _safe_voice_id(name)
    directory = VOICES_DIR / voice_id
    if directory.exists():
        raise HTTPException(409, detail=_error_body(f"Voice '{voice_id}' already exists", "voice_conflict"))
    if not samples and not (description and description.strip()):
        raise HTTPException(400, detail=_error_body("Provide samples or a description", "invalid_request"))
    if samples and not (transcript and transcript.strip()):
        raise HTTPException(400, detail=_error_body("transcript is required when samples are provided", "invalid_request", "transcript"))
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    try:
        metadata: dict[str, Any] = {"name": name}
        if description:
            metadata["description"] = description.strip()
        if samples:
            data = await samples[0].read(50 * 1024 * 1024 + 1)
            if len(data) > 50 * 1024 * 1024:
                raise HTTPException(413, detail=_error_body("A reference audio sample must be 50 MiB or smaller", "invalid_request", "samples"))
            audio, sample_rate = _decode_upload(data, samples[0].filename or "sample")
            temp_audio = directory / "reference.wav"
            sf.write(temp_audio, audio, sample_rate)

            def create_prompt() -> None:
                prompt = _model.create_voice_clone_prompt(ref_audio=str(temp_audio), ref_text=transcript)
                prompt.save(str(directory / "voice.pt"))

            await run_in_threadpool(create_prompt)
            metadata["source_sample"] = "reference.wav"
        elif description:
            metadata["instructions"] = description.strip()
        (directory / "voice.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        voice = _load_voice(voice_id)
        if voice is None:
            raise RuntimeError("Voice was saved but could not be reloaded")
        return voice.as_public_dict()
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(directory, ignore_errors=True)
        LOG.exception("Voice creation failed")
        raise HTTPException(422, detail=_error_body(f"Could not create voice: {exc}", "voice_creation_failed")) from exc


def _parse_batch(payload: Any) -> tuple[str, str, int, dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise HTTPException(400, detail=_error_body("Request body must be a JSON object", "invalid_request"))
    model = payload.get("model")
    if model != MODEL_ID:
        raise HTTPException(404, detail=_error_body(f"No model '{model}'", "model_not_found", "model"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise HTTPException(400, detail=_error_body("items must be a non-empty array", "invalid_request", "items"))
    if len(items) > MAX_BATCH_ITEMS:
        raise HTTPException(400, detail=_error_body(f"At most {MAX_BATCH_ITEMS} items are allowed", "too_many_items", "items"))
    response_format = payload.get("response_format", "wav")
    if response_format not in _formats():
        raise HTTPException(400, detail=_error_body(f"Unsupported response_format '{response_format}'", "unsupported", "response_format"))
    sample_rate = payload.get("sample_rate", SAMPLE_RATE)
    if sample_rate != SAMPLE_RATE:
        raise HTTPException(400, detail=_error_body(f"Only sample_rate {SAMPLE_RATE} is supported", "unsupported", "sample_rate"))
    defaults = payload.get("extra", {})
    if not isinstance(defaults, dict):
        raise HTTPException(400, detail=_error_body("extra must be an object", "invalid_request", "extra"))
    total_chars = 0
    ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise HTTPException(400, detail=_error_body(f"items[{index}] must be an object", "invalid_request", "items"))
        item_id, text, voice = item.get("id"), item.get("input"), item.get("voice")
        if not isinstance(item_id, str) or not 1 <= len(item_id) <= 200:
            raise HTTPException(400, detail=_error_body(f"items[{index}].id must be 1–200 characters", "invalid_request", "items"))
        if item_id in ids:
            raise HTTPException(400, detail=_error_body(f"Duplicate item id '{item_id}'", "invalid_request", "items"))
        ids.add(item_id)
        if not isinstance(text, str) or not isinstance(voice, str) or not voice:
            raise HTTPException(400, detail=_error_body(f"items[{index}] requires string input and voice fields", "invalid_request", "items"))
        total_chars += len(text)
        if "speed" in item and (not isinstance(item["speed"], (int, float)) or isinstance(item["speed"], bool) or not 0.5 <= item["speed"] <= 2.0):
            raise HTTPException(400, detail=_error_body(f"items[{index}].speed must be between 0.5 and 2.0", "invalid_request", "speed"))
        if "extra" in item and not isinstance(item["extra"], dict):
            raise HTTPException(400, detail=_error_body(f"items[{index}].extra must be an object", "invalid_request", "extra"))
        for field in ("instructions", "language"):
            if item.get(field) is not None and not isinstance(item[field], str):
                raise HTTPException(400, detail=_error_body(f"items[{index}].{field} must be a string", "invalid_request", field))
    if total_chars > MAX_BATCH_CHARS:
        raise HTTPException(400, detail=_error_body(f"Batch exceeds {MAX_BATCH_CHARS} input characters", "too_many_items", "items"))
    return model, response_format, sample_rate, defaults, items


def _gen_kwargs(extra: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "num_step", "guidance_scale", "t_shift", "denoise", "postprocess_output",
        "layer_penalty_factor", "position_temperature", "class_temperature",
    }
    kwargs = {key: value for key, value in extra.items() if key in allowed}
    kwargs.setdefault("num_step", DEFAULT_NUM_STEPS)
    return kwargs


def _render_group(group: list[dict[str, Any]], extra: dict[str, Any]) -> list[np.ndarray]:
    voices = [_voice_or_error(item["voice"]) for item in group]
    clone_mode = voices[0].prompt is not None
    if any((voice.prompt is not None) != clone_mode for voice in voices):
        raise RuntimeError("Internal batch grouping mixed clone and designed voices")
    kwargs: dict[str, Any] = {
        "text": [item["input"] for item in group],
        "language": [item.get("language") for item in group],
        "speed": [item.get("speed", 1.0) for item in group],
        "instruct": [item.get("instructions") or voice.instructions for item, voice in zip(group, voices)],
        **_gen_kwargs(extra),
    }
    if clone_mode:
        kwargs["voice_clone_prompt"] = [voice.prompt for voice in voices]
    with _generation_lock, torch.inference_mode():
        audios = _model.generate(**kwargs)
    return [np.asarray(audio, dtype=np.float32).reshape(-1) for audio in audios]


def _wav_bytes(audio: np.ndarray) -> bytes:
    output = io.BytesIO()
    sf.write(output, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _encode_audio(audio: np.ndarray, audio_format: str) -> tuple[bytes, str]:
    if audio_format == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        return (pcm * 32767).astype("<i2").tobytes(), "audio/pcm"
    if audio_format in ("wav", "flac"):
        output = io.BytesIO()
        sf.write(output, audio, SAMPLE_RATE, format=audio_format.upper(), subtype="PCM_16")
        content_type = "audio/wav" if audio_format == "wav" else "audio/flac"
        return output.getvalue(), content_type
    codec = "libmp3lame" if audio_format == "mp3" else "libopus"
    container = "mp3" if audio_format == "mp3" else "ogg"
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", container, "-acodec", codec, "pipe:1"],
        input=_wav_bytes(audio), capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[:500] or "ffmpeg encoding failed")
    return result.stdout, "audio/mpeg" if audio_format == "mp3" else "audio/ogg; codecs=opus"


def _item_result(item: dict[str, Any], index: int, audio_format: str, audio: np.ndarray) -> dict[str, Any]:
    raw, _ = _encode_audio(audio, audio_format)
    seconds = len(audio) / SAMPLE_RATE
    return {
        "type": "item", "id": item["id"], "index": index, "status": "done",
        "format": audio_format, "sample_rate": SAMPLE_RATE, "duration": seconds,
        "audio": base64.b64encode(raw).decode("ascii"),
        "usage": {"input_characters": len(item["input"]), "audio_seconds": seconds},
    }


def _failure(item: dict[str, Any], index: int, error: ItemFailure) -> dict[str, Any]:
    return {
        "type": "item", "id": item["id"], "index": index, "status": "failed",
        "error": {"code": error.code, "message": error.message, "retryable": error.retryable},
    }


@app.post("/v1/audio/speech/batch", dependencies=[Depends(require_api_key)])
async def speech_batch(request: Request) -> StreamingResponse:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(400, detail=_error_body("Request body is not valid JSON", "invalid_request")) from exc
    _, audio_format, _, defaults, items = _parse_batch(payload)
    if _model is None:
        raise HTTPException(503, detail=_error_body("Speech model is not ready", "overloaded"), headers={"Retry-After": "5"})

    groups: dict[tuple[str, str], list[tuple[int, dict[str, Any], dict[str, Any]]]] = {}
    for index, item in enumerate(items):
        extra = {**defaults, **(item.get("extra") or {})}
        voice = _load_voice(item["voice"])
        mode = "clone" if voice and voice.prompt is not None else "design"
        key = (mode, json.dumps(extra, sort_keys=True, separators=(",", ":")))
        groups.setdefault(key, []).append((index, item, extra))

    async def stream() -> AsyncIterator[bytes]:
        done = 0
        failed = 0
        chars = 0
        seconds = 0.0
        for entries in groups.values():
            group = [entry[1] for entry in entries]
            extra = entries[0][2]
            bad: dict[int, ItemFailure] = {}
            renderable: list[tuple[int, dict[str, Any]]] = []
            for index, item in enumerate(group):
                if not item["input"].strip():
                    bad[index] = ItemFailure("empty_input", "input must not be empty", False)
                elif len(item["input"]) > MAX_ITEM_CHARS:
                    bad[index] = ItemFailure("input_too_long", f"input exceeds {MAX_ITEM_CHARS} characters", False)
                else:
                    try:
                        _voice_or_error(item["voice"])
                        renderable.append((index, item))
                    except ItemFailure as exc:
                        bad[index] = exc
            for local_index, error in bad.items():
                original_index, item, _ = entries[local_index]
                failed += 1
                chars += len(item["input"])
                yield (json.dumps(_failure(item, original_index, error), ensure_ascii=False) + "\n").encode()
            if renderable:
                render_task = asyncio.create_task(run_in_threadpool(_render_group, [row[1] for row in renderable], extra))
                try:
                    while not render_task.done():
                        try:
                            audios = await asyncio.wait_for(asyncio.shield(render_task), timeout=PING_SECONDS)
                            break
                        except TimeoutError:
                            yield b'{"type":"ping"}\n'
                    else:
                        audios = await render_task
                except Exception as exc:
                    LOG.exception("Batch generation failed")
                    code = "out_of_memory" if "out of memory" in str(exc).lower() else "render_failed"
                    for local_index, original_item in renderable:
                        original_index, original_item, _ = entries[local_index]
                        failed += 1
                        chars += len(original_item["input"])
                        error = ItemFailure(code, str(exc)[:500] or "Speech generation failed", True)
                        yield (json.dumps(_failure(original_item, original_index, error), ensure_ascii=False) + "\n").encode()
                    continue
                if len(audios) != len(renderable):
                    LOG.error("OmniVoice returned %d audio results for %d inputs", len(audios), len(renderable))
                for (local_index, _), audio in zip(renderable, audios):
                    original_index, original_item, _ = entries[local_index]
                    try:
                        result = _item_result(original_item, original_index, audio_format, audio)
                        done += 1
                        chars += len(original_item["input"])
                        seconds += result["duration"]
                        yield (json.dumps(result, ensure_ascii=False) + "\n").encode()
                    except Exception as exc:
                        LOG.exception("Batch audio encoding failed for item %s", original_item["id"])
                        failed += 1
                        chars += len(original_item["input"])
                        error = ItemFailure("render_failed", f"Audio encoding failed: {exc}", True)
                        yield (json.dumps(_failure(original_item, original_index, error), ensure_ascii=False) + "\n").encode()
                if len(audios) < len(renderable):
                    for local_index, original_item in renderable[len(audios):]:
                        original_index, original_item, _ = entries[local_index]
                        failed += 1
                        chars += len(original_item["input"])
                        error = ItemFailure("render_failed", "The model returned no audio for this item", True)
                        yield (json.dumps(_failure(original_item, original_index, error), ensure_ascii=False) + "\n").encode()
        yield (json.dumps({"type": "done", "items": {"done": done, "failed": failed},
                           "usage": {"input_characters": chars, "audio_seconds": seconds}}) + "\n").encode()

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.post("/v1/audio/speech", dependencies=[Depends(require_api_key)])
async def speech(request: Request) -> Response:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(400, detail=_error_body("Request body is not valid JSON", "invalid_request")) from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, detail=_error_body("Request body must be a JSON object", "invalid_request"))
    if payload.get("model", MODEL_ID) != MODEL_ID:
        raise HTTPException(404, detail=_error_body(f"No model '{payload.get('model')}'", "model_not_found", "model"))
    text, voice_id = payload.get("input"), payload.get("voice")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, detail=_error_body("input must be a non-empty string", "invalid_request", "input"))
    if len(text) > MAX_ITEM_CHARS:
        raise HTTPException(400, detail=_error_body(f"input exceeds {MAX_ITEM_CHARS} characters", "input_too_long", "input"))
    if not isinstance(voice_id, str) or not voice_id:
        existing = _all_voices()
        if len(existing) != 1:
            raise HTTPException(400, detail=_error_body("voice is required unless exactly one voice exists", "invalid_request", "voice"))
        voice_id = existing[0].id
    voice = _load_voice(voice_id)
    if voice is None:
        raise HTTPException(400, detail=_error_body(f"No voice '{voice_id}' on this server", "voice_not_found", "voice"))
    audio_format = payload.get("response_format", "wav")
    if audio_format not in _formats():
        raise HTTPException(400, detail=_error_body(f"Unsupported response_format '{audio_format}'", "unsupported", "response_format"))
    speed = payload.get("speed", 1.0)
    if not isinstance(speed, (int, float)) or isinstance(speed, bool) or not 0.5 <= speed <= 2.0:
        raise HTTPException(400, detail=_error_body("speed must be between 0.5 and 2.0", "invalid_request", "speed"))
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        raise HTTPException(400, detail=_error_body("extra must be an object", "invalid_request", "extra"))
    for field in ("instructions", "language"):
        if payload.get(field) is not None and not isinstance(payload[field], str):
            raise HTTPException(400, detail=_error_body(f"{field} must be a string", "invalid_request", field))
    item = {"id": "single", "input": text, "voice": voice_id, "speed": speed,
            "language": payload.get("language"), "instructions": payload.get("instructions")}
    if _model is None:
        raise HTTPException(503, detail=_error_body("Speech model is not ready", "overloaded"), headers={"Retry-After": "5"})
    try:
        audios = await run_in_threadpool(_render_group, [item], extra)
        data, content_type = _encode_audio(audios[0], audio_format)
        return Response(content=data, media_type=content_type,
                        headers={"Content-Disposition": f"inline; filename=speech.{audio_format}"})
    except ItemFailure as exc:
        raise HTTPException(400, detail=_error_body(exc.message, exc.code, "voice")) from exc
    except Exception as exc:
        LOG.exception("Single speech generation failed")
        raise HTTPException(500, detail=_error_body("Speech generation failed", "render_failed")) from exc
