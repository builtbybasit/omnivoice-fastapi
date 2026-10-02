"""The HTTP API: Audiobook Studio's speech batch API plus OpenAI's single speech route.

Spec: https://github.com/builtbybasit/audiobook-studio/blob/main/docs/speech-batch-api.md
"""

from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from . import audio, vocab
from .batch import Renderer
from .config import Settings
from .engines import Engine, ModelThread, load_engine, resolve_options
from .errors import ItemFailure, api_error, error_body, is_out_of_memory
from .schemas import BatchRequest, SpeechItem, SpeechRequest
from .voices import VoiceExists, VoiceStore

LOG = logging.getLogger("omnivoice_api")


def configure_logging(level: str) -> None:
    if not LOG.handlers:
        from rich.logging import RichHandler

        LOG.addHandler(RichHandler(show_path=False, rich_tracebacks=True, markup=False))
        LOG.propagate = False
    LOG.setLevel(level.upper())


class Server:
    """What the routes share. ``engine`` and ``renderer`` exist once the model has loaded."""

    def __init__(self, settings: Settings, engine: Engine | None):
        self.settings = settings
        self.engine = engine
        self.model = ModelThread()
        self.voices = VoiceStore(settings.voices_dir, settings.prompts_dir)
        self.formats = audio.available_formats()
        self.renderer: Renderer | None = None

    async def start(self) -> None:
        await run_in_threadpool(self.voices.migrate_folders)
        if self.engine is None:
            self.engine = await self.model.run(load_engine, self.settings)
        self.renderer = Renderer(
            self.engine,
            self.model,
            self.voices,
            self.settings.max_item_chars,
            self.settings.ping_seconds,
            self.settings.engine_batch_size or self.settings.max_batch_items,
        )

    def ready(self) -> Renderer:
        if self.renderer is None:
            raise api_error(
                503, "Speech model is not ready", "overloaded", headers={"Retry-After": "5"}
            )
        return self.renderer

    def capabilities(self) -> dict[str, Any]:
        settings = self.settings
        options = self.engine.options if self.engine else {}
        return {
            "object": "speech.capabilities",
            "version": 1,
            "models": [
                {
                    "id": settings.api_model,
                    "backend": settings.resolved_backend,
                    "batch": {
                        "max_items": settings.max_batch_items,
                        "max_input_chars": settings.max_batch_chars,
                    },
                    "max_item_chars": settings.max_item_chars,
                    "response_formats": self.formats,
                    "sample_rates": [self.engine.sample_rate if self.engine else 24_000],
                    "instructions": True,
                    "instruction_vocabulary": vocab.valid_instructions(),
                    "speed": {"min": 0.5, "max": 2.0},
                    "languages": None,
                    "tags": {"open": "[", "close": "]", "known": vocab.NONVERBAL_TAGS},
                    "extra": {name: option.public() for name, option in options.items()},
                }
            ],
        }


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    """The app; pass ``engine`` to skip loading a model (tests pass a FakeEngine)."""
    settings = settings or Settings()
    configure_logging(settings.log_level)
    server = Server(settings, engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await server.start()
        yield
        server.model.close()

    app = FastAPI(title="OmniVoice Speech API", version="2.0.0", lifespan=lifespan)
    app.state.server = server

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        body = detail if isinstance(detail, dict) and "error" in detail else {"detail": detail}
        return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        location = [part for part in first.get("loc", ()) if part != "body"]
        path = "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in location)
        param = str(location[0]) if location else None
        return JSONResponse(
            error_body(
                f"{path.lstrip('.') or 'body'}: {first.get('msg', 'invalid')}",
                "invalid_request",
                param,
            ),
            status_code=400,
        )

    async def require_api_key(authorization: str | None = Header(default=None)) -> None:
        if settings.api_key and not secrets.compare_digest(
            (authorization or "").encode(), f"Bearer {settings.api_key}".encode()
        ):
            raise api_error(401, "Invalid API key", "unauthorized")

    authorized = [Depends(require_api_key)]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok" if server.renderer else "loading",
            "model": settings.api_model,
            "backend": settings.resolved_backend,
        }

    @app.get("/v1/audio/speech/capabilities", dependencies=authorized)
    async def capabilities() -> dict[str, Any]:
        return server.capabilities()

    @app.get("/v1/audio/voices", dependencies=authorized)
    async def list_voices() -> dict[str, list[dict[str, str]]]:
        voices = await run_in_threadpool(server.voices.all)
        return {"voices": [voice.public() for voice in voices]}

    @app.post("/v1/audio/voices", status_code=201, dependencies=authorized)
    async def create_voice(
        name: str = Form(...),
        samples: list[UploadFile] = File(default=[]),
        transcript: str | None = Form(default=None),
        description: str | None = Form(default=None),
    ) -> dict[str, str]:
        renderer = server.ready()
        name, transcript, description = (
            name.strip(),
            (transcript or "").strip() or None,
            (description or "").strip() or None,
        )
        if not name:
            raise api_error(400, "name must not be empty", "invalid_request", "name")
        if not samples and not description:
            raise api_error(
                400, "Send samples, or a description for a designed voice", "invalid_request"
            )
        reference = None
        if samples:
            # The first recording becomes the reference; OmniVoice clones from a single clip.
            limit = settings.max_upload_mb * 1024 * 1024
            data = await samples[0].read(limit + 1)
            if len(data) > limit:
                raise api_error(
                    413,
                    f"A sample must be {settings.max_upload_mb} MiB or smaller",
                    "invalid_request",
                    "samples",
                )
            try:
                reference = await run_in_threadpool(audio.decode, data)
            except Exception as exc:
                raise api_error(
                    400,
                    f"Could not read audio file '{samples[0].filename}'",
                    "invalid_request",
                    "samples",
                ) from exc
        else:
            if unknown := vocab.unknown_instructions(description):
                raise api_error(
                    400,
                    f"OmniVoice cannot design a voice from {', '.join(map(repr, unknown))}. "
                    "Describe it with comma-separated items from: "
                    + ", ".join(vocab.valid_instructions()),
                    "invalid_request",
                    "description",
                )
        try:
            voice = await server.model.run(
                server.voices.create,
                name,
                renderer.engine,
                reference,
                transcript,
                description,
            )
        except VoiceExists as exc:
            raise api_error(409, f"Voice '{exc}' already exists", "voice_conflict", "name") from exc
        except Exception as exc:
            LOG.exception("Voice creation failed")
            raise api_error(422, f"Could not create voice: {exc}", "voice_creation_failed") from exc
        return voice.public()

    @app.delete("/v1/audio/voices/{voice_id}", status_code=204, dependencies=authorized)
    async def delete_voice(voice_id: str) -> Response:
        if not await server.model.run(server.voices.delete, voice_id):
            raise api_error(404, f"No voice '{voice_id}' on this server", "voice_not_found")
        return Response(status_code=204)

    def check_format(audio_format: str) -> None:
        if audio_format not in server.formats:
            raise api_error(
                400,
                f"Unsupported response_format '{audio_format}'",
                "unsupported",
                "response_format",
            )

    def check_extra(extra: dict[str, Any] | None) -> None:
        if server.engine is not None:
            try:
                resolve_options(server.engine.options, extra)
            except ValueError as exc:
                raise api_error(400, str(exc), "invalid_request", "extra") from exc

    @app.post("/v1/audio/speech/batch", dependencies=authorized)
    async def speech_batch(body: BatchRequest) -> StreamingResponse:
        if body.model != settings.api_model:
            raise api_error(404, f"No model '{body.model}'", "model_not_found", "model")
        if len(body.items) > settings.max_batch_items:
            raise api_error(
                400,
                f"At most {settings.max_batch_items} items are allowed",
                "too_many_items",
                "items",
            )
        if sum(len(item.input) for item in body.items) > settings.max_batch_chars:
            raise api_error(
                400,
                f"Batch exceeds {settings.max_batch_chars} input characters",
                "too_many_items",
                "items",
            )
        seen: set[str] = set()
        for item in body.items:
            if item.id in seen:
                raise api_error(400, f"Duplicate item id '{item.id}'", "invalid_request", "items")
            seen.add(item.id)
        check_format(body.response_format)
        renderer = server.ready()
        if body.sample_rate is not None and body.sample_rate != renderer.engine.sample_rate:
            raise api_error(
                400,
                f"Only sample_rate {renderer.engine.sample_rate} is supported",
                "unsupported",
                "sample_rate",
            )
        check_extra(body.extra)
        return StreamingResponse(
            renderer.stream(body.items, body.extra, body.response_format),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/v1/audio/speech", dependencies=authorized)
    async def speech(body: SpeechRequest) -> Response:
        if body.model not in (None, settings.api_model):
            raise api_error(404, f"No model '{body.model}'", "model_not_found", "model")
        check_format(body.response_format)
        renderer = server.ready()
        check_extra(body.extra)
        voice = body.voice
        if not voice:
            existing = await run_in_threadpool(server.voices.all)
            if len(existing) != 1:
                raise api_error(
                    400,
                    "voice is required unless exactly one voice exists",
                    "invalid_request",
                    "voice",
                )
            voice = existing[0].id
        item = SpeechItem(
            id="speech",
            input=body.input,
            voice=voice,
            speed=body.speed,
            instructions=body.instructions,
            language=body.language,
        )
        [job] = await server.model.run(renderer.prepare, [item], body.extra)
        if job.failure is not None:
            raise _speech_error(job.failure)
        engine = renderer.engine
        try:
            [samples] = await server.model.run(renderer.generate, [job.line], job.options)
            data = await run_in_threadpool(
                audio.encode, samples, body.response_format, engine.sample_rate
            )
        except Exception as exc:
            LOG.exception("Speech generation failed")
            code = "out_of_memory" if is_out_of_memory(exc) else "render_failed"
            raise _speech_error(ItemFailure(code, str(exc)[:500], True)) from exc
        return Response(
            content=data,
            media_type=audio.CONTENT_TYPES[body.response_format],
            headers={"Content-Disposition": f"inline; filename=speech.{body.response_format}"},
        )

    return app


def _speech_error(failure: ItemFailure) -> HTTPException:
    if failure.code == "out_of_memory":
        return api_error(503, failure.message, "overloaded", headers={"Retry-After": "5"})
    if failure.code == "render_failed":
        return api_error(500, failure.message or "Speech generation failed", "render_failed")
    param = {"voice_not_found": "voice", "invalid_request": "extra"}.get(failure.code, "input")
    return api_error(400, failure.message, failure.code, param)
