"""Turning request items into engine calls, and engine results into the NDJSON stream."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rich.markup import escape
from starlette.concurrency import run_in_threadpool

from . import audio, vocab
from .engines import Engine, Line, ModelThread, resolve_options
from .errors import ItemFailure, is_out_of_memory
from .schemas import SpeechItem
from .voices import VoiceStore

LOG = logging.getLogger("omnivoice_api")
PING = {"type": "ping"}
RICH = {"markup": True, "highlighter": None}  # log lines that colour themselves


@dataclass
class Job:
    index: int
    item: SpeechItem
    line: Line | None = None
    options: dict[str, Any] = field(default_factory=dict)
    failure: ItemFailure | None = None

    @property
    def group(self) -> tuple[bool, str]:
        """Lines share an engine call when both are cloned (or both not) and options match."""
        assert self.line is not None
        return self.line.prompt is not None, json.dumps(self.options, sort_keys=True)


class Renderer:
    def __init__(
        self,
        engine: Engine,
        model: ModelThread,
        voices: VoiceStore,
        max_item_chars: int,
        ping_seconds: float,
        engine_batch_size: int,
    ):
        self.engine = engine
        self.model = model
        self.voices = voices
        self.max_item_chars = max_item_chars
        self.ping_seconds = ping_seconds
        self.engine_batch_size = engine_batch_size

    def prepare(self, items: Sequence[SpeechItem], extra: dict[str, Any] | None) -> list[Job]:
        """Check each item and resolve its voice and options. Model thread: loads clone prompts."""
        jobs = []
        for index, item in enumerate(items):
            job = Job(index, item)
            try:
                job.line, job.options = self._line(item, extra)
            except ItemFailure as failure:
                # A copy that was never raised: the original's traceback holds this frame, whose
                # jobs hold it, a cycle that keeps a failed voice load's audio until the gc runs.
                job.failure = ItemFailure(failure.code, failure.message, failure.retryable)
            jobs.append(job)
        return jobs

    def _line(self, item: SpeechItem, extra: dict[str, Any] | None) -> tuple[Line, dict[str, Any]]:
        if not item.input.strip():
            raise ItemFailure("empty_input", "input must not be empty", False)
        if len(item.input) > self.max_item_chars:
            raise ItemFailure(
                "input_too_long", f"input exceeds {self.max_item_chars} characters", False
            )
        try:
            options = resolve_options(self.engine.options, extra, item.extra)
        except ValueError as exc:
            raise ItemFailure("invalid_request", str(exc), False) from exc
        voice = self.voices.get(item.voice)
        if voice is None:
            raise ItemFailure("voice_not_found", f"No voice '{item.voice}' on this server", False)
        prompt = None
        if voice.clone:
            try:
                prompt = self.voices.prompt(voice, self.engine)
            except Exception as exc:
                LOG.exception("Could not load voice %s", voice.id)
                raise ItemFailure(
                    "render_failed", f"Voice '{voice.id}' could not be loaded: {exc}", False
                ) from exc
        if item.instructions and (ignored := vocab.unknown_instructions(item.instructions)):
            LOG.debug(
                "Item %s: ignoring instructions OmniVoice cannot follow: %s", item.id, ignored
            )
        line = Line(
            text=item.input,
            language=vocab.resolve_language(item.language or voice.language),
            instruct=vocab.resolve_instruct(voice.instructions, item.instructions, text=item.input),
            speed=item.speed or 1.0,
            prompt=prompt,
        )
        return line, options

    def generate(self, lines: Sequence[Line], options: dict[str, Any]) -> list[np.ndarray]:
        """Model thread: one engine call, logged with its size, time and speed."""
        started = time.perf_counter()
        audios = self.engine.generate(lines, options)
        wall = time.perf_counter() - started
        seconds = [len(samples) / self.engine.sample_rate for samples in audios] or [0.0]
        chars = [len(line.text) for line in lines]
        # Every line in a call is padded to the longest, so this share of the work is thrown away.
        padding = 1 - sum(seconds) / (len(seconds) * max(seconds)) if max(seconds) else 0.0
        kind = "clone" if lines[0].prompt is not None else "designed"
        LOG.info(
            f"[bold]{len(lines)} line{'s' * (len(lines) > 1)}[/] [dim]{kind} · "
            f"{options.get('num_step')} steps[/] │ {sum(chars)} chars → "
            f"[bold]{sum(seconds):.1f} s[/] audio in {wall:.2f} s │ "
            f"{_speed(sum(seconds), wall)} │ {sum(chars) / wall:.0f} chars/s │ {_padding(padding)}",
            extra=RICH,
        )
        if LOG.isEnabledFor(logging.DEBUG):
            for number, (line, length) in enumerate(zip(lines, seconds), 1):
                LOG.debug(
                    f"  [cyan]#{number:<2}[/] {len(line.text):>4} chars → {length:5.1f} s  "
                    f"[dim]{escape(line.text[:80])}[/]",
                    extra=RICH,
                )
        return audios

    async def _pinging(self, work: Awaitable[Any], result: list[Any]) -> AsyncIterator[dict]:
        """Await ``work``, yielding a ping every ``ping_seconds``; its result goes in ``result``."""
        task = asyncio.ensure_future(work)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self.ping_seconds)
                if done:
                    break
                yield PING
            result.append(task.result())
        finally:
            task.cancel()

    async def render(self, jobs: list[Job], audio_format: str) -> AsyncIterator[dict]:
        """Item lines for ``jobs`` (all one group). A failed call is retried in halves, which gets
        a batch past an out-of-memory error and confines a bad line's failure to that line."""
        outcome: list[Any] = []
        try:
            async for ping in self._pinging(
                self.model.run(self.generate, [job.line for job in jobs], jobs[0].options),
                outcome,
            ):
                yield ping
            audios = outcome[0]
            if len(audios) != len(jobs):
                raise RuntimeError(
                    f"The model returned {len(audios)} results for {len(jobs)} lines"
                )
        except Exception as exc:
            if len(jobs) > 1:
                LOG.warning("A batch of %d failed (%s); retrying it in halves", len(jobs), exc)
                middle = len(jobs) // 2
                for half in (jobs[:middle], jobs[middle:]):
                    async for line in self.render(half, audio_format):
                        yield line
                return
            LOG.exception("Rendering item %s failed", jobs[0].item.id)
            code = "out_of_memory" if is_out_of_memory(exc) else "render_failed"
            yield failed(jobs[0], ItemFailure(code, str(exc)[:500] or "Rendering failed", True))
            return
        for job, samples in zip(jobs, audios):
            try:
                yield await run_in_threadpool(self.finished, job, samples, audio_format)
            except Exception as exc:
                LOG.exception("Encoding item %s failed", job.item.id)
                yield failed(
                    job, ItemFailure("render_failed", f"Audio encoding failed: {exc}", True)
                )

    def finished(self, job: Job, samples: np.ndarray, audio_format: str) -> dict[str, Any]:
        rate = self.engine.sample_rate
        seconds = round(len(samples) / rate, 3)
        return {
            "type": "item",
            "id": job.item.id,
            "index": job.index,
            "status": "done",
            "format": audio_format,
            "sample_rate": rate,
            "duration": seconds,
            "audio": base64.b64encode(audio.encode(samples, audio_format, rate)).decode("ascii"),
            "usage": {"input_characters": len(job.item.input), "audio_seconds": seconds},
        }

    async def stream(
        self, items: Sequence[SpeechItem], extra: dict[str, Any] | None, audio_format: str
    ) -> AsyncIterator[bytes]:
        answered: dict[int, dict] = {}
        started, first = time.perf_counter(), None
        try:
            prepared: list[list[Job]] = []
            async for ping in self._pinging(self.model.run(self.prepare, items, extra), prepared):
                yield encode_line(ping)
            groups: dict[tuple[bool, str], list[Job]] = {}
            for job in prepared[0]:
                if job.failure:
                    answered[job.index] = failed(job, job.failure)
                    yield encode_line(answered[job.index])
                else:
                    groups.setdefault(job.group, []).append(job)
            size = self.engine_batch_size
            chunks = [
                group[i : i + size] for group in groups.values() for i in range(0, len(group), size)
            ]
            for chunk in chunks:
                async for line in self.render(chunk, audio_format):
                    if line["type"] == "item":  # what the summary needs, not the audio
                        first = first or time.perf_counter() - started
                        answered[line["index"]] = {k: line.get(k) for k in ("status", "duration")}
                    yield encode_line(line)
        except Exception as exc:
            # The 200 is already sent: fail what is left rather than the request.
            LOG.exception("Batch failed")
            for index, item in enumerate(items):
                if index not in answered:
                    error = ItemFailure("render_failed", str(exc)[:500] or "Batch failed", True)
                    answered[index] = failed(Job(index, item), error)
                    yield encode_line(answered[index])
        done = [line for line in answered.values() if line["status"] == "done"]
        wall, audio_seconds = time.perf_counter() - started, sum(line["duration"] for line in done)
        failures = len(answered) - len(done)
        LOG.info(
            f"[bold magenta]batch[/] [bold]{len(items)} items[/] │ [green]{len(done)} done[/]"
            + (f" [bold red]{failures} failed[/]" if failures else "")
            + f" │ {sum(len(item.input) for item in items)} chars → [bold]{audio_seconds:.1f} s"
            f"[/] audio in {wall:.2f} s │ {_speed(audio_seconds, wall)} │ "
            f"first item after {first or 0.0:.2f} s",
            extra=RICH,
        )
        yield encode_line(
            {
                "type": "done",
                "items": {"done": len(done), "failed": len(answered) - len(done)},
                "usage": {
                    "input_characters": sum(len(item.input) for item in items),
                    "audio_seconds": round(sum(line["duration"] for line in done), 3),
                },
            }
        )


def _speed(audio_seconds: float, wall: float) -> str:
    speed = audio_seconds / wall if wall else 0.0
    return f"[bold {'green' if speed >= 1 else 'red'}]{speed:.2f}x[/] real time"


def _padding(share: float) -> str:
    colour = "green" if share < 0.25 else "yellow" if share < 0.5 else "bold red"
    return f"[{colour}]padding {share:.0%}[/]"


def failed(job: Job, error: ItemFailure) -> dict[str, Any]:
    return {
        "type": "item",
        "id": job.item.id,
        "index": job.index,
        "status": "failed",
        "error": error.as_dict(),
    }


def encode_line(line: dict[str, Any]) -> bytes:
    return (json.dumps(line, ensure_ascii=False) + "\n").encode()
