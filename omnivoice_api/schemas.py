"""Request bodies. Unknown fields are ignored, as the batch API's conventions ask."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SpeechItem(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    input: str
    voice: str
    instructions: str | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)
    language: str | None = None
    extra: dict[str, Any] | None = None


class BatchRequest(BaseModel):
    model: str
    response_format: str = "wav"
    sample_rate: int | None = None
    extra: dict[str, Any] | None = None
    items: list[SpeechItem] = Field(min_length=1)


class SpeechRequest(BaseModel):
    """OpenAI's single-line body; ``voice`` may be left out when the server has exactly one."""

    model: str | None = None
    input: str
    voice: str | None = None
    instructions: str | None = None
    speed: float | None = Field(None, ge=0.5, le=2.0)
    language: str | None = None
    response_format: str = "wav"
    extra: dict[str, Any] | None = None
