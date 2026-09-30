"""Errors in the OpenAI shape the speech batch API uses."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def error_body(message: str, code: str, param: str | None = None) -> dict[str, Any]:
    error = {"message": message, "type": "invalid_request_error", "code": code}
    if param:
        error["param"] = param
    return {"error": error}


def api_error(
    status: int,
    message: str,
    code: str,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    return HTTPException(status, detail=error_body(message, code, param), headers=headers)


class ItemFailure(Exception):
    """One batch item's failure; the other items carry on."""

    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}


def is_out_of_memory(exc: BaseException) -> bool:
    """CUDA and Metal both report exhaustion as a RuntimeError with one of these phrases."""
    message = str(exc).lower()
    return any(
        phrase in message
        for phrase in ("out of memory", "insufficient memory", "maximum allowed buffer size")
    )
