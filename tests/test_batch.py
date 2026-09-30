"""The batch route against the contract in audiobook-studio's docs/speech-batch-api.md."""

from __future__ import annotations

import pytest
from conftest import decode_wav, lines, make_voice


def batch(client, items, **body):
    return client.post(
        "/v1/audio/speech/batch", json={"model": "omnivoice", "items": items, **body}
    )


def items_by_id(stream):
    return {line["id"]: line for line in stream if line["type"] == "item"}


def test_every_item_is_answered_once_then_done(client):
    make_voice(client)
    stream = lines(
        batch(
            client,
            [
                {"id": "a", "input": "We are short again.", "voice": "mara"},
                {"id": "b", "input": "Then we count it twice.", "voice": "mara"},
            ],
        )
    )
    answers = [line for line in stream if line["type"] == "item"]
    assert sorted(line["id"] for line in answers) == ["a", "b"]
    assert stream[-1] == {
        "type": "done",
        "items": {"done": 2, "failed": 0},
        "usage": {"input_characters": 42, "audio_seconds": stream[-1]["usage"]["audio_seconds"]},
    }
    a = items_by_id(stream)["a"]
    assert a["index"] == 0 and a["status"] == "done" and a["format"] == "wav"
    samples, rate = decode_wav(a)
    assert rate == a["sample_rate"] == 24_000
    assert abs(len(samples) / rate - a["duration"]) < 0.01


def test_one_bad_item_fails_alone(client):
    make_voice(client)
    stream = lines(
        batch(
            client,
            [
                {"id": "ok", "input": "Fine.", "voice": "mara"},
                {"id": "ghost", "input": "Hello.", "voice": "nobody"},
                {"id": "empty", "input": "   ", "voice": "mara"},
                {"id": "long", "input": "x" * 61, "voice": "mara"},
            ],
        )
    )
    answers = items_by_id(stream)
    assert answers["ok"]["status"] == "done"
    assert answers["ghost"]["error"] == {
        "code": "voice_not_found",
        "message": "No voice 'nobody' on this server",
        "retryable": False,
    }
    assert answers["empty"]["error"]["code"] == "empty_input"
    assert answers["long"]["error"]["code"] == "input_too_long"
    assert stream[-1]["items"] == {"done": 1, "failed": 3}


def test_a_bad_item_option_fails_that_item_without_a_retry(client):
    make_voice(client)
    answers = items_by_id(
        lines(
            batch(
                client,
                [
                    {"id": "a", "input": "One.", "voice": "mara", "extra": {"num_step": "many"}},
                    {"id": "b", "input": "Two.", "voice": "mara", "extra": {"unknown_option": 1}},
                ],
            )
        )
    )
    assert answers["a"]["error"]["code"] == "invalid_request"
    assert answers["a"]["error"]["retryable"] is False
    assert answers["b"]["status"] == "done"


@pytest.mark.parametrize(
    "body, status, code",
    [
        ({"items": [{"id": "a", "input": "x", "voice": "v"}] * 2}, 400, "invalid_request"),
        (
            {"items": [{"id": str(i), "input": "x", "voice": "v"} for i in range(5)]},
            400,
            "too_many_items",
        ),
        ({"items": [{"id": "a", "input": "x" * 201, "voice": "v"}]}, 400, "too_many_items"),
        ({"items": []}, 400, "invalid_request"),
        (
            {"model": "omni", "items": [{"id": "a", "input": "x", "voice": "v"}]},
            404,
            "model_not_found",
        ),
        (
            {"response_format": "aac", "items": [{"id": "a", "input": "x", "voice": "v"}]},
            400,
            "unsupported",
        ),
        (
            {"sample_rate": 44100, "items": [{"id": "a", "input": "x", "voice": "v"}]},
            400,
            "unsupported",
        ),
        (
            {"extra": {"num_step": 0}, "items": [{"id": "a", "input": "x", "voice": "v"}]},
            400,
            "invalid_request",
        ),
        ({"items": [{"id": "a", "input": 5, "voice": "v"}]}, 400, "invalid_request"),
        ({"items": [{"id": "a", "input": "x", "voice": "v", "speed": 3}]}, 400, "invalid_request"),
    ],
)
def test_unreadable_requests_are_refused_whole(client, body, status, code):
    response = client.post("/v1/audio/speech/batch", json={"model": "omnivoice", **body})
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code


def test_invalid_json_is_an_invalid_request(client):
    response = client.post(
        "/v1/audio/speech/batch", content=b"{nope", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_unknown_fields_are_ignored(client):
    make_voice(client)
    stream = lines(
        batch(
            client, [{"id": "a", "input": "Hi.", "voice": "mara", "mood": "x"}], future_field=True
        )
    )
    assert items_by_id(stream)["a"]["status"] == "done"


def test_lines_sharing_options_share_one_engine_call(client, engine):
    make_voice(client)
    make_voice(client, name="Designed", transcript=None, description="female, british accent")
    lines(
        batch(
            client,
            [
                {"id": "a", "input": "One.", "voice": "mara"},
                {"id": "b", "input": "Two.", "voice": "mara", "extra": {"num_step": 32}},
                {"id": "c", "input": "Three.", "voice": "mara", "extra": {"num_step": 8}},
                {"id": "d", "input": "Four.", "voice": "designed"},
            ],
        )
    )
    assert sorted(len(call_lines) for call_lines, _ in engine.calls) == [1, 1, 2]


def test_free_form_instructions_do_not_fail_the_batch(client, engine):
    make_voice(client, name="Designed", transcript=None, description="female, british accent")
    stream = lines(
        batch(
            client,
            [
                {
                    "id": "a",
                    "input": "We are short.",
                    "voice": "designed",
                    "instructions": "Tired, flat, under her breath.",
                },
                {
                    "id": "b",
                    "input": "Count it twice.",
                    "voice": "designed",
                    "instructions": "Whisper, elderly",
                },
            ],
        )
    )
    assert stream[-1]["items"] == {"done": 2, "failed": 0}
    [(call_lines, _)] = engine.calls
    assert [line.instruct for line in call_lines] == [
        "female, british accent",
        "female, elderly, whisper, british accent",
    ]


def test_a_failing_line_is_isolated_from_its_group(client, engine):
    make_voice(client)
    real_generate = engine.generate

    def generate(lines, options):
        if any("boom" in line.text for line in lines):
            raise RuntimeError("model exploded")
        return real_generate(lines, options)

    engine.generate = generate
    answers = items_by_id(
        lines(
            batch(
                client,
                [
                    {"id": str(i), "input": "boom" if i == 2 else f"Line {i}.", "voice": "mara"}
                    for i in range(4)
                ],
            )
        )
    )
    assert [answers[str(i)]["status"] for i in range(4)] == ["done", "done", "failed", "done"]
    assert answers["2"]["error"] == {
        "code": "render_failed",
        "message": "model exploded",
        "retryable": True,
    }


def test_out_of_memory_retries_in_smaller_batches(client, engine):
    make_voice(client)
    real_generate = engine.generate
    sizes = []

    def generate(lines, options):
        sizes.append(len(lines))
        if len(lines) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return real_generate(lines, options)

    engine.generate = generate
    stream = lines(
        batch(client, [{"id": str(i), "input": "Hi.", "voice": "mara"} for i in range(4)])
    )
    assert stream[-1]["items"] == {"done": 4, "failed": 0}
    assert sizes == [4, 2, 1, 1, 2, 1, 1]


def test_out_of_memory_on_one_line_is_retryable(client, engine):
    make_voice(client)

    def generate(lines, options):
        raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")

    engine.generate = generate
    [line] = [
        line
        for line in lines(batch(client, [{"id": "a", "input": "Hi.", "voice": "mara"}]))
        if line["type"] == "item"
    ]
    assert line["error"]["code"] == "out_of_memory" and line["error"]["retryable"] is True


def test_slow_renders_send_pings(client, engine):
    make_voice(client)
    engine.delay = 0.2
    stream = lines(batch(client, [{"id": "a", "input": "Hi.", "voice": "mara"}]))
    assert {"type": "ping"} in stream
    assert stream[-1]["type"] == "done"


@pytest.mark.parametrize("audio_format", ["flac", "pcm"])
def test_other_formats(client, audio_format):
    make_voice(client)
    [line] = [
        line
        for line in lines(
            batch(
                client, [{"id": "a", "input": "Hi.", "voice": "mara"}], response_format=audio_format
            )
        )
        if line["type"] == "item"
    ]
    assert line["format"] == audio_format and line["status"] == "done"


def test_the_model_not_being_ready_is_a_503(settings):
    from fastapi.testclient import TestClient

    from omnivoice_api.app import create_app

    app = create_app(settings, engine=None)
    with TestClient(app) as client:
        app.state.server.renderer = None
        response = batch(client, [{"id": "a", "input": "Hi.", "voice": "mara"}])
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert response.json()["error"]["code"] == "overloaded"


def test_engine_batch_size_splits_a_group_into_calls(settings, engine):
    from fastapi.testclient import TestClient

    from omnivoice_api.app import create_app

    settings.engine_batch_size = 2
    with TestClient(create_app(settings, engine)) as client:
        make_voice(client)
        stream = lines(
            batch(client, [{"id": str(i), "input": "Hi.", "voice": "mara"} for i in range(4)])
        )
    assert stream[-1]["items"] == {"done": 4, "failed": 0}
    assert [len(call_lines) for call_lines, _ in engine.calls] == [2, 2]
