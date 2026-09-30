from __future__ import annotations

import io
import json

import soundfile as sf
from conftest import make_voice, wav_bytes


def files(directory):
    return sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []


def test_clone_voice_round_trip(client, settings):
    voice = make_voice(client, name="Mara Voss!", description="Cloned from the author")
    assert voice == {
        "id": "mara-voss",
        "name": "Mara Voss!",
        "description": "Cloned from the author",
    }
    assert client.get("/v1/audio/voices").json() == {"voices": [voice]}
    assert files(settings.voices_dir) == ["mara-voss.json", "mara-voss.txt", "mara-voss.wav"]
    assert (settings.voices_dir / "mara-voss.txt").read_text() == "Hello there.\n"
    assert json.loads((settings.voices_dir / "mara-voss.json").read_text()) == {
        "name": "Mara Voss!",
        "description": "Cloned from the author",
    }
    assert files(settings.prompts_dir) == ["mara-voss.fake.json"]


def test_a_missing_transcript_is_written_for_you(client, settings):
    response = client.post(
        "/v1/audio/voices",
        data={"name": "tobin"},
        files={"samples": ("t.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    assert files(settings.voices_dir) == ["tobin.txt", "tobin.wav"]
    assert (settings.voices_dir / "tobin.txt").read_text() == "(transcribed by the fake engine)\n"


def test_files_you_add_are_voices(settings, engine):
    from fastapi.testclient import TestClient

    from omnivoice_api.app import create_app

    settings.voices_dir.mkdir()
    (settings.voices_dir / "Old Narrator.wav").write_bytes(wav_bytes())
    (settings.voices_dir / "Old Narrator.txt").write_text(" Words in the recording. \n")
    (settings.voices_dir / "narrator.json").write_text('{"description": "female, whisper"}')
    (settings.voices_dir / "notes.txt").write_text("a transcript alone is not a voice")
    with TestClient(create_app(settings, engine)) as client:
        assert client.get("/v1/audio/voices").json() == {
            "voices": [
                {"id": "narrator", "name": "narrator", "description": "female, whisper"},
                {"id": "old-narrator", "name": "Old Narrator"},
            ]
        }
        stream = client.post(
            "/v1/audio/speech/batch",
            json={
                "model": "omnivoice",
                "items": [
                    {"id": "a", "input": "Hi.", "voice": "old-narrator"},
                    {"id": "b", "input": "Hi.", "voice": "narrator"},
                ],
            },
        ).text
        assert stream.count('"status": "done"') == 2
    assert [line.instruct for lines, _ in engine.calls for line in lines if not line.prompt] == [
        "female, whisper"
    ]


def test_a_prompt_is_encoded_once_and_saved(settings):
    from omnivoice_api.engines.fake import FakeEngine
    from omnivoice_api.voices import VoiceStore

    settings.voices_dir.mkdir()
    (settings.voices_dir / "mara.wav").write_bytes(wav_bytes())
    (settings.voices_dir / "mara.txt").write_text("Hello there.")
    engine = FakeEngine()
    store = VoiceStore(settings.voices_dir, settings.prompts_dir)
    voice = store.get("mara")
    assert store.prompt(voice, engine) == store.prompt(voice, engine) == "prompt:mara"
    assert engine.encodes == 1

    restarted = VoiceStore(settings.voices_dir, settings.prompts_dir)  # loads prompts/mara.*
    assert restarted.prompt(restarted.get("mara"), engine) == "prompt:mara"
    assert engine.encodes == 1

    (settings.voices_dir / "mara.txt").write_text("A corrected transcript.")
    restarted.prompt(restarted.get("mara"), engine)
    assert engine.encodes == 2  # a changed transcript is encoded again

    VoiceStore(settings.voices_dir, settings.prompts_dir).prompt(restarted.get("mara"), engine)
    assert engine.encodes == 2


def test_voices_from_earlier_versions_are_converted(settings, engine):
    from fastapi.testclient import TestClient

    from omnivoice_api.app import create_app

    old = settings.voices_dir / "mara"
    old.mkdir(parents=True)
    (old / "voice.json").write_text('{"name": "Mara", "source_sample": "reference.wav"}')
    (old / "reference.wav").write_bytes(wav_bytes())
    (old / "voice.pt").write_bytes(b"old prompt")
    designed = settings.voices_dir / "narrator"
    designed.mkdir()
    (designed / "voice.json").write_text('{"name": "Narrator", "instructions": "male"}')
    with TestClient(create_app(settings, engine)) as client:
        voices = client.get("/v1/audio/voices").json()["voices"]
    assert voices == [
        {"id": "mara", "name": "Mara"},
        {"id": "narrator", "name": "Narrator", "description": "male"},
    ]
    assert files(settings.voices_dir) == ["mara.json", "mara.wav", "narrator.json"]


def test_duplicate_voice_is_a_conflict(client):
    make_voice(client)
    response = client.post(
        "/v1/audio/voices",
        data={"name": "mara", "transcript": "x"},
        files={"samples": ("m.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "voice_conflict"


def test_unreadable_sample_is_refused(client, settings):
    response = client.post(
        "/v1/audio/voices",
        data={"name": "Bad"},
        files={"samples": ("bad.wav", b"not audio", "audio/wav")},
    )
    assert response.status_code == 400
    assert files(settings.voices_dir) == []


def test_designed_voice_needs_words_omnivoice_knows(client, settings):
    response = client.post(
        "/v1/audio/voices", data={"name": "Narrator", "description": "female, low, British"}
    )
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "'low'" in message and "'British'" in message and "low pitch" in message

    voice = make_voice(
        client, name="Narrator", transcript=None, description="Female, low pitch, British accent"
    )
    assert voice["id"] == "narrator"
    assert files(settings.voices_dir) == ["narrator.json"]


def test_failed_encoding_leaves_nothing_behind(client, engine, settings):
    def broken(recording, transcript):
        raise RuntimeError("tokenizer failed")

    engine.encode_prompt = broken
    response = client.post(
        "/v1/audio/voices",
        data={"name": "Broken"},
        files={"samples": ("b.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 422
    assert files(settings.voices_dir) == files(settings.prompts_dir) == []


def test_delete_voice(client, settings):
    make_voice(client)
    assert client.delete("/v1/audio/voices/mara").status_code == 204
    assert client.get("/v1/audio/voices").json() == {"voices": []}
    assert files(settings.voices_dir) == files(settings.prompts_dir) == []
    assert client.delete("/v1/audio/voices/mara").status_code == 404


def test_single_speech_returns_audio(client):
    make_voice(client)
    response = client.post(
        "/v1/audio/speech",
        json={"model": "omnivoice", "input": "Hello.", "voice": "mara", "speed": 1.5},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/wav"
    samples, rate = sf.read(io.BytesIO(response.content))
    assert rate == 24_000 and len(samples) > 0


def test_single_speech_uses_the_only_voice(client):
    make_voice(client)
    response = client.post("/v1/audio/speech", json={"input": "Hello."})
    assert response.status_code == 200


def test_single_speech_errors(client):
    make_voice(client)
    missing = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "nobody"})
    assert missing.status_code == 400
    assert missing.json()["error"] == {
        "message": "No voice 'nobody' on this server",
        "type": "invalid_request_error",
        "code": "voice_not_found",
        "param": "voice",
    }
    too_long = client.post("/v1/audio/speech", json={"input": "x" * 61, "voice": "mara"})
    assert too_long.json()["error"]["code"] == "input_too_long"


def test_capabilities(client):
    [model] = client.get("/v1/audio/speech/capabilities").json()["models"]
    assert model["id"] == "omnivoice"
    assert model["batch"] == {"max_items": 4, "max_input_chars": 200}
    assert model["sample_rates"] == [24_000]
    assert model["extra"]["num_step"] == {
        "type": "integer",
        "default": 32,
        "minimum": 1,
        "maximum": 200,
        "description": "Diffusion decoding steps",
    }
    assert "whisper" in model["instruction_vocabulary"]
    assert "laughter" in model["tags"]["known"]


def test_api_key(settings, engine):
    from fastapi.testclient import TestClient

    from omnivoice_api.app import create_app

    settings.api_key = "secret"
    with TestClient(create_app(settings, engine)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/audio/voices").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        assert client.get("/v1/audio/voices", headers=headers).status_code == 200
