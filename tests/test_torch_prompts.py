"""Prompt encoding must not retain autograd graphs or failed tokenizer frames."""

import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from omnivoice_api.engines.torch_backend import TorchEngine


def test_prompt_encoding_disables_autograd_and_saves_cpu_tokens(tmp_path, monkeypatch):
    layer = torch.nn.Linear(4, 4)

    def encode(**kwargs):
        assert torch.is_inference_mode_enabled()
        assert kwargs == {"ref_audio": "reference.wav", "ref_text": "Hello."}
        return SimpleNamespace(
            ref_audio_tokens=layer(torch.ones(1, 4)), ref_text="Hello.", ref_rms=0.1
        )

    engine = TorchEngine.__new__(TorchEngine)
    engine.model = SimpleNamespace(create_voice_clone_prompt=encode)
    released = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(True))

    prompt, transcript = engine.encode_prompt(Path("reference.wav"), "Hello.")

    assert transcript == "Hello."
    assert prompt.ref_audio_tokens.device.type == "cpu"
    assert not prompt.ref_audio_tokens.requires_grad
    assert prompt.ref_audio_tokens.grad_fn is None
    assert released == [True]
    assert not torch.is_inference_mode_enabled()

    path = tmp_path / "voice.pt"
    engine.save_prompt(prompt, path, "fingerprint")
    saved = torch.load(path, weights_only=True)
    assert torch.equal(saved["ref_audio_tokens"], prompt.ref_audio_tokens)
    assert saved["ref_text"] == transcript
    assert saved["source"] == "fingerprint"


def test_failed_encoding_releases_traceback_tensors_before_emptying_cache(monkeypatch):
    tensors = []

    def encode(**kwargs):
        activation = torch.ones(4, requires_grad=True)
        tensors.append(weakref.ref(activation))
        raise RuntimeError("tokenizer failed")

    def empty_cache():
        assert tensors[0]() is None

    engine = TorchEngine.__new__(TorchEngine)
    engine.model = SimpleNamespace(create_voice_clone_prompt=encode)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    with pytest.raises(RuntimeError, match="RuntimeError: tokenizer failed") as caught:
        engine.encode_prompt(Path("reference.wav"), None)

    assert caught.value.__context__ is None
    assert not torch.is_inference_mode_enabled()
