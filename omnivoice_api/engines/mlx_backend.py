"""OmniVoice on Apple Silicon through mlx-audio's MLX port.

The port's ``generate_batch()`` renders the model faithfully but leaves out parts of upstream's
``generate()``. This engine fills them in so both backends answer alike:

- target length from the reference voice's own pace, divided by ``speed`` (the port has no
  ``speed`` and estimates every line against a stock phrase);
- upstream's output post-processing: silence trimming, loudness matched to the reference, and a
  short fade and pad;
- upstream's voice-clone prompt: the same preprocessing of the recording, the transcript's closing
  punctuation, and the loudness kept for output matching. MLX saves it as ``.safetensors``.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .. import vocab
from ..config import Settings
from . import Line, Option, sampling_options

LOG = logging.getLogger("omnivoice_api")
FRAME_SAMPLES = 960  # audio tokenizer hop: 25 frames a second at 24 kHz


@dataclass(frozen=True)
class MlxPrompt:
    tokens: Any  # mx.array (frames, codebooks)
    transcript: str
    rms: float


class MlxEngine:
    prompt_suffix = ".safetensors"

    def __init__(self, settings: Settings):
        import mlx.core as mx
        from mlx_audio.tts.models.omnivoice.duration import RuleDurationEstimator
        from mlx_audio.tts.utils import load_model

        LOG.info("Loading %s with MLX", settings.resolved_model)
        # MLX keeps freed buffers for reuse, by default up to about the whole memory limit; batches
        # of varying shape fill it with buffers of many sizes, held long after the work is done.
        mx.set_cache_limit(int(settings.mlx_cache_gb * 2**30))
        self.model = load_model(settings.resolved_model)
        self.sample_rate = int(self.model.sample_rate)
        self.transcribe_model = settings.transcribe_model
        self._asr: Any = None
        self._durations = RuleDurationEstimator()
        self.options: dict[str, Option] = {
            **sampling_options(settings.default_num_steps),
            "postprocess_output": Option("boolean", True, "Trim silence from the output"),
        }

    def _frames(self, line: Line) -> int:
        """Upstream's ``_estimate_target_tokens``: the line's length at the voice's pace."""
        if line.prompt is not None and line.prompt.transcript:
            ref_text, ref_frames = line.prompt.transcript, int(line.prompt.tokens.shape[0])
        else:
            ref_text, ref_frames = "Nice to meet you.", 25
        estimate = self._durations.estimate_duration(line.text, ref_text, ref_frames)
        return max(1, int(estimate / line.speed))

    def generate(self, lines: Sequence[Line], options: Mapping[str, Any]) -> list[np.ndarray]:
        clone = lines[0].prompt is not None
        frame_rate = self.sample_rate / FRAME_SAMPLES
        try:
            results = self.model.generate_batch(
                text=[line.text for line in lines],
                language=[line.language or "None" for line in lines],
                instruct=[line.instruct or "None" for line in lines],
                ref_tokens=[line.prompt.tokens for line in lines] if clone else None,
                ref_text=[line.prompt.transcript for line in lines] if clone else None,
                duration_s=[self._frames(line) / frame_rate for line in lines],
                num_steps=options["num_step"],
                guidance_scale=options["guidance_scale"],
                t_shift=options["t_shift"],
                layer_penalty_factor=options["layer_penalty_factor"],
                position_temperature=options["position_temperature"],
                class_temperature=options["class_temperature"],
                max_batch_size=len(lines),
            )
            # Inside the try: MLX is lazy, so running out of memory can surface here.
            audios = [np.array(result.audio, dtype=np.float32).reshape(-1) for result in results]
        except Exception as exc:
            # As in the torch engine: keep only the message, so the traceback's frames and the
            # arrays they hold are freed before the cache is cleared and the caller retries.
            error = f"{type(exc).__name__}: {exc}"
        else:
            return [
                self._postprocess(audio, line.prompt.rms if clone else None,
                                  options["postprocess_output"])
                for line, audio in zip(lines, audios)
            ]
        import mlx.core as mx

        gc.collect()
        mx.clear_cache()
        raise RuntimeError(error)

    def _postprocess(self, audio: np.ndarray, ref_rms: float | None, trim: bool) -> np.ndarray:
        """Upstream's ``_post_process_audio`` with its default pad and fade of 0.1 s."""
        from mlx_audio.tts.models.omnivoice.utils import _remove_silence

        if trim:
            audio = _remove_silence(
                audio, self.sample_rate, mid_sil=500, lead_sil=100, trail_sil=100
            )
        if ref_rms is not None and ref_rms < 0.1:
            audio = audio * ref_rms / 0.1
        elif ref_rms is None and audio.size and np.abs(audio).max() > 1e-6:
            audio = audio / np.abs(audio).max() * 0.5
        edge = int(0.1 * self.sample_rate)
        fade = min(edge, audio.size // 2)
        if fade:
            audio = audio.copy()
            audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        silence = np.zeros(edge, dtype=np.float32)
        return np.concatenate([silence, audio, silence]).astype(np.float32)

    def encode_prompt(self, recording: Path, transcript: str | None) -> tuple[Any, str]:
        """Upstream's ``create_voice_clone_prompt``, step for step, so the prompt matches one made
        with k2-fsa/OmniVoice (the port's own helper trims silence differently and keeps no
        transcript or loudness)."""
        import mlx.core as mx
        from mlx_audio.codec.models.higgs_audio.higgs_audio import _sinc_resample
        from mlx_audio.tts.models.omnivoice.utils import _remove_silence, _trim_long_audio

        samples, rate = sf.read(recording, dtype="float32", always_2d=True)
        wav = samples.mean(axis=1).astype(np.float32)
        if rate != self.sample_rate:
            wav = np.asarray(_sinc_resample(wav, rate, self.sample_rate), dtype=np.float32)
        rms = float(np.sqrt(np.mean(wav**2)))
        if 0 < rms < 0.1:
            wav = wav * (0.1 / rms)
        if transcript is None:
            wav = _trim_long_audio(wav, self.sample_rate, max_duration=15.0, trim_threshold=20.0)
        wav = _remove_silence(wav, self.sample_rate, mid_sil=200, lead_sil=100, trail_sil=200)
        if wav.size == 0:
            raise ValueError("The recording is empty after silence removal")
        if len(wav) / self.sample_rate > 20:
            LOG.warning("%s is %.1f s long; 3-10 s clones better and renders faster",
                        recording.name, len(wav) / self.sample_rate)
        if transcript is None:
            transcript = self._transcribe(wav)
        if clip := len(wav) % FRAME_SAMPLES:
            wav = wav[:-clip]
        tokens = self.model.audio_tokenizer.encode(mx.array(wav)[None, :, None])[0]
        mx.eval(tokens)
        prompt = MlxPrompt(tokens, vocab.add_punctuation(transcript), rms)
        return prompt, prompt.transcript

    def save_prompt(self, prompt: Any, path: Path, source: str) -> None:
        import mlx.core as mx

        temp = path.with_name(path.name + ".tmp.safetensors")
        metadata = {"ref_text": prompt.transcript, "ref_rms": repr(prompt.rms), "source": source}
        mx.save_safetensors(str(temp), {"tokens": prompt.tokens}, metadata=metadata)
        temp.replace(path)

    def load_prompt(self, path: Path) -> tuple[Any, str | None]:
        import mlx.core as mx

        arrays, metadata = mx.load(str(path), return_metadata=True)
        prompt = MlxPrompt(arrays["tokens"], metadata["ref_text"], float(metadata["ref_rms"]))
        return prompt, metadata.get("source")

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe the reference after preprocessing, as upstream does, so the text matches."""
        if self._asr is None:
            from mlx_audio.stt.utils import load_model as load_stt

            LOG.info("Loading %s to transcribe a reference recording", self.transcribe_model)
            self._asr = load_stt(self.transcribe_model)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio.reshape(-1), self.sample_rate)
            text = self._asr.generate(tmp.name).text.strip()
        if not text:
            raise ValueError("Could not transcribe the reference recording; send a transcript")
        return text
