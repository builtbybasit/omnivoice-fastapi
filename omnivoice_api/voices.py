"""Voices are plain files in the voices directory; their encoded prompts live apart, in prompts/.

voices/ is yours to manage. For a voice named ``example``:

- ``example.wav`` (or .flac, .mp3, .m4a, .ogg, .opus): a recording to clone;
- ``example.txt``: what the recording says. The server writes it when it transcribes a recording
  itself, so a wrong transcript can be corrected by editing the file;
- ``example.json`` (optional): ``{"name", "description", "gender", "language"}``. A voice with no
  recording is designed from its ``description`` (words from OmniVoice's vocabulary).

A voice's id is its file name made URL-safe: ``Old Narrator.wav`` is ``old-narrator``.

prompts/ is a cache, safe to delete. A recording is encoded once into ``<id><suffix>``: upstream's
``VoiceClonePrompt`` ``.pt`` with the torch backend, ``.safetensors`` with MLX. Each prompt keeps a
fingerprint of the recording and transcript it came from, and is encoded again only when they
change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import vocab
from .engines import Engine

LOG = logging.getLogger("omnivoice_api")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus")
ID_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class VoiceExists(Exception):
    pass


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    stem: str  # the voice's file name in the voices directory, without suffix
    recording: Path | None = None
    transcript: str | None = None
    description: str | None = None
    gender: str | None = None
    language: str | None = None

    @property
    def clone(self) -> bool:
        return self.recording is not None

    @property
    def instructions(self) -> str | None:
        """A designed voice is its description; a clone's description is only for people."""
        return None if self.clone else vocab.resolve_instruct(self.description)

    def public(self) -> dict[str, str]:
        result = {"id": self.id, "name": self.name}
        for key in ("gender", "language", "description"):
            if value := getattr(self, key):
                result[key] = value
        return result


def voice_id_for(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48].strip("-") or "voice"


def fingerprint(recording: Path, transcript: str | None) -> str:
    digest = hashlib.sha256(recording.read_bytes())
    digest.update(b"\0" + (transcript or "").encode())
    return digest.hexdigest()


def glob_escape(text: str) -> str:
    return re.sub(r"([*?\[])", r"[\1]", text)


class VoiceStore:
    def __init__(self, root: Path, prompts: Path):
        self.root = root.resolve()
        self.prompts = prompts.resolve()
        self._lock = threading.Lock()
        self._prompts: dict[str, tuple[tuple, Any]] = {}

    def _files(self) -> dict[str, dict[str, Path]]:
        """Each voice id's files by role: "recording", ".txt" and ".json"."""
        groups: dict[str, dict[str, Path]] = {}
        if not self.root.is_dir():
            return groups
        for path in sorted(self.root.iterdir()):
            suffix = path.suffix.lower()
            if path.name.startswith(".") or not path.is_file():
                continue
            if suffix not in (*AUDIO_SUFFIXES, ".txt", ".json"):
                continue
            files = groups.setdefault(voice_id_for(path.stem), {})
            if files and next(iter(files.values())).stem != path.stem:
                LOG.warning("Ignoring %s: another file name makes the same voice id", path.name)
                continue
            files.setdefault("recording" if suffix in AUDIO_SUFFIXES else suffix, path)
        # A transcript alone is not a voice: it needs a recording or a .json.
        return {vid: fs for vid, fs in groups.items() if {"recording", ".json"} & fs.keys()}

    def _voice(self, voice_id: str, files: dict[str, Path]) -> Voice | None:
        try:
            metadata = json.loads(files[".json"].read_text("utf-8")) if ".json" in files else {}
            transcript = files[".txt"].read_text("utf-8").strip() if ".txt" in files else ""
        except (OSError, ValueError) as exc:
            LOG.warning("Ignoring voice %s: %s", voice_id, exc)
            return None
        stem = next(iter(files.values())).stem
        return Voice(
            id=voice_id,
            name=str(metadata.get("name") or stem),
            stem=stem,
            recording=files.get("recording"),
            transcript=transcript or None,
            description=metadata.get("description"),
            gender=metadata.get("gender"),
            language=metadata.get("language"),
        )

    def get(self, voice_id: str) -> Voice | None:
        if not ID_PATTERN.fullmatch(voice_id) or len(voice_id) > 48:
            return None
        files = self._files().get(voice_id)
        return self._voice(voice_id, files) if files else None

    def all(self) -> list[Voice]:
        voices = (self._voice(vid, files) for vid, files in sorted(self._files().items()))
        return [voice for voice in voices if voice is not None]

    def prompt(self, voice: Voice, engine: Engine) -> Any:
        """The voice's clone prompt: from memory, else from prompts/, else encoded now and saved.

        Model thread only.
        """
        assert voice.recording is not None
        transcript_file = self.root / f"{voice.stem}.txt"
        key = _stat(voice.recording, transcript_file, engine)
        with self._lock:
            cached = self._prompts.get(voice.id)
        if cached and cached[0] == key:
            return cached[1]
        path = self.prompts / f"{voice.id}{engine.prompt_suffix}"
        source = fingerprint(voice.recording, voice.transcript)
        prompt = None
        if path.is_file():
            try:
                saved, saved_source = engine.load_prompt(path)
                prompt = saved if saved_source == source else None
            except Exception as exc:
                LOG.warning("Encoding %s again: %s is unreadable (%s)", voice.id, path.name, exc)
        if prompt is None:
            LOG.info("Encoding voice %s from %s", voice.id, voice.recording.name)
            prompt, transcript = engine.encode_prompt(voice.recording, voice.transcript)
            if voice.transcript is None:
                transcript_file.write_text(transcript + "\n", encoding="utf-8")
                LOG.info("Transcribed %s into %s", voice.recording.name, transcript_file.name)
                source = fingerprint(voice.recording, transcript)
                key = _stat(voice.recording, transcript_file, engine)
            self.prompts.mkdir(parents=True, exist_ok=True)
            engine.save_prompt(prompt, path, source)
        with self._lock:
            self._prompts[voice.id] = (key, prompt)
        return prompt

    def create(
        self,
        name: str,
        engine: Engine,
        recording: tuple[np.ndarray, int] | None = None,
        transcript: str | None = None,
        description: str | None = None,
    ) -> Voice:
        """Write a voice's files. A clone is encoded now, so a bad recording fails here and its
        files are removed again. Model thread only."""
        voice_id = voice_id_for(name)
        if voice_id in self._files():
            raise VoiceExists(voice_id)
        self.root.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        try:
            if recording is not None:
                written.append(self.root / f"{voice_id}.wav")
                sf.write(written[-1], recording[0], recording[1])
            if transcript:
                written.append(self.root / f"{voice_id}.txt")
                written[-1].write_text(transcript + "\n", encoding="utf-8")
            metadata = {"name": name if name != voice_id else None, "description": description}
            metadata = {key: value for key, value in metadata.items() if value}
            if metadata or recording is None:
                written.append(self.root / f"{voice_id}.json")
                text = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
                written[-1].write_text(text, encoding="utf-8")
            voice = self.get(voice_id)
            assert voice is not None
            if voice.clone:
                self.prompt(voice, engine)
            return self.get(voice_id) or voice
        except BaseException:
            for path in [*written, self.root / f"{voice_id}.txt"]:
                path.unlink(missing_ok=True)
            self._forget(voice_id)
            raise

    def delete(self, voice_id: str) -> bool:
        files = self._files().get(voice_id) if ID_PATTERN.fullmatch(voice_id) else None
        if not files:
            return False
        for path in files.values():
            path.unlink()
        self._forget(voice_id)
        return True

    def _forget(self, voice_id: str) -> None:
        with self._lock:
            self._prompts.pop(voice_id, None)
        if self.prompts.is_dir():
            for path in self.prompts.glob(f"{voice_id}.*"):
                path.unlink()

    def migrate_folders(self) -> None:
        """Flatten voices made by earlier versions, a folder each (voice.json, reference.wav,
        voice.pt), into this layout. Their old prompts are dropped and encoded again once."""
        if not self.root.is_dir():
            return
        for folder in sorted(path for path in self.root.iterdir() if path.is_dir()):
            metadata_file = folder / "voice.json"
            if not metadata_file.is_file():
                continue
            name = folder.name
            if any(self.root.glob(f"{glob_escape(name)}.*")):
                LOG.warning("Not converting %s/: files named %s.* already exist", folder.name, name)
                continue
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            transcript = metadata.get("transcript") or _transcript_from_old_prompt(folder)
            if (folder / "reference.wav").is_file():
                (folder / "reference.wav").rename(self.root / f"{name}.wav")
            if transcript:
                (self.root / f"{name}.txt").write_text(transcript + "\n", encoding="utf-8")
            kept = {
                "name": metadata.get("name"),
                "description": metadata.get("description") or metadata.get("instructions"),
                "gender": metadata.get("gender"),
                "language": metadata.get("language"),
            }
            kept = {key: value for key, value in kept.items() if value}
            text = json.dumps(kept, indent=2, ensure_ascii=False) + "\n"
            (self.root / f"{name}.json").write_text(text, encoding="utf-8")
            for leftover in ("voice.json", "voice.pt", "voice.mlx.npz"):
                (folder / leftover).unlink(missing_ok=True)
            if not any(folder.iterdir()):
                folder.rmdir()
            LOG.info("Converted voice folder %s/ into %s.*", name, name)


def _stat(recording: Path, transcript_file: Path, engine: Engine) -> tuple:
    stats = [path.stat() if path.is_file() else None for path in (recording, transcript_file)]
    return (engine.prompt_suffix, *((s.st_mtime_ns, s.st_size) if s else None for s in stats))


def _transcript_from_old_prompt(folder: Path) -> str | None:
    """Earlier versions kept the transcript only inside voice.pt (torch backend)."""
    if not (folder / "voice.pt").is_file():
        return None
    try:
        import torch

        data = torch.load(folder / "voice.pt", map_location="cpu", weights_only=True)
        return str(data["ref_text"])
    except Exception:
        return None
