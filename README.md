# OmniVoice FastAPI

A local FastAPI server implementing the [Audiobook Studio speech batch API](https://github.com/builtbybasit/audiobook-studio/blob/main/docs/speech-batch-api.md) on top of [OmniVoice](https://github.com/k2-fsa/OmniVoice).

It runs OmniVoice on one of three backends:

| Backend | Where | Model |
| --- | --- | --- |
| `torch` | NVIDIA GPU on Linux/WSL2, optionally with FlashInfer | `k2-fsa/OmniVoice` (upstream PyTorch) |
| `mlx` | Apple Silicon Macs | `mlx-community/OmniVoice-bf16` via [mlx-audio](https://github.com/Blaizzy/mlx-audio) |
| `fake` | anywhere | none: renders tones, for tests and API work |

`OMNIVOICE_BACKEND=auto` (the default) picks `mlx` on Apple Silicon and `torch` elsewhere. Voices are stored in a backend-neutral form, so a voice made on the Mac also works on the GPU box.

## Start the server

### Apple Silicon

```bash
bash scripts/run-mac.sh
```

The script installs the `mlx` extra, downloads the model (1.6 GB) to `~/.cache/huggingface/hub` if it isn't there yet, and starts the API at `http://127.0.0.1:8000`. Rerunning it resumes an interrupted download. On an M1 with 16 GB, a line takes a few seconds and a batch of four peaks around 2.6 GB. For a laptop, `OMNIVOICE_MAX_BATCH_ITEMS=4` and `extra.num_step=16` keep batches quick.

### Windows with an NVIDIA GPU (through WSL2)

On a Windows PC the server runs inside WSL2, not natively. FlashInfer roughly doubles OmniVoice's generation speed, and it only runs on Linux.

You need:
- The NVIDIA driver installed on **Windows**, recent enough for CUDA 12.8 (R570 or newer). Don't install a Linux driver inside WSL; WSL uses the Windows one.
- [uv](https://docs.astral.sh/uv/) installed inside WSL.
- The repository cloned under the Linux home directory (not `/mnt/c`).

Then:

```bash
bash scripts/run-wsl.sh
```

That one command does the setup and starts the server. It skips any step that's already done, so after the first run it starts in seconds. `bash scripts/run-wsl.sh --setup-only` prepares everything without starting.

On each run it:

1. Checks WSL, the project location, and that the Windows NVIDIA driver supports CUDA 12.8 or newer.
2. Installs `ffmpeg` with `apt` if it's missing, for `mp3` and `opus` output. `sudo` may ask for your WSL password. If the install fails, the server still starts without those formats.
3. Installs the `cuda` extra from `uv.lock`, including `flashinfer-python`.
4. Installs the FlashInfer JIT cache if it's missing: `flashinfer_jit_cache-0.6.15.post1+cu128`, a 1.3 GB wheel of precompiled kernels, kept outside `uv.lock`.
   - Downloads of it through pip/uv have stalled, so the script fetches it from the GitHub release with `curl`. Stalled transfers are restarted, and rerunning the script resumes an interrupted download.
   - The wheel is checked against its SHA-256 and kept in `~/.cache/omnivoice-fastapi/wheels`.
   - A plain `uv sync` removes the JIT cache. The next run reinstalls it from that saved wheel without downloading again.
   - To use a wheel you downloaded some other way: `FLASHINFER_JIT_CACHE_WHEEL=/path/to/wheel bash scripts/run-wsl.sh`.
5. Downloads the model (3.3 GB) the first time, so a slow download shows up in the terminal and resumes on the next run.
6. Starts the API with FlashInfer required. Set `OMNIVOICE_ENABLE_FLASHINFER=false` to run the slower baseline path on purpose.

Windows apps can call the WSL server at `http://localhost:8000/v1`. Interactive API docs are at `/docs`. Both run scripts take `HOST` and `PORT` (default `127.0.0.1:8000`). For access from another machine, use `HOST=0.0.0.0` and set a long random `OMNIVOICE_API_KEY`; requests then need `Authorization: Bearer <key>`.

## Development

```bash
uv sync --extra mlx          # or no extra: the tests only need the fake backend
uv run pytest                # API tests against the fake engine, in about a second
OMNIVOICE_TEST_MLX=1 uv run pytest -m mlx     # the real MLX model end to end
OMNIVOICE_BACKEND=fake uv run uvicorn main:app --reload     # the API with no model
```

The code lives in `omnivoice_api/`:

- `app.py`: routes.
- `batch.py`: items to engine calls, and results to the NDJSON stream.
- `voices.py`: the voice store.
- `vocab.py`: language and instruction rules.
- `engines/`: one module per backend behind a small `Engine` protocol.

Every model call runs on a single dedicated thread.

## Routes

- `GET /v1/audio/speech/capabilities` advertises model `omnivoice`, batch limits, formats, sample rate, the instruction vocabulary, non-verbal tags and the backend's generation options (`extra`).
- `POST /v1/audio/speech/batch` takes the documented JSON batch and streams `application/x-ndjson`: one `done` or `failed` line per item (in finishing order; match by `id`), `ping` lines while rendering, and a final `done` summary. Lines with the same kind of voice and the same `extra` share one model call. A call that fails, for example by running out of memory, is retried in halves, so one bad line fails alone and an oversized batch still gets through.
- `POST /v1/audio/speech` is the OpenAI-style single-line route and returns audio bytes.
- `GET /v1/audio/voices` lists the voices stored on this server.
- `POST /v1/audio/voices` makes a voice from multipart `name` and `samples` (plus an optional `transcript`), or from `name` and `description` for a designed voice. See [Voices](#voices).
- `DELETE /v1/audio/voices/{id}` removes a voice's files and its saved prompts.
- `GET /health` reports whether the model has loaded, and which backend is in use.

In Audiobook Studio, add an OpenAI-compatible endpoint with base URL `http://127.0.0.1:8000/v1` and model id `omnivoice` (or `OMNIVOICE_API_MODEL`).

## Voices

A voice is a set of plain files in `voices/`, which you can add, edit or delete directly. Voices created through the API are written the same way.

```
voices/                   yours
  example2.wav            a recording to clone (.wav, .flac, .mp3, .m4a, .ogg or .opus)
  example2.txt            what the recording says
  narrator.json           {"description": "female, british accent"}: a designed voice
prompts/                  generated cache, safe to delete
  example2.safetensors    the recording encoded once by MLX (Mac)
  example2.pt             the same by torch (WSL): upstream's VoiceClonePrompt file
```

A voice's id is its file name made URL-safe, so `Old Narrator.wav` becomes `old-narrator`. An optional `<name>.json` can add `name`, `description`, `gender` and `language` to any voice. Files added while the server is running are picked up on the next request.

**Voice prompts.** The first time a cloned voice is used, its recording is encoded into a prompt in `prompts/` (about a second). After that the prompt is reused, and loading it after a restart takes milliseconds.
- Each prompt stores a fingerprint of the recording and transcript it came from. Replacing the recording or editing the transcript encodes it again once.
- The torch backend writes OmniVoice's own `VoiceClonePrompt` format, so `VoiceClonePrompt.load("prompts/example2.pt")` works in any OmniVoice script.
- mlx-audio has no prompt file format of its own, so the MLX backend saves `.safetensors` (no torch needed on the Mac).
- The two backends don't share prompt files; each encodes once from the same recording. The MLX backend follows upstream's preprocessing of the recording (silence trimming, loudness, transcript punctuation), so both produce the same kind of prompt.

**Cloned voices.** Use a 3–10 second recording, as OmniVoice recommends; longer ones render more slowly. When uploading through the API, only the first sample is used. If there's no transcript, the server transcribes the recording and saves the text as `<name>.txt` for you to check. The torch backend transcribes with Whisper; MLX uses `OMNIVOICE_TRANSCRIBE_MODEL`, which downloads about 1 GB on first use.

Voices from earlier versions of this server, which used one folder per voice, are converted to this layout at startup.

**Designed voices.** OmniVoice only understands a fixed set of voice-design words: `male`/`female`, an age (`child` … `elderly`), a pitch (`very low pitch` … `very high pitch`), `whisper`, an English accent (`british accent`, …) or a Chinese dialect. A `description` must use only these words, comma-separated, for example `female, low pitch, british accent`. The server refuses anything else and lists the valid words.

**Line `instructions`.** These are merged over the voice's own words, with the line winning within a category. Free-form delivery notes such as "tired, flat" are ignored rather than failing the line.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/voices \
  -F "name=Example Voice" -F "samples=@reference.wav" \
  -F "transcript=The words spoken in the reference recording."
```

```bash
curl http://127.0.0.1:8000/v1/audio/speech/batch -H "Content-Type: application/json" -d '{
  "model": "omnivoice", "response_format": "wav", "extra": {"num_step": 16},
  "items": [
    {"id": "ch1-l1", "input": "Hello there.", "voice": "example-voice", "language": "en"},
    {"id": "ch1-l2", "input": "How are you?", "voice": "example-voice", "speed": 1.2}
  ]}'
```

`wav`, `flac` and raw 16-bit little-endian mono `pcm` are always available. If `ffmpeg` is on `PATH`, `mp3` and `opus` are too. Run one Uvicorn worker per GPU: each worker loads its own copy of the model.

## Configuration

Settings come from `OMNIVOICE_*` environment variables, then `.env` (copy `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMNIVOICE_BACKEND` | `auto` | `auto`, `torch`, `mlx` or `fake` |
| `OMNIVOICE_MODEL` | per backend | Checkpoint id or local directory |
| `OMNIVOICE_API_MODEL` | `omnivoice` | Model id in capabilities and requests |
| `OMNIVOICE_API_KEY` | empty | Optional Bearer token |
| `OMNIVOICE_VOICES_DIR` | `voices` | Your voices (recordings, transcripts, descriptions) |
| `OMNIVOICE_PROMPTS_DIR` | `prompts` | Encoded voice prompts (a cache) |
| `OMNIVOICE_MAX_BATCH_ITEMS` | `16` | Items per batch request |
| `OMNIVOICE_ENGINE_BATCH_SIZE` | `0` | Lines per model call; each call's lines stream back as it finishes. `0` renders a request's lines in one call. `2` was fastest on an M1 Air |
| `OMNIVOICE_MAX_BATCH_CHARS` | `12000` | Combined input characters per batch |
| `OMNIVOICE_MAX_ITEM_CHARS` | `1500` | Characters in one item |
| `OMNIVOICE_MAX_UPLOAD_MB` | `50` | Largest reference recording |
| `OMNIVOICE_DEFAULT_NUM_STEPS` | `32` | Default diffusion steps |
| `OMNIVOICE_PING_SECONDS` | `10` | Keepalive interval while rendering |
| `OMNIVOICE_DEVICE` | `auto` | torch: inference device |
| `OMNIVOICE_ENABLE_FLASHINFER` | `auto` | torch: `true`, `false`, or `auto` (on in WSL with CUDA) |
| `OMNIVOICE_TRANSCRIBE_MODEL` | `mlx-community/Qwen3-ASR-0.6B-8bit` | mlx: speech-to-text model for transcript-less samples |

`omnivoice_api/vendor/` holds OmniVoice's language map and voice-design tables, copied unchanged (Apache-2.0).
