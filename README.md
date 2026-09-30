# OmniVoice FastAPI

A local FastAPI server implementing the [Audiobook Studio speech batch API](https://github.com/builtbybasit/audiobook-studio/blob/main/docs/speech-batch-api.md) on top of the OmniVoice inference flow used by `omnivoice-test`.

## Start the server

Run the API inside WSL2, using Python 3.12 and `uv`. Keep the working copy under the Linux home directory rather than `/mnt/c`; WSL GPU workloads and dependency environments work better on the Linux filesystem.

From an Ubuntu WSL shell, copy the API source into your Linux home directory:

```bash
mkdir -p ~/projects/omnivoice-fastapi
cd ~/projects/omnivoice-fastapi
cp /mnt/c/Documents/Projects/omnivoice-fastapi/{main.py,pyproject.toml,uv.lock,.python-version,.env.example,.gitignore,README.md} .
mkdir -p scripts
cp /mnt/c/Documents/Projects/omnivoice-fastapi/scripts/*.sh scripts/
cp -n .env.example .env
bash scripts/setup-wsl-flashinfer.sh
bash scripts/run-wsl.sh
```

The setup script runs `uv sync --no-install-project` and first installs `flashinfer-python==0.6.15.post1` plus `flashinfer-jit-cache==0.6.15.post1+cu128` through FlashInfer's cu128 index. If the large JIT-cache wheel cannot finish through that index, setup reuses a SHA-256-verified local wheel or downloads it with resume support from the official GitHub release, verifies it, and installs it with `uv`. The run script verifies WSL CUDA/FlashInfer imports and starts the API at `http://127.0.0.1:8000`. Windows apps can call it at `http://localhost:8000/v1`; interactive API docs are at `http://localhost:8000/docs`. The model loads once when the server starts and uses `k2-fsa/OmniVoice` by default. On first use, missing model files download to Hugging Face's WSL cache (`~/.cache/huggingface/hub` by default) and are reused on later starts.

The API defaults to requiring FlashInfer and fails startup if it is missing or unavailable. For later dependency refreshes, use `uv sync --inexact --no-install-project` so uv preserves the FlashInfer packages installed from the separate index. Set `OMNIVOICE_ENABLE_FLASHINFER=false` only when you deliberately want to run the slower baseline path.

For access from another machine, bind Uvicorn to `0.0.0.0` and set a long random `OMNIVOICE_API_KEY` first. Requests then need `Authorization: Bearer <key>`. Leave the key empty for loopback-only local use.

## Routes

- `GET /v1/audio/speech/capabilities` advertises model `omnivoice`, batch limits, formats, sample rate, language/speed/instruction support, and OmniVoice generation options.
- `POST /v1/audio/speech/batch` accepts the documented JSON batch body and streams `application/x-ndjson`. Every item receives one `done` or `failed` line, and a final `done` summary. Results may arrive out of input order; match by `id`.
- `POST /v1/audio/speech` provides the OpenAI-style single speech route and returns audio bytes.
- `GET /v1/audio/voices` lists voices stored on this server.
- `POST /v1/audio/voices` creates a voice from multipart `name`, `samples` and `transcript`, or from `name` and `description` for an instruction-designed voice. The current OmniVoice integration uses the first uploaded sample to create the clone prompt.
- `GET /health` reports whether the model finished loading.

In Audiobook Studio, add an OpenAI-compatible endpoint with base URL `http://127.0.0.1:8000/v1` and model id `omnivoice` (or the value configured in `OMNIVOICE_API_MODEL`).

Voice ids are generated from names and voice files are kept under `voices/`. Clone voice prompt tensors are created at upload time and reused for later requests. A designed voice stores its description as the OmniVoice instruction.

Example voice creation:

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/audio/voices `
  -F "name=Example Voice" `
  -F "samples=@C:\path\to\reference.wav" `
  -F "transcript=The words spoken in the reference recording."
```

Example batch request:

```powershell
$body = @{
  model = "omnivoice"
  response_format = "wav"
  extra = @{ num_step = 16 }
  items = @(
    @{ id = "chapter-1-line-1"; input = "Hello there."; voice = "example-voice"; language = "en" },
    @{ id = "chapter-1-line-2"; input = "How are you?"; voice = "example-voice"; instructions = "Warm and curious." }
  )
} | ConvertTo-Json -Depth 8

Invoke-WebRequest -Method Post `
  -Uri http://127.0.0.1:8000/v1/audio/speech/batch `
  -ContentType "application/json" `
  -Headers @{ Accept = "application/x-ndjson" } `
  -Body $body -OutFile batch.ndjson
```

`wav`, `flac`, and raw 16-bit little-endian mono `pcm` are available by default. If `ffmpeg` is on `PATH`, the server also advertises and encodes `mp3` and `opus`. The generation API is serialized through one in-process model lock, so run one Uvicorn worker per GPU process; multiple workers each load a separate model copy.

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | Model checkpoint or local checkpoint directory |
| `OMNIVOICE_API_MODEL` | `omnivoice` | Model id returned in capabilities and expected in requests |
| `OMNIVOICE_DEVICE` | `auto` | Inference device; FlashInfer mode requires CUDA |
| `OMNIVOICE_ENABLE_FLASHINFER` | `true` | Require and patch in OmniVoice's FlashInfer integration |
| `OMNIVOICE_API_KEY` | empty | Optional Bearer token |
| `OMNIVOICE_VOICES_DIR` | `voices` | Persistent voice directory |
| `OMNIVOICE_MAX_BATCH_ITEMS` | `16` | Maximum items per request |
| `OMNIVOICE_MAX_BATCH_CHARS` | `12000` | Maximum combined input characters |
| `OMNIVOICE_MAX_ITEM_CHARS` | `1500` | Maximum characters in one item |
| `OMNIVOICE_DEFAULT_NUM_STEPS` | `32` | Default OmniVoice diffusion steps |
| `OMNIVOICE_PING_SECONDS` | `10` | Keepalive interval during long generation |

Batch items must name a stored `voice`. Item-specific `extra` options override batch-level options; generation options are grouped so compatible lines share an OmniVoice batch call. Individual invalid items fail independently. The batch response remains replay-safe because speech generation does not mutate voice state.
