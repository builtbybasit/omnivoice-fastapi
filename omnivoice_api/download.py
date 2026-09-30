"""Download the configured model before the server starts: ``python -m omnivoice_api.download``.

Run by the start scripts, so a slow or stalled download fails (and resumes on the next run) in the
terminal, rather than leaving a started server stuck at "loading". Files already in the Hugging
Face cache are not fetched again; with no network, a model already cached is used as it is.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

from .config import Settings

ATTEMPTS = 5


def main() -> int:
    settings = Settings()
    model = settings.resolved_model
    if settings.resolved_backend == "fake" or Path(model).expanduser().is_dir():
        return 0
    for attempt in range(1, ATTEMPTS + 1):
        try:
            print(f"Checking model {model} (downloads what is missing)", flush=True)
            snapshot_download(model)
            return 0
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            # Also raised for gated or private repos without a token: retrying cannot help.
            print(f"{model}: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Download attempt {attempt}/{ATTEMPTS} failed: {exc}", file=sys.stderr)
            if attempt < ATTEMPTS:
                time.sleep(10)
    try:
        snapshot_download(model, local_files_only=True)
    except Exception:
        print(f"Could not download {model}; rerun to resume.", file=sys.stderr)
        return 1
    print(f"Could not reach Hugging Face; using the cached copy of {model}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
