"""Measure generation speed on this machine's backend (the one the server would use).

    uv run python scripts/benchmark.py
    uv run python scripts/benchmark.py --voice example2 --batch-sizes 1,4,8 --steps 32

Speed is the real-time factor: seconds of audio rendered per second of wall time (higher is
faster; 1.0x keeps pace with playback). Each case is timed after a warm-up call. Text is typical
audiobook narration, about 150 characters (roughly 10 s of speech) a line.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnivoice_api.config import Settings  # noqa: E402
from omnivoice_api.engines import Line, load_engine, resolve_options  # noqa: E402
from omnivoice_api.voices import VoiceStore  # noqa: E402

TEXTS = [
    "The river was quiet that morning, and nobody on the boat spoke until the fog began to lift "
    "from the water.",
    "She counted the coins twice, then a third time, as if the number might change if she only "
    "looked at it long enough.",
    "By the time the lamps were lit along the harbour, the rumour had already reached every "
    "tavern on the east side.",
    "He had promised himself he would not go back to that house, and yet here he was, standing "
    "at the gate in the rain.",
    "The letter was short, written in a careful hand, and it asked for nothing except that she "
    "come home before winter.",
    "Somewhere below deck a dog was barking, and the sound carried strangely through the old "
    "timbers of the ship.",
    "They walked in silence for most of the afternoon, each of them waiting for the other to say "
    "what they were both thinking.",
    "When the storm finally broke, it broke all at once, and the whole valley seemed to hold its "
    "breath before the first thunder.",
]


def peak_memory_gb(backend: str) -> float | None:
    if backend == "mlx":
        import mlx.core as mx

        return mx.get_peak_memory() / 1e9
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    return None


def reset_peak_memory(backend: str) -> None:
    if backend == "mlx":
        import mlx.core as mx

        mx.reset_peak_memory()
    elif backend == "torch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--steps", default="16,32")
    parser.add_argument("--voice", help="a stored cloned voice to benchmark too")
    parser.add_argument("--reference", type=Path, help="a recording to also benchmark cloning")
    parser.add_argument("--transcript", help="what --reference says (else it is transcribed)")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    steps = [int(step) for step in args.steps.split(",")]

    settings = Settings()
    backend = settings.resolved_backend
    print(f"{platform.machine()} {platform.platform()} · backend {backend} · "
          f"model {settings.resolved_model}")
    started = time.perf_counter()
    engine = load_engine(settings)
    print(f"Model loaded in {time.perf_counter() - started:.1f} s\n")

    voices = {"designed": (None, "female, british accent")}
    if args.voice:
        store = VoiceStore(settings.voices_dir, settings.prompts_dir)
        voice = store.get(args.voice)
        if voice is None or not voice.clone:
            parser.error(f"no cloned voice '{args.voice}' in {store.root}")
        voices[args.voice] = (store.prompt(voice, engine), None)
    if args.reference:
        voices["cloned"] = (engine.encode_prompt(args.reference, args.transcript)[0], None)

    def render(size: int, num_step: int, prompt, instruct) -> tuple[float, float]:
        lines = [Line(TEXTS[i % len(TEXTS)], "en", instruct, 1.0, prompt) for i in range(size)]
        options = resolve_options(engine.options, {"num_step": num_step})
        started = time.perf_counter()
        audios = engine.generate(lines, options)
        return time.perf_counter() - started, sum(len(a) for a in audios) / engine.sample_rate

    render(1, 8, *voices["designed"])  # warm-up: kernel compilation, caches

    print(f"{'voice':<9} {'steps':>5} {'batch':>5} {'wall s':>7} {'audio s':>8} {'speed':>7} "
          f"{'s/line':>7} {'peak GB':>8}")
    for voice, (prompt, instruct) in voices.items():
        for num_step in steps:
            for size in batch_sizes:
                reset_peak_memory(backend)
                wall = audio = 0.0
                for _ in range(args.repeats):
                    w, a = render(size, num_step, prompt, instruct)
                    wall, audio = wall + w, audio + a
                peak = peak_memory_gb(backend)
                print(f"{voice:<9} {num_step:>5} {size:>5} {wall:>7.1f} {audio:>8.1f} "
                      f"{audio / wall:>6.2f}x {wall / (size * args.repeats):>7.2f} "
                      f"{peak if peak is not None else float('nan'):>8.2f}", flush=True)


if __name__ == "__main__":
    main()
