"""Find the fastest OMNIVOICE_ENGINE_BATCH_SIZE for this machine's backend (the server's own).

    uv run python scripts/benchmark.py --voice example2           # a cloned voice, its saved prompt
    uv run python scripts/benchmark.py                             # a designed voice (no recording)
    uv run python scripts/benchmark.py --text-file chapter.txt     # your own text, one item a line
    uv run python scripts/benchmark.py --find-max                  # the largest batch that fits

Each row renders the same request (--lines lines) the way the server does: split into model calls
of the batch size. So rows compare directly:

- speed: seconds of audio per second of wall time (higher is faster; 1.0x keeps pace with playback)
- vs 1: against rendering the lines one at a time; below 1.0x the batch is slower
- first line: how long a client waits for the first streamed line
- padding: work spent on silence, since every line in a call is padded to the longest one
  ("packed" with FlashInfer, which runs the lines end to end without padding)

Text is audiobook narration; --text mixed (the default) varies line length as books do. Each case
is timed after a warm-up call. --find-max doubles one call's lines until it fails (usually out of
memory), then narrows down to the largest that fits.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

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
# Sentences (about 115 characters each) per line, repeated down the request.
SENTENCES_PER_LINE = {"short": [1], "long": [4], "mixed": [1, 4, 1, 2, 1, 3, 1, 1]}

console = Console()


@dataclass
class Result:
    calls: int
    wall: float  # every call
    first: float  # the first call: when a client gets its first line
    audio: float  # seconds rendered
    padding: float  # share of the work spent padding lines to their call's longest
    peak: float | None  # GB


def request_texts(kind: str, path: Path | None, count: int) -> list[str]:
    if path:
        source = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        source = [line for line in source if line]
        if not source:
            raise SystemExit(f"{path} has no text")
        return list(itertools.islice(itertools.cycle(source), count))
    sentences, pattern = itertools.cycle(TEXTS), itertools.cycle(SENTENCES_PER_LINE[kind])
    return [" ".join(itertools.islice(sentences, next(pattern))) for _ in range(count)]


def device_name(backend: str) -> str:
    if backend == "torch":
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            return f"{gpu.name} ({gpu.total_memory / 1e9:.1f} GB)"
        return "CPU"
    return platform.processor() or platform.machine()


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


def free_memory(backend: str) -> None:
    """Release what a failed call left cached, so the next size starts from a clean slate."""
    gc.collect()
    if backend == "mlx":
        import mlx.core as mx

        mx.clear_cache()
    elif backend == "torch":
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def coloured(text: str, good: bool, bad: bool) -> str:
    return f"[green]{text}[/]" if good else f"[red]{text}[/]" if bad else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-sizes", default="1,2,4,8",
                        help="OMNIVOICE_ENGINE_BATCH_SIZE values to compare")
    parser.add_argument("--lines", type=int, help="lines per request (default: the largest batch)")
    parser.add_argument("--steps", default="16,32")
    parser.add_argument("--text", choices=SENTENCES_PER_LINE, default="mixed",
                        help="line lengths: short ~115 chars, long ~460, mixed both")
    parser.add_argument("--text-file", type=Path, help="your own text, one item a line")
    parser.add_argument("--voice", help="a cloned voice from the voices directory, by id")
    parser.add_argument("--reference", type=Path, help="a recording to clone, encoded afresh")
    parser.add_argument("--transcript", help="what --reference says (else it is transcribed)")
    parser.add_argument("--repeats", type=int, default=1, help="runs per case; the fastest counts")
    parser.add_argument("--find-max", action="store_true",
                        help="find the largest batch that fits instead of timing --batch-sizes")
    parser.add_argument("--max-batch", type=int, default=256, help="where --find-max stops")
    args = parser.parse_args()
    batch_sizes = sorted({int(size) for size in args.batch_sizes.split(",")})
    steps = [int(step) for step in args.steps.split(",")]
    count = args.lines or batch_sizes[-1]
    texts = request_texts(args.text, args.text_file, max(count, args.max_batch))

    settings = Settings()
    backend = settings.resolved_backend
    with console.status(f"Loading {settings.resolved_model} ({backend})…"):
        started = time.perf_counter()
        engine = load_engine(settings)
    lengths = [len(text) for text in texts[:count]]
    console.print(Panel.fit(
        f"[bold]{device_name(backend)}[/] · {platform.platform()}\n"
        f"backend [cyan]{backend}[/] · model [cyan]{settings.resolved_model}[/] · "
        f"loaded in {time.perf_counter() - started:.1f} s\n"
        f"request: [bold]{count} lines[/] of {args.text_file or args.text + ' text'}, "
        f"{min(lengths)}–{max(lengths)} chars (avg {sum(lengths) // len(lengths)})",
        title="OmniVoice benchmark",
    ))

    voices = {}
    if args.voice:
        store = VoiceStore(settings.voices_dir, settings.prompts_dir)
        voice = store.get(args.voice)
        if voice is None or not voice.clone:
            parser.error(f"no cloned voice '{args.voice}' in {store.root}")
        saved = store.prompts / f"{voice.id}{engine.prompt_suffix}"
        how = "loaded from " + saved.name if saved.is_file() else "encoded (none saved yet)"
        started = time.perf_counter()
        voices[args.voice] = (store.prompt(voice, engine), None)
        console.print(f"Voice [bold]{voice.id}[/]: prompt {how} in "
                      f"{time.perf_counter() - started:.3f} s")
    if args.reference:
        voices["cloned"] = (engine.encode_prompt(args.reference, args.transcript)[0], None)
    if not voices:
        voices["designed"] = (None, "female, british accent")

    def run(size: int, batch: int, num_step: int, prompt, instruct) -> Result:
        """The first ``size`` texts as the server renders them: calls of ``batch`` lines."""
        options = resolve_options(engine.options, {"num_step": num_step})
        lines = [Line(text, "en", instruct, 1.0, prompt) for text in texts[:size]]
        reset_peak_memory(backend)
        calls, wall, first, audio, work = 0, 0.0, 0.0, 0.0, 0.0
        for start in range(0, len(lines), batch):
            began = time.perf_counter()
            audios = engine.generate(lines[start : start + batch], options)
            took = time.perf_counter() - began
            seconds = [len(samples) / engine.sample_rate for samples in audios]
            calls, wall, first = calls + 1, wall + took, first or took
            audio, work = audio + sum(seconds), work + len(seconds) * max(seconds)
        return Result(calls, wall, first, audio, 1 - audio / work if work else 0.0,
                      peak_memory_gb(backend))

    with console.status("Warming up (kernel compilation, caches)…"):
        run(1, 1, 8, *next(iter(voices.values())))

    table = Table(header_style="bold", title_justify="left")
    for name in ("voice", "steps", "batch", "calls", "wall s", "audio s", "speed", "vs 1",
                 "first line s", "padding", "peak GB"):
        table.add_column(name, justify="left" if name == "voice" else "right")
    results: list[tuple[str, int, int, Result]] = []
    baseline: dict[tuple[str, int, int], float] = {}  # (voice, steps, lines) -> batch-1 wall

    def measure(voice: str, num_step: int, size: int, batch: int, prompt,
                instruct) -> Result | None:
        """Add one row; None if a call failed."""
        case = f"{voice} · {num_step} steps · batch {batch}"
        status.update(f"{case} · {size} lines…")
        try:
            result = min((run(size, batch, num_step, prompt, instruct)
                          for _ in range(args.repeats)), key=lambda r: r.wall)
        except Exception as exc:  # usually out of memory; the message says
            free_memory(backend)
            error = (str(exc).splitlines() or [type(exc).__name__])[0][:120]
            console.print(f"[red]✗ {case} failed:[/] {escape(error)}")
            table.add_row(voice, str(num_step), str(batch), "", "", "", "[bold red]failed")
            return None
        speed = result.audio / result.wall
        if batch == 1:
            baseline[voice, num_step, size] = result.wall
        one_by_one = baseline.get((voice, num_step, size))
        versus = one_by_one / result.wall if one_by_one else None
        peak = f"{result.peak:.2f}" if result.peak is not None else "–"
        table.add_row(
            voice, str(num_step), str(batch), str(result.calls), f"{result.wall:.1f}",
            f"{result.audio:.1f}", coloured(f"{speed:.2f}x", speed >= 1, speed < 1),
            coloured(f"{versus:.2f}x", versus > 1.05, versus < 0.95) if versus else "",
            f"{result.first:.1f}",
            coloured(f"{result.padding:.0%}", result.padding < 0.25, result.padding >= 0.5)
            if getattr(engine, "pads_batches", True) else "[dim]packed[/]",
            peak,
        )
        results.append((voice, num_step, batch, result))
        console.print(f"[green]✓[/] {case}: {speed:.2f}x real time, first line after "
                      f"{result.first:.1f} s")
        return result

    found = []
    with console.status("Starting…") as status:
        for voice, (prompt, instruct) in voices.items():
            for num_step in steps:
                if not args.find_max:
                    for batch in batch_sizes:
                        measure(voice, num_step, count, batch, prompt, instruct)
                    continue

                def fits(size: int) -> bool:
                    return measure(voice, num_step, size, size, prompt, instruct) is not None

                good, size = 0, 1
                while size <= args.max_batch and fits(size):
                    good, size = size, size * 2
                bad = size
                while size <= args.max_batch and bad - good > 1:
                    mid = (good + bad) // 2
                    good, bad = (mid, bad) if fits(mid) else (good, mid)
                found.append((voice, num_step, good))
    console.print()
    console.print(table)

    if args.find_max:
        for voice, num_step, size in found:
            limit = f" (stopped at --max-batch {args.max_batch})" if size == args.max_batch else ""
            console.print(f"Largest batch for [bold]{voice}[/] at {num_step} steps: "
                          f"[bold green]{size} lines[/]{limit}")
        console.print(f"[dim]The longest line here is {max(map(len, texts))} characters; longer "
                      "lines need more memory. Set OMNIVOICE_ENGINE_BATCH_SIZE at or below this.")
        return

    best = {}
    for (voice, num_step), rows in itertools.groupby(results, key=lambda row: row[:2]):
        rows = list(rows)
        fastest = max(row[3].audio / row[3].wall for row in rows)
        # Within 5% is noise; of those, the smallest batch streams its first line soonest.
        close = [row for row in rows if row[3].audio / row[3].wall >= 0.95 * fastest]
        *_, batch, result = min(close, key=lambda row: row[2])
        best.setdefault(num_step, batch)
        one_by_one = baseline.get((voice, num_step, count))
        versus = f", {one_by_one / result.wall:.2f}x one line at a time" if one_by_one else ""
        console.print(f"[bold]{voice} · {num_step} steps:[/] batch [bold]{batch}[/] at "
                      f"{result.audio / result.wall:.2f}x real time{versus}; first line after "
                      f"{result.first:.1f} s")
    if best:
        num_step = settings.default_num_steps if settings.default_num_steps in best else steps[-1]
        console.print(Panel.fit(
            f"Set [bold green]OMNIVOICE_ENGINE_BATCH_SIZE={best[num_step]}[/] in .env "
            f"(at {num_step} steps)\n[dim]The smallest batch within 5% of the fastest: about as "
            "fast, and the first line streams sooner.[/]",
            border_style="green",
        ))

if __name__ == "__main__":
    main()
