#!/usr/bin/env python3
"""
conf_notes.py — batch pipeline: scans an input folder for recordings and,
one at a time, turns each into a transcript PDF, a notes markdown file, a
notes HTML file, and a notes PDF (4 files per recording).

Transcription is done locally with mlx-whisper (no cloud calls, no API
key needed). The raw transcript is first lightly cleaned up by pi (also
forced to run on a local model — see CLEANUP_MODEL below — so this pass
stays fully offline too), then that cleaned transcript feeds a 4-pass pi
pipeline:
  0. cleanup      — de-dupe/de-hallucinate raw whisper output, drop filler
                    ("grazie grazie"-style repeats), fix obvious typos —
                    no content removed, nothing paraphrased. Runs on
                    CLEANUP_MODEL (local, via ollama), not --model-fast.
  1. extract     — pull claims/topics/quotes from the cleaned transcript only
  2. enrich       — look up each topic across free, no-key sources
  3. synthesize   — combine into full academic notes, nothing dropped
  4. verify       — flag (never delete) anything not traceable to a source

Only pass 1 (extract) and pass 0 (cleanup) ever see the transcript text
itself — passes 2-4 work from extract.md/background.md/notes.md, which
already contain everything they need, so the (much longer) transcript
isn't re-sent to pi on every later pass.

Folder layout (created automatically next to the script if missing):
  input/   — drop recordings here; processed files are moved to
             input/processed/ so reruns don't redo them
  output/  — final "<name>-notes.md", "<name>-notes.html",
             "<name>-notes.pdf", and "<name>-transcript.pdf" land here
             (4 files per recording)
  logs/    — one log file per run, named yyyy-MM-dd_hh-mm_<name>.log

Requirements:
    pip install mlx-whisper weasyprint markdown
    brew install cairo pango gdk-pixbuf libffi ffmpeg   # WeasyPrint + mlx-whisper native deps
    pi CLI already installed and configured with your provider fallbacks

macOS note: if WeasyPrint fails to import with a library-loading error,
Homebrew's libs usually just aren't on the dynamic linker's search path —
see check_environment()'s error message for the exact fix (one `export`).

Usage:
    python conf_notes.py                     # scan INPUT_DIR, process all found
    python conf_notes.py --file talk.m4a      # process a single file instead
    python conf_notes.py --file talk.m4a --from-stage synthesize   # debug/rerun
    python conf_notes.py -v                   # debug-level console logging
"""

from __future__ import annotations
import argparse
import html
import logging
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Optional third-party imports, resolved once at startup ----------------
# Kept optional (not a hard `import` at call time) so a missing backend
# degrades gracefully instead of crashing the whole script.
# WeasyPrint in particular can fail with OSError (not ImportError) when it
# can't dlopen its native libraries (pango/cairo/gdk-pixbuf) — see
# check_environment() below for the macOS fix.
# Pin the HuggingFace cache to a fixed folder next to the script *before*
# mlx_whisper/huggingface_hub are imported. Without this, the cache
# location depends on $HOME at the time of the call — if this script is
# ever run from cron/launchd/a different shell with a different (or
# unset) $HOME, huggingface_hub resolves to a different cache dir and it
# looks like the model gets re-downloaded every day even though nothing
# is actually wrong with the download itself.
SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".hf-cache"))

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

try:
    import weasyprint
except (ImportError, OSError) as _weasyprint_error:
    weasyprint = None
    WEASYPRINT_IMPORT_ERROR = str(_weasyprint_error)
else:
    WEASYPRINT_IMPORT_ERROR = None

try:
    import markdown
except ImportError:
    markdown = None
# -----------------------------------------------------------------------

PROMPTS_DIR = SCRIPT_DIR / "prompts"
WORK_DIR = SCRIPT_DIR / "work"  # scratch space for intermediate files per run

# ---------------------------------------------------------------------------
# Self-contained: these live next to the script and are created automatically
# if missing. Override with different absolute paths here if you'd rather
# keep recordings/output somewhere else (e.g. iCloud Drive, an external disk).
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"
LOG_DIR = SCRIPT_DIR / "logs"
# ---------------------------------------------------------------------------

STAGES = ["transcribe", "cleanup", "extract", "enrich", "synthesize", "verify", "pdf"]
WHISPER_MODEL_DEFAULT = "mlx-community/whisper-large-v3-turbo"
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".mp4", ".aac", ".flac", ".ogg", ".mov"}

# Pass 0 (cleanup) is light, mechanical work (de-dupe/de-hallucinate/typo-fix
# on the raw whisper output, no real reasoning about content) so it's forced
# to run on a local ollama model instead of --model-fast/PI_MODEL_FAST — it
# never leaves the machine and doesn't spend cloud-model budget on a pass
# that doesn't need a strong model. See stage_cleanup() below, which passes
# this constant to pi regardless of what --model-fast is set to.
GEMINI_MODEL_1 = "gemini-flash-lite-latest"  # "gemma4:e4b-mlx"
GEMINI_MODEL_2 = "gemini-3.1-flash-lite"  # "gemma4:e4b-mlx"
LOCAL_MODEL = "gemma4:e4b-mlx"
PI_TIMEOUT_SECONDS = int(os.environ.get("PI_TIMEOUT_SECONDS", "1800"))
BATCH_SIZE = 80

# Files each stage needs already present when resuming with --from-stage
# (irrelevant for a normal full run, where the prior stage just wrote them).
#
# Note: only "cleanup" and "extract" need the (raw/cleaned) transcript text.
# From "synthesize" onward, extract.md's "Speaker's claims & arguments"
# section is already a compressed-but-complete sentence-by-sentence record
# of the talk, so the full transcript is not re-attached to later pi calls
# (it's long, and re-sending it on every pass just burns tokens for no
# extra information pi doesn't already have via extract.md).
STAGE_REQUIRES = {
    "cleanup": ["transcript-raw.txt"],
    "extract": ["transcript.txt"],
    "enrich": ["extract.md"],
    "synthesize": ["extract.md", "background.md"],
    "verify": ["notes.md", "extract.md", "background.md"],
    "pdf": ["notes.md", "transcript.txt"],
}

_ITALIAN_MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
                   "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]

# Date patterns to look for in a recording's filename, tried in order.
# Each tuple is (regex, group order). Deliberately conservative (requires
# 4-digit years, non-digit boundaries) to avoid false positives on
# filenames that just happen to contain other numbers.
_DATE_PATTERNS = [
    (re.compile(r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})(?!\d)"), "ymd"),
    (re.compile(r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})(?!\d)"), "dmy"),
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"), "ymd"),
]


def _format_italian_date(y: int, m: int, d: int) -> str | None:
    if 1 <= m <= 12 and 1 <= d <= 31 and 2000 <= y <= 2100:
        return f"{d} {_ITALIAN_MONTHS[m - 1]} {y}"
    return None


def _parse_extract_items(extract_path: Path) -> list[str]:
    """Return every topic/entity listed under 'Topics & concepts mentioned'
    and 'Named entities' in extract.md."""
    lines = extract_path.read_text().splitlines()
    items = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Topics & concepts mentioned") or stripped.startswith("## Named entities"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("## "):
                in_section = False
                continue
            if stripped.startswith("- "):
                items.append(stripped[2:].strip())
    return items


def derive_date_from_filename(audio: Path) -> str:
    """Best-effort 'Data' value for notes.md's header.

    The speaker rarely states the date out loud in a conference talk, so
    asking pi to find it in the transcript fails most of the time (see
    extract.md's Session metadata section usually saying "not stated").
    We derive it ourselves instead and hand it to pi as a fixed value to
    drop in, rather than a lookup task for it to attempt and get wrong.

    Tries common date patterns in the filename first (most reliable,
    since recordings are typically named by whoever made them, e.g.
    "2026-03-05_convegno.m4a"), then falls back to the file's
    last-modified time, clearly labelled as such since it's a weaker
    signal (could be a copy/export date rather than the talk's date).
    """
    name = audio.stem
    for pattern, order in _DATE_PATTERNS:
        m = pattern.search(name)
        if not m:
            continue
        y, mo, d = (m.group(1), m.group(2), m.group(3)) if order == "ymd" else (m.group(3), m.group(2), m.group(1))
        formatted = _format_italian_date(int(y), int(mo), int(d))
        if formatted:
            return formatted

    try:
        mtime = datetime.fromtimestamp(audio.stat().st_mtime)
        formatted = _format_italian_date(mtime.year, mtime.month, mtime.day)
        if formatted:
            return f"{formatted} (dedotta dalla data del file, non dal nome file)"
    except OSError:
        pass

    return "non determinata"


logger = logging.getLogger("conf_notes")


def setup_logging(logfile: Path, verbose: bool) -> None:
    """Log to the console and to a dedicated file for this run."""
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    logger.addHandler(file_handler)


def discover_audio_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name)


# Pi's `read` tool truncates (silently, per line) at roughly this many bytes
# — see run_pi()'s docstring. Kept as its own constant (not folded into
# run_pi) since it's a property of pi's read tool, not of how we invoke pi.
PI_READ_LINE_LIMIT_BYTES = 50_000


def ensure_no_long_lines(path: Path, max_bytes: int = PI_READ_LINE_LIMIT_BYTES) -> None:
    """Safety net: insert line breaks at whitespace so no single line in
    `path` exceeds max_bytes, without changing any word or character in the
    file — only where the newlines fall.

    Why this exists in addition to transcribe_with_local_whisper() writing
    transcript-raw.txt pre-split into short lines: that only guarantees the
    *first* file pi reads is safe. Every later stage's output is written by
    pi itself (following prompt instructions like "keep line breaks as in
    the original"), and a model — especially a small local one — isn't
    guaranteed to honor that. If e.g. pass 0's cleanup model flattens
    transcript.txt back into one long paragraph while "cleaning" it, pass 1
    would silently see only a truncated prefix of the talk, with no error
    raised anywhere. Called on every stage's output file right after
    _require() confirms it exists, so this holds regardless of what the
    model actually did with formatting.
    """
    text = path.read_text()
    lines = text.split("\n")
    if all(len(line.encode("utf-8")) <= max_bytes for line in lines):
        return  # already safe — nothing to rewrite

    out_lines = []
    for line in lines:
        if len(line.encode("utf-8")) <= max_bytes:
            out_lines.append(line)
            continue
        # Break only at existing spaces, never mid-word, so this can't
        # alter or split any actual word/character in the transcript.
        current = ""
        for word in line.split(" "):
            candidate = f"{current} {word}" if current else word
            if current and len(candidate.encode("utf-8")) > max_bytes:
                out_lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            out_lines.append(current)

    path.write_text("\n".join(out_lines))
    logger.debug("  rewrapped long line(s) in %s to stay under pi's ~%dKB read limit",
                 path.name, max_bytes // 1000)


def run_pi(
        *,
        attachments: list[Path],
        prompt_file: Path,
        tools: str | None,
        model: str | None,
        logfile: Path,
        cwd: Path,
        substitutions: dict[str, str] | None = None,
) -> None:
    """Invoke pi non-interactively: pi -p @file1 @file2 "<prompt>" --tools ...

    pi resolves relative file operations (like "write to extract.md")
    against its own process working directory, not against the paths of
    any @attachments — there's no --cwd flag (see earendil-works/pi#4745),
    so we set the subprocess's cwd explicitly to `workdir`. Without this,
    pi writes/reads relative filenames wherever this script happened to be
    launched from, not into workdir, and the pipeline silently looks for
    its output in the wrong place.
    """
    prompt_text = prompt_file.read_text()
    # Fill in any values the script already knows (e.g. the talk's date,
    # derived from the filename) instead of leaving pi to infer them from
    # the transcript, which is unreliable for things speakers rarely
    # state out loud.
    for key, value in (substitutions or {}).items():
        prompt_text = prompt_text.replace(key, value)

    cmd = ["pi", "-p"]
    cmd += [f"@{p}" for p in attachments]
    cmd += [prompt_text]
    cmd += [" --no-notify "]
    if tools:
        cmd += ["--tools", tools]
    if model:
        cmd += ["--model", model]
    # Extra flags for pi itself (e.g. a verbosity/debug flag so it prints
    # which provider/model in provider-fallback.json actually served each
    # request). Flag name varies by pi version/config, so it's left as an
    # opt-in env var rather than hardcoded — check `pi --help` for yours,
    # e.g.: export PI_EXTRA_ARGS="--log-level debug"
    cmd += os.environ.get("PI_EXTRA_ARGS", "").split()

    logger.info("Running pi (cwd=%s): pi -p %s ... --tools %s%s",
                cwd, " ".join(f"@{p.name}" for p in attachments),
                tools, f" --model {model}" if model else "")

    # Stream instead of subprocess.run(): the old version piped stdout
    # straight into logfile with a raw open(), bypassing the logging
    # module entirely — so those per-pass log files had no timestamps,
    # and pi's real-time activity (tool calls, retries, which provider
    # actually answered) was invisible until the whole call finished.
    # Streaming line-by-line lets us timestamp every row AND mirror it
    # into the main run log so it's visible live with -v.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=cwd, text=False, bufsize=0,
        # NOTE: this looks equivalent to stdin=subprocess.DEVNULL but is
        # NOT — that was the actual bug. pi has a confirmed, still-open
        # issue (earendil-works/pi#4303): in -p/print mode, if stdin is
        # /dev/null it emits its full response and then never exits,
        # sitting in epoll_wait forever; but if stdin is a *pipe* that's
        # closed immediately (even one that received zero bytes), it
        # exits normally right after finishing. So we must hand it a
        # closed pipe, not /dev/null, even though both "look like" EOF
        # with nothing to read.
        stdin=subprocess.PIPE,
        # New process group so we can clean up any straggler descendants
        # below (see the killpg call) without touching this script itself.
        start_new_session=True,
    )
    proc.stdin.close()

    # Read with a poll/timeout loop instead of `for line in proc.stdout`.
    # That plain form blocks until the pipe gets EOF, which requires every
    # process holding the write end open to close it — not just pi itself.
    # pi's tools (aio-websearch in particular looks like it shells out to
    # fetch/render pages) can spawn a subprocess that inherits this pipe's
    # fd and never explicitly closes it. When that happens, pi finishes,
    # prints its answer, and exits cleanly, but the pipe never sees EOF
    # because a grandchild is still holding it open — so the old loop sat
    # there forever after batch 1's very output you saw. Polling lets us
    # treat "pi's own process has exited" as the real completion signal,
    # independent of whatever else might still be holding the pipe.
    buf = b""
    with open(logfile, "w") as lf:
        started = time.monotonic()
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if ready:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if chunk:
                    buf += chunk
                    while b"\n" in buf:
                        raw, buf = buf.split(b"\n", 1)
                        line = raw.decode("utf-8", errors="replace")
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        lf.write(f"{ts} {line}\n")
                        lf.flush()
                        logger.debug("[pi] %s", line)
                    continue  # more may be buffered; keep draining first
            # No data ready right now. If pi's own process has already
            # exited, we're done — anything still holding the pipe open
            # is a leftover grandchild, not pi, so stop waiting on it.
            if proc.poll() is not None:
                break

            elapsed = time.monotonic() - started
            if elapsed >= PI_TIMEOUT_SECONDS:
                logger.error(
                    "pi timed out after %.0f seconds; aborting pipeline",
                    elapsed,
                )
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                raise TimeoutError(
                    f"pi exceeded the hard timeout of {PI_TIMEOUT_SECONDS}s"
                )
        if buf:
            line = buf.decode("utf-8", errors="replace")
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lf.write(f"{ts} {line}\n")
            logger.debug("[pi] %s", line)

    returncode = proc.returncode

    # Best-effort cleanup of any leftover descendants (e.g. a headless
    # browser/fetcher aio-websearch spawned) so they don't pile up over
    # 53 batches instead of being reaped when pi itself exited.
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    if returncode != 0:
        raise RuntimeError(
            f"pi call failed (exit {returncode}). See {logfile} for details."
        )


def _whisper_model_is_cached(whisper_model: str) -> bool:
    """Best-effort check of whether whisper_model is already in the local
    HF cache, purely so the log can say which one happened (helps confirm
    the caching fix is actually working, instead of guessing from timing)."""
    try:
        from huggingface_hub import scan_cache_dir
        cached_repos = {repo.repo_id for repo in scan_cache_dir().repos}
        return whisper_model in cached_repos
    except Exception:
        return False  # can't tell -> don't claim either way in the log


def transcribe_with_local_whisper(audio: Path, whisper_model: str) -> str:
    if mlx_whisper is None:
        raise RuntimeError("mlx-whisper not installed. Run: pip install mlx-whisper")

    if _whisper_model_is_cached(whisper_model):
        logger.debug("  Whisper model %s found in cache (%s) — no download needed",
                     whisper_model, os.environ.get("HF_HOME"))
    else:
        logger.info("  Whisper model %s not in cache (%s) — downloading once, "
                    "will be cached for future runs", whisper_model, os.environ.get("HF_HOME"))

    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=whisper_model)

    # Join whisper's own per-phrase segments with newlines instead of using
    # result["text"] (one giant single-line string for the whole recording).
    # A ~1-hour talk produces a ~50-60KB single line, which blows past pi's
    # per-line read limit (50KB) and silently truncates whatever pass reads
    # this file first (pass 0 cleanup) to just the opening minutes — with no
    # error, just a quiet content-loss bug downstream. Segment boundaries are
    # whisper's own phrase breaks, not a rewording, so this changes nothing
    # about the transcribed words, only how they're laid out on disk.
    segments = result.get("segments") or []
    if segments:
        lines = [seg.get("text", "").strip() for seg in segments]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    # Fallback if a whisper backend/version ever omits segments.
    return result["text"].strip()


def stage_transcribe(audio: Path, workdir: Path, whisper_model: str) -> None:
    logger.info("[1/7] Transcribing audio (local mlx-whisper, model=%s)", whisper_model)
    text = transcribe_with_local_whisper(audio, whisper_model)

    out = workdir / "transcript-raw.txt"
    out.write_text(text)
    ensure_no_long_lines(out)  # belt-and-suspenders alongside the segment-based split above
    logger.info("  wrote %s (%d words)", out, len(text.split()))


def stage_cleanup(workdir: Path, model_fast: str | None) -> None:
    """Light pi pass over the raw whisper output: drop duplicated/hallucinated
    lines and repeated filler ("grazie grazie", "buongiorno buongiorno", ...)
    and fix obvious mis-transcription typos — nothing summarized, nothing
    reworded, no content removed. Writes the transcript.txt that every later
    stage (extract, and the final transcript PDF) actually consumes.

    model_fast is accepted (and still used by every other stage) but
    deliberately ignored here: cleanup is forced onto the local CLEANUP_MODEL
    instead, since this pass is mechanical enough not to need a cloud model
    and runs fully offline via ollama.

    tools is deliberately "write" only, NOT "read,write": the @attachment
    mechanism already embeds transcript-raw.txt's full content inline in
    the prompt pi sends the model (confirmed via pi's own session log — the
    whole file shows up as a <file> block in the first user message, no
    size cap), so the model never needs to fetch it again. Offering `read`
    anyway invites a small local model to redundantly re-read a file it
    already has in full — and pi's `read` tool caps total output at ~50KB
    per call regardless of line length, so on a ~55KB transcript that
    self-read comes back silently truncated (~93% of the file, no loud
    error) and the model proceeds to "clean" only what it got. Removing
    `read` here closes that path outright rather than trying to make the
    truncated read safe to consume."""
    logger.info("[2/7] Pass 0: transcript cleanup (de-dupe/de-hallucinate, no content dropped, local model=%s)",
                GEMINI_MODEL_1)
    run_pi(
        attachments=[workdir / "transcript-raw.txt"],
        prompt_file=PROMPTS_DIR / "pass0-cleanup.md",
        tools="write",
        model=GEMINI_MODEL_2,
        logfile=workdir / "pass0.log",
        cwd=workdir,
    )
    _require(workdir / "transcript.txt", "Pass 0")
    logger.info("  wrote %s", workdir / "transcript.txt")


def stage_extract(workdir: Path, model_fast: str | None) -> None:
    logger.info("[3/7] Pass 1: extraction (transcript-only, no outside knowledge)")
    # tools="write" only, same reasoning as stage_cleanup: transcript.txt is
    # already fully inlined via @attachment, and this stage never needs to
    # open any other file, so `read` would only ever be a redundant,
    # truncation-prone re-fetch of content the model already has.
    run_pi(
        attachments=[workdir / "transcript.txt"],
        prompt_file=PROMPTS_DIR / "pass1-extract.md",
        tools="write",
        model=GEMINI_MODEL_1,
        logfile=workdir / "pass1.log",
        cwd=workdir,
    )
    _require(workdir / "extract.md", "Pass 1")
    logger.info("  wrote %s", workdir / "extract.md")


def stage_enrich(workdir: Path, model_fast: str | None) -> None:
    logger.info("[4/7] Pass 2: multi-source enrichment (batched)")

    extract_path = workdir / "extract.md"
    if not extract_path.exists():
        raise RuntimeError("extract.md not found — cannot run enrichment")

    items = _parse_extract_items(extract_path)
    if not items:
        raise RuntimeError("No topics/entities found in extract.md")

    batches = [
        items[i:i + BATCH_SIZE]
        for i in range(0, len(items), BATCH_SIZE)
    ]

    logger.info(
        "  splitting %d items into %d batches of up to %d",
        len(items),
        len(batches),
        BATCH_SIZE,
    )

    base_prompt = (PROMPTS_DIR / "pass2-enrich.md").read_text()
    batch_files = []

    for idx, batch in enumerate(batches):
        batch_file = workdir / f"background_batch_{idx}.md"
        prompt_file = workdir / f"pass2_batch_{idx}_prompt.md"

        items_block = "\n".join(f"- {item}" for item in batch)
        prompt_text = base_prompt.replace("{{ITEMS}}", items_block)
        prompt_text = prompt_text.replace(
            "{{OUTPUT_FILE}}",
            batch_file.name,
        )

        prompt_file.write_text(prompt_text)

        logger.info(
            "  batch %d/%d: %d items -> %s",
            idx + 1,
            len(batches),
            len(batch),
            batch_file.name,
        )

        run_pi(
            attachments=[],
            prompt_file=prompt_file,
            tools=None,
            model=GEMINI_MODEL_2,
            logfile=workdir / f"pass2_batch_{idx}.log",
            cwd=workdir,
        )

        logger.info(" Sleeping 1 min... TPM limit ")
        time.sleep(60)

        # Do not proceed until this batch has actually produced
        # usable output.
        _require(
            batch_file,
            f"Pass 2 batch {idx + 1}/{len(batches)}",
        )

        batch_files.append(batch_file)

        logger.info("  batch %d/%d completed successfully",idx + 1,len(batches),)

    combined = []

    for bf in batch_files:
        content = bf.read_text().strip()
        if content:
            combined.append(content)
        else:
            raise RuntimeError(
                f"Pass 2 produced an empty batch file: {bf.name}"
            )

    if not combined:
        raise RuntimeError(
            "Pass 2 produced no background content in any batch"
        )

    background_md = workdir / "background.md"
    background_md.write_text("\n\n".join(combined))

    logger.info(
        "  wrote %s by combining %d validated batch files",
        background_md,
        len(batch_files),
    )

    _require(background_md, "Pass 2")


def stage_synthesize(workdir: Path, model_strong: str | None, date_str: str) -> None:
    logger.info("[5/7] Pass 3: synthesis into full academic notes (no info dropped)")
    logger.debug("  using date_str=%r for notes.md header (derived from filename/mtime, not asked of pi)", date_str)
    # Deliberately NOT attaching transcript.txt here: extract.md's "Speaker's
    # claims & arguments" section already is a compressed-but-complete,
    # sentence-by-sentence record of the talk (see pass1-extract.md's
    # no-ellipsis rule), so re-sending the full transcript on top of it would
    # just burn tokens for content pi already has.
    # tools="write" only — same reasoning as stage_cleanup/stage_extract:
    # both attachments are already fully inlined, and this stage never
    # needs to open any other file.
    run_pi(
        attachments=[
            workdir / "extract.md",
            workdir / "background.md",
        ],
        prompt_file=PROMPTS_DIR / "pass3-synthesize.md",
        tools="write",
        model=GEMINI_MODEL_1,
        logfile=workdir / "pass3.log",
        cwd=workdir,
        substitutions={"{{DATA}}": date_str},
    )
    _require(workdir / "notes.md", "Pass 3")
    logger.info("  wrote %s", workdir / "notes.md")


def stage_verify(workdir: Path, model_strong: str | None) -> None:
    logger.info("[6/7] Pass 4: verification (flags unsupported claims, deletes nothing)")
    # extract.md (not transcript.txt) is the ground truth pass4-verify.md
    # actually checks the speaker's-view sentences against, so it's attached
    # here instead of the full transcript.
    # `read` kept here (unlike cleanup/extract/synthesize above): this pass
    # uses `edit` to annotate notes.md in place, which likely needs to read
    # current file state to locate exact text to replace — not the same
    # safe no-op removing `read` is for a pass that only ever writes fresh
    # output from fully-inlined attachments.
    run_pi(
        attachments=[
            workdir / "notes.md",
            workdir / "extract.md",
            workdir / "background.md",
        ],
        prompt_file=PROMPTS_DIR / "pass4-verify.md",
        tools="read,write,edit",
        model=GEMINI_MODEL_1,
        logfile=workdir / "pass4.log",
        cwd=workdir,
    )
    _require(workdir / "notes.md", "Pass 4")
    logger.info("  updated %s", workdir / "notes.md")


# ---------------------------------------------------------------------------
# PDF styling (WeasyPrint: pure HTML/CSS -> PDF, no LaTeX install required)
PDF_MARGIN = "2cm"
PDF_FONT_SIZE = "10.5pt"
PDF_FONT_FAMILY = "Georgia, 'Times New Roman', serif"

PDF_CSS = """\
@page {{
    size: A4;
    margin: {margin};

    @top-center {{
        content: string(doc-title);
        font-size: 8.5pt;
        color: #888;
        font-family: {font_family};
    }}
    @bottom-right {{
        content: "Pagina " counter(page) " di " counter(pages);
        font-size: 8.5pt;
        color: #aaa;
        font-family: {font_family};
    }}
}}

* {{ box-sizing: border-box; }}

body {{
    font-family: {font_family};
    font-size: {font_size};
    line-height: 1.75;
    color: #1a1a1a;
}}

h1 {{
    string-set: doc-title content();
    font-size: 22pt;
    font-weight: 700;
    color: #0f1f44;
    border-bottom: 3px solid #0f1f44;
    padding-bottom: 8px;
    margin: 0 0 24px 0;
    page-break-after: avoid;
}}

h2 {{
    font-size: 14pt;
    font-weight: 700;
    color: #0f1f44;
    border-left: 4px solid #3762cc;
    padding-left: 10px;
    margin: 30px 0 10px 0;
    page-break-after: avoid;
}}

h3 {{
    font-size: 12pt;
    font-weight: 700;
    color: #1e3468;
    margin: 18px 0 8px 0;
    page-break-after: avoid;
}}

p {{
    margin: 0 0 10px 0;
    text-align: justify;
    orphans: 3;
    widows: 3;
}}

ul, ol {{
    margin: 6px 0 12px 0;
    padding-left: 22px;
}}

li {{
    margin-bottom: 5px;
    line-height: 1.65;
}}

blockquote {{
    border-left: 4px solid #3762cc;
    background: #f0f4ff;
    padding: 10px 16px;
    margin: 14px 0;
    color: #1e3468;
    font-style: italic;
    border-radius: 0 4px 4px 0;
}}

blockquote p {{ margin: 0; }}

code {{
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    background: #f4f4f8;
    padding: 1px 5px;
    border-radius: 3px;
    color: #c0392b;
}}

strong {{ font-weight: 700; }}
em {{ font-style: italic; color: #333; }}

a {{
    color: #3762cc;
    text-decoration: none;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}}

th {{
    background: #0f1f44;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
}}

td {{
    border: 1px solid #ccd;
    padding: 7px 12px;
    vertical-align: top;
}}

tr:nth-child(even) td {{ background: #f7f8fc; }}

hr {{
    border: none;
    border-top: 1px solid #dde;
    margin: 24px 0;
}}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""


def _markdown_to_html(md_text: str) -> str:
    if markdown is None:
        raise RuntimeError("markdown not installed. Run: pip install markdown")
    md = markdown.Markdown(
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.sane_lists",
            "markdown.extensions.smarty",
        ]
    )
    return md.convert(md_text)


def _text_to_html(text: str, title: str) -> str:
    """Render plain transcript text as simple justified paragraphs, with a
    title heading so the PDF header (string-set on h1) picks it up."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    body = [f"<h1>{html.escape(title)}</h1>"]
    for para in paragraphs:
        escaped = html.escape(para).replace("\n", "<br>")
        body.append(f"<p>{escaped}</p>")
    return "\n".join(body)


def _render_pdf(html_body: str, dest: Path) -> None:
    if weasyprint is None:
        raise RuntimeError(
            f"weasyprint could not be imported ({WEASYPRINT_IMPORT_ERROR}). "
            f"This is almost always a native-library path issue, not a missing pip "
            f"install — see check_environment()'s message at startup for the fix."
        )
    css = PDF_CSS.format(margin=PDF_MARGIN, font_size=PDF_FONT_SIZE, font_family=PDF_FONT_FAMILY)
    full_html = HTML_TEMPLATE.format(css=css, body=html_body)
    weasyprint.HTML(string=full_html).write_pdf(str(dest))
    logger.info("  wrote %s", dest)


def _render_html(html_body: str, dest: Path) -> None:
    """Write a standalone, self-contained HTML file (same CSS/layout as the
    PDF, just without WeasyPrint's page-specific @page rules mattering)."""
    css = PDF_CSS.format(margin=PDF_MARGIN, font_size=PDF_FONT_SIZE, font_family=PDF_FONT_FAMILY)
    full_html = HTML_TEMPLATE.format(css=css, body=html_body)
    dest.write_text(full_html, encoding="utf-8")
    logger.info("  wrote %s", dest)


def stage_pdf(workdir: Path, run_name: str) -> None:
    logger.info("[7/7] Rendering PDF/HTML outputs (WeasyPrint)")

    notes_md = (workdir / "notes.md").read_text()
    notes_html_body = _markdown_to_html(notes_md)
    _render_pdf(notes_html_body, workdir / "notes.pdf")
    _render_html(notes_html_body, workdir / "notes.html")

    transcript_text = (workdir / "transcript.txt").read_text()
    _render_pdf(_text_to_html(transcript_text, f"Trascrizione — {run_name}"), workdir / "transcript.pdf")

    notes_md_dest = OUTPUT_DIR / f"{run_name}-notes.md"
    notes_pdf_dest = OUTPUT_DIR / f"{run_name}-notes.pdf"
    notes_html_dest = OUTPUT_DIR / f"{run_name}-notes.html"
    transcript_pdf_dest = OUTPUT_DIR / f"{run_name}-transcript.pdf"
    shutil.copy2(workdir / "notes.md", notes_md_dest)
    shutil.copy2(workdir / "notes.pdf", notes_pdf_dest)
    shutil.copy2(workdir / "notes.html", notes_html_dest)
    shutil.copy2(workdir / "transcript.pdf", transcript_pdf_dest)
    logger.info("  copied 4 output files to %s", OUTPUT_DIR)
    logger.info("    %s", notes_md_dest)
    logger.info("    %s", notes_pdf_dest)
    logger.info("    %s", notes_html_dest)
    logger.info("    %s", transcript_pdf_dest)


def _require(path: Path, stage_name: str) -> None:
    if not path.exists() or not path.read_text().strip():
        raise RuntimeError(
            f"{stage_name} did not produce {path.name} — check the matching "
            f"pass log in {path.parent} before continuing."
        )
    # Belt-and-suspenders: whatever pi/the model actually wrote, make sure
    # it doesn't contain a line pi itself can't fully read back in a later
    # stage. See ensure_no_long_lines() for why this can't be guaranteed
    # just by prompt instructions alone.
    ensure_no_long_lines(path)


def _check_resume_prereqs(workdir: Path, from_stage: str) -> None:
    """When resuming with --from-stage, fail fast with a clear message if
    the files that stage depends on aren't already in workdir, instead of
    letting pi run against a missing @attachment."""
    missing = [f for f in STAGE_REQUIRES.get(from_stage, []) if not (workdir / f).exists()]
    if missing:
        raise RuntimeError(
            f"--from-stage {from_stage} requires {', '.join(missing)} to already "
            f"exist in {workdir}, but they don't. Run from an earlier stage first."
        )


def process_file(audio: Path, args: argparse.Namespace, from_stage: str) -> None:
    run_name = audio.stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logfile = LOG_DIR / f"{timestamp}_{run_name}.log"
    setup_logging(logfile, args.verbose)

    logger.info("=== Processing %s ===", audio.name)
    logger.debug("Log file: %s", logfile)

    workdir = WORK_DIR / run_name
    workdir.mkdir(parents=True, exist_ok=True)

    start = STAGES.index(from_stage)
    if start > 0:
        _check_resume_prereqs(workdir, from_stage)

    if start <= STAGES.index("transcribe"):
        stage_transcribe(audio, workdir, args.whisper_model)
    if start <= STAGES.index("cleanup"):
        stage_cleanup(workdir, args.model_fast)
        logger.info(" Sleeping 1 min... TPM limit ")
        time.sleep(60)
    if start <= STAGES.index("extract"):
        stage_extract(workdir, args.model_fast)
    if start <= STAGES.index("enrich"):
        stage_enrich(workdir, args.model_fast)
    if start <= STAGES.index("synthesize"):
        stage_synthesize(workdir, args.model_strong, derive_date_from_filename(audio))
        logger.info(" Sleeping 1 min... TPM limit ")
        time.sleep(60)
    if start <= STAGES.index("verify"):
        stage_verify(workdir, args.model_strong)
    if start <= STAGES.index("pdf"):
        stage_pdf(workdir, run_name)

    if not args.keep_input and from_stage == "transcribe":
        try:
            inside_input_dir = audio.parent.resolve() == INPUT_DIR.resolve()
        except OSError:
            inside_input_dir = False
        if inside_input_dir:
            processed_dir = audio.parent / "processed"
            processed_dir.mkdir(exist_ok=True)
            dest = processed_dir / audio.name
            shutil.move(str(audio), str(dest))
            logger.info("Moved %s -> %s", audio.name, dest)
        else:
            logger.debug("Not moving %s to processed/ (outside INPUT_DIR)", audio)

    logger.info("Done with %s.", audio.name)
    logger.info("  4 output files in: %s", OUTPUT_DIR)
    logger.info("  Intermediate files kept in: %s", workdir)


def check_environment() -> None:
    """Ensure the working folders exist, and fail fast with a clear message
    if a required binary/package is missing — rather than discovering it
    after transcribing a 1-hour recording."""
    for path in (INPUT_DIR, OUTPUT_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)

    problems = []
    if shutil.which("pi") is None:
        problems.append("`pi` not found on PATH — required for extract/enrich/synthesize/verify")

    if weasyprint is None:
        problems.append(
            "weasyprint could not be loaded "
            f"({WEASYPRINT_IMPORT_ERROR}). On macOS this is almost always "
            "WeasyPrint failing to find its native libraries (pango/cairo/"
            "gdk-pixbuf) installed via Homebrew, not a missing `pip install`. Fix:\n"
            "      brew install cairo pango gdk-pixbuf libffi\n"
            "      export DYLD_FALLBACK_LIBRARY_PATH=\"$(brew --prefix)/lib:$DYLD_FALLBACK_LIBRARY_PATH\"\n"
            "    Add that export to your ~/.zshrc so it's set for every future run. "
            "Full troubleshooting: "
            "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#troubleshooting"
        )
    if markdown is None:
        problems.append("`markdown` not installed — required to render notes.md. Run: pip install markdown")
    if mlx_whisper is None:
        problems.append(
            "mlx-whisper isn't installed — it's the only transcription "
            "backend now. Run: pip install mlx-whisper"
        )
    elif shutil.which("ffmpeg") is None:
        problems.append(
            "`ffmpeg` not found on PATH — mlx-whisper needs it to decode audio. "
            "Run: brew install ffmpeg"
        )

    if problems:
        print("Setup problem(s) found before starting:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None,
                        help="Process a single audio file instead of scanning INPUT_DIR")
    parser.add_argument("--model-fast", default=os.environ.get("PI_MODEL_FAST"),
                        help="Model for extract/enrich passes (env: PI_MODEL_FAST). "
                             f"Does NOT affect cleanup (pass 0), which is always forced to "
                             f"the local CLEANUP_MODEL ({GEMINI_MODEL_1}) regardless of this flag.")
    parser.add_argument("--model-strong", default=os.environ.get("PI_MODEL_STRONG"),
                        help="Model for synthesize/verify passes (env: PI_MODEL_STRONG)")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL_DEFAULT,
                        help=f"MLX Whisper model (default: {WHISPER_MODEL_DEFAULT})")
    parser.add_argument("--from-stage", choices=STAGES, default="transcribe",
                        help="Resume from this stage (requires --file; reuses existing intermediate files)")
    parser.add_argument("--keep-input", action="store_true",
                        help="Don't move processed files into INPUT_DIR/processed/")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print debug-level logs to console")
    args = parser.parse_args()

    if args.from_stage != "transcribe" and args.file is None:
        parser.error("--from-stage other than 'transcribe' requires --file")

    check_environment()

    if args.file is not None:
        audio = args.file.resolve()
        if args.from_stage == "transcribe" and not audio.exists():
            parser.error(f"Audio file not found: {audio}")
        try:
            process_file(audio, args, args.from_stage)
        except Exception as e:
            logger.error("%s", e)
            sys.exit(1)
        return

    # Batch mode: scan INPUT_DIR, process one file at a time.
    audio_files = discover_audio_files(INPUT_DIR)

    if not audio_files:
        print(f"No audio files found in {INPUT_DIR}")
        return

    print(f"Found {len(audio_files)} file(s) in {INPUT_DIR}. Processing one at a time...")
    failures = []
    for audio in audio_files:
        try:
            process_file(audio, args, "transcribe")
        except TimeoutError as e:
            logger.error("FATAL: %s", e)
            raise
        except Exception as e:
            logger.error("Failed on %s: %s", audio.name, e)
            failures.append(audio.name)
            continue

    print("\nBatch complete.")
    if failures:
        print(f"  {len(failures)} file(s) failed: {', '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
