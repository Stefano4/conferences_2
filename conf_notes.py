#!/usr/bin/env python3
"""
conf_notes.py — batch pipeline: scans an input folder for recordings and, one at a time,
turns each into a transcript PDF, a notes markdown file, a notes HTML file, and a notes
PDF (4 files per recording).

Transcription is done locally with mlx-whisper (no cloud calls, no API key needed). The
raw transcript is first cleaned by pass 0 (a deterministic loop-remover in plain Python,
then a chunked, validated opencode pass — see stage_cleanup()), then that cleaned
transcript feeds a 4-pass opencode pipeline:
  0. cleanup     — (a) Python collapses exact whisper loops, no LLM; (b) the text is split
                   into ~2000-word chunks, each cleaned by one small opencode call
                   (filler repeats, obvious typos) and VALIDATED in code before it is
                   accepted; a chunk that never validates keeps its un-cleaned text, so
                   nothing is ever dropped. Finished chunks are reused on re-runs.
  1. extract     — pull claims/topics/quotes from the cleaned transcript only.
  2. enrich      — look up each topic across free, no-key sources (opencode's hosted
                   websearch tool — see run_opencode()'s OPENCODE_ENABLE_EXA note).
  3. synthesize  — combine into full academic notes, nothing dropped.
  4. verify      — flag (never delete) anything not traceable to a source.

Only passes 0 (cleanup) and 1 (extract) ever see the transcript text itself — passes 2-4
work from extract.md/background.md/notes.md, which already contain everything they need,
so the (much longer) transcript isn't re-sent to opencode on every later pass.

Tool access per pass is controlled by one of three custom opencode agents (AGENT_PROFILES)
that run_opencode() defines through OPENCODE_CONFIG_CONTENT and selects with `--agent` on
every call. Every call's log is then audited for tools the agent should not have been able
to use (shell, python, re-reading the attached file...) and a WARNING is printed if any
show up, so a silently-ignored restriction can't go unnoticed again.

Rate limits: input-token usage is estimated per call and spread over a sliding 60 s window
(TokenThrottle, OPENCODE_TPM_BUDGET) instead of fixed sleeps; if the provider still
answers "quota exceeded", the "retry in Ns" hint in its error is honoured.

Folder layout (created automatically next to the script if missing):
  input/   — drop recordings here; processed files move to input/processed/ so reruns
             don't redo them.
  output/  — final "<name>-notes.{md,html,pdf}" and "<name>-transcript.pdf" land here.
  logs/    — one log file per run, named yyyy-MM-dd_hh-mm_<name>.log.

Requirements:
    pip install mlx-whisper weasyprint markdown
    brew install cairo pango gdk-pixbuf libffi ffmpeg   # WeasyPrint + mlx-whisper native deps
    opencode CLI already installed and authenticated (`opencode auth login`) for whichever
    providers the `Models` defaults (section 1) resolve to. Run `opencode models` to confirm the exact
    provider/model strings your account has — the ones baked in below are a starting guess.

macOS note: if WeasyPrint fails to import with a library-loading error, Homebrew's libs
usually just aren't on the dynamic linker's search path — see check_environment()'s error
message for the exact fix (one `export`).

Code layout (one file, organised so that every numbered section can later move into its
own module without edits):
   1. CONFIG     frozen dataclasses (Config & friends). Built once in main() and passed
                 around explicitly; nothing is read from the environment at import time.
   2. ERRORS
   3. TEXT UTILS date-from-filename, extract.md parsing, long-line wrapping
   4. LOGGING & FILES
   5. OPENCODE   agent profiles, TokenThrottle, OpencodeRunner
   6. TRANSCRIBE
   7. CLEANUP    pass 0
   8. PASSES     passes 1-4 (extract / enrich / synthesize / verify)
   9. RENDER     CSS + HTML templates, PDF/HTML output
  10. PIPELINE   RunContext, the STAGES registry, retries, per-file processing
  11. CLI        environment check, argparse, main()

Environment variables (all optional, read once by Config.from_env()):
  OPENCODE_MODEL_FAST       model for extract + enrich      (--model-fast)
  OPENCODE_MODEL_STRONG     model for synthesize + verify   (--model-strong)
  OPENCODE_MODEL_CLEANUP    model for cleanup               (--model-cleanup)
  OPENCODE_TIMEOUT_SECONDS, OPENCODE_TPM_BUDGET, CLEANUP_CHUNK_WORDS, OPENCODE_EXTRA_ARGS
  CONF_NOTES_INPUT_DIR / _OUTPUT_DIR / _LOG_DIR / _WORK_DIR   relocate the folders

Usage:
    python conf_notes.py                                            # scan INPUT_DIR
    python conf_notes.py --file talk.m4a                             # single file
    python conf_notes.py --file talk.m4a --from-stage synthesize     # debug/rerun
    python conf_notes.py -v                                          # debug console log
"""

from __future__ import annotations

import argparse
import functools
import html
import importlib
import json
import logging
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

logger = logging.getLogger("conf_notes")


# =========================================================================================
# 1. CONFIG  (future: conf_notes/config.py)
# =========================================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


class ConfigError(ValueError):
    """Invalid configuration (bad env var, out-of-range setting...)."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


# opencode model IDs are always "provider/model" (run `opencode models` to see the exact
# strings your authenticated providers expose). "google" below assumes you've run
# `opencode auth login` for Gemini/Google AI Studio, and "ollama" assumes a local Ollama
# server serving the tag after the slash — adjust both prefixes to match your setup.
# NOTE: GEMINI_* are HOSTED Google models, not local ones — only LOCAL_MODEL (ollama/...)
# runs on this machine.
GEMINI_LATEST = "google/gemini-flash-lite-latest"
GEMINI_LATEST_LOW = "google/gemini-flash-lite-latest#low"
GEMINI_31_LOW = "google/gemini-3.1-flash-lite#low"
LOCAL_MODEL = "ollama/gemma4:e4b-mlx"

WHISPER_MODEL_DEFAULT = "mlx-community/whisper-large-v3-turbo"
AUDIO_EXTENSIONS = frozenset({".m4a", ".mp3", ".wav", ".mp4", ".aac", ".flac", ".ogg", ".mov"})


def is_local_model(model: str | None) -> bool:
    """Local (ollama/...) models have no cloud rate limit and a small context window."""
    return (model or "").startswith("ollama/")


@dataclass(frozen=True)
class Paths:
    """Every folder the script touches. `Paths.under(root)` gives the default layout
    (everything next to the script); individual folders can be relocated afterwards
    (iCloud Drive, external disk...) via replace() or the CONF_NOTES_*_DIR env vars."""
    root: Path
    prompts: Path
    work: Path       # scratch space for intermediate files, one sub-folder per recording
    input: Path
    output: Path
    logs: Path
    hf_cache: Path   # Hugging Face cache (pinned so cron/launchd runs reuse the same models)

    @classmethod
    def under(cls, root: Path) -> "Paths":
        return cls(
            root=root, prompts=root / "prompts", work=root / "work", input=root / "input",
            output=root / "output", logs=root / "logs", hf_cache=root / ".hf-cache",
        )

    def ensure(self) -> None:
        """Create the folders that must exist before a run (input/ output/ logs/)."""
        for path in (self.input, self.output, self.logs):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Models:
    """One model per stage, as "provider/model". The CLI/env "fast" and "strong" groups are
    shortcuts that set several of these at once (see with_overrides())."""
    whisper: str = WHISPER_MODEL_DEFAULT
    cleanup: str = GEMINI_LATEST_LOW
    extract: str = GEMINI_LATEST
    # Swap enrich to LOCAL_MODEL to run pass 2 fully offline. stage_enrich() adapts by itself:
    # batch size drops to EnrichSettings.batch_size_local (one item per call, since a local
    # model's context can't hold 8 items' worth of web results), and the TPM throttle is
    # skipped because it only exists for cloud rate limits.
    enrich: str = GEMINI_31_LOW
    synthesize: str = GEMINI_LATEST
    verify: str = GEMINI_LATEST_LOW

    def with_overrides(
        self, *, fast: str | None = None, strong: str | None = None,
        cleanup: str | None = None, whisper: str | None = None,
    ) -> "Models":
        """fast -> extract + enrich, strong -> synthesize + verify, cleanup -> cleanup.
        Falsy values are ignored, so callers can pass argparse/env values straight in."""
        updates: dict[str, str] = {}
        if fast:
            updates.update(extract=fast, enrich=fast)
        if strong:
            updates.update(synthesize=strong, verify=strong)
        if cleanup:
            updates["cleanup"] = cleanup
        if whisper:
            updates["whisper"] = whisper
        return replace(self, **updates)


@dataclass(frozen=True)
class CleanupSettings:
    """Pass 0 (see stage_cleanup())."""
    chunk_words: int = 2000          # words per LLM call
    chunk_attempts: int = 3          # valid-output attempts per chunk before falling back to un-cleaned text
    min_word_ratio: float = 0.85     # accepted output must keep >= this share of the chunk's words ...
    max_word_ratio: float = 1.03     # ... and not grow by more than this
    max_novel_ratio: float = 0.03    # max share of output words that don't occur in the input chunk

    def __post_init__(self) -> None:
        _check(self.chunk_words >= 1, "cleanup.chunk_words must be >= 1")
        _check(self.chunk_attempts >= 1, "cleanup.chunk_attempts must be >= 1")
        _check(0 < self.min_word_ratio <= 1 <= self.max_word_ratio, "cleanup word ratios must satisfy 0 < min <= 1 <= max")


@dataclass(frozen=True)
class EnrichSettings:
    """Pass 2 batching."""
    batch_size: int = 8          # items per opencode call when the enrich model is a cloud model
    batch_size_local: int = 1    # items per opencode call when it is a local (ollama/...) model

    def __post_init__(self) -> None:
        _check(self.batch_size >= 1 and self.batch_size_local >= 1, "enrich batch sizes must be >= 1")


@dataclass(frozen=True)
class RateLimitSettings:
    max_waits: int = 6              # "retry in Ns" waits tolerated per chunk before giving up
    max_wait_seconds: int = 300     # a longer hint than this means a daily quota: waiting won't help
    # Input-token budget per rolling 60 s, used by TokenThrottle to pace calls to a hosted
    # model. Keep it below the provider's real limit (Google free tier: 250K/min) because the
    # per-call estimate is rough. 0 disables throttling. Local models are never throttled.
    tpm_budget: int = 200_000
    chars_per_token: float = 3.0    # rough figure for Italian text
    agent_base_tokens: int = 3000   # system prompt + tool definitions of the restricted agents

    def __post_init__(self) -> None:
        _check(self.tpm_budget >= 0, "rate.tpm_budget must be >= 0")
        _check(self.chars_per_token > 0, "rate.chars_per_token must be > 0")


@dataclass(frozen=True)
class RetrySettings:
    stage_attempts: int = 3
    stage_sleep_seconds: int = 15   # pause after an ordinary failure; provider rate limits use their own hinted delay

    def __post_init__(self) -> None:
        _check(self.stage_attempts >= 1, "retry.stage_attempts must be >= 1")


@dataclass(frozen=True)
class RenderSettings:
    margin: str = "2cm"
    font_size: str = "12pt"              # notes.pdf
    transcript_font_size: str = "10pt"   # transcript.pdf — smaller since it's raw reference text, not the primary read
    font_family: str = "Georgia, 'Times New Roman', serif"


@dataclass(frozen=True)
class Config:
    """Everything tunable, in one immutable object. Build it once (Config.from_env(), then
    .with_overrides() for CLI flags) and pass it down; no function reads module constants
    or os.environ for settings."""
    paths: Paths
    models: Models = field(default_factory=Models)
    cleanup: CleanupSettings = field(default_factory=CleanupSettings)
    enrich: EnrichSettings = field(default_factory=EnrichSettings)
    rate: RateLimitSettings = field(default_factory=RateLimitSettings)
    retry: RetrySettings = field(default_factory=RetrySettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    opencode_timeout_seconds: int = 1800
    verify_max_shrink: float = 0.10      # pass 4 may not shrink notes.md by more than this share of its words
    audio_extensions: frozenset[str] = AUDIO_EXTENSIONS
    keep_input: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        _check(self.opencode_timeout_seconds > 0, "opencode_timeout_seconds must be > 0")
        _check(0 <= self.verify_max_shrink < 1, "verify_max_shrink must be in [0, 1)")

    @classmethod
    def from_env(cls, root: Path | None = None, env: Mapping[str, str] | None = None) -> "Config":
        """Defaults + environment variables. `env` is injectable for tests."""
        env = os.environ if env is None else env

        def env_int(name: str, default: int) -> int:
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                return int(raw)
            except ValueError:
                raise ConfigError(f"{name} must be an integer, got {raw!r}") from None

        paths = Paths.under(Path(root) if root else SCRIPT_DIR)
        folder_vars = {
            "input": "CONF_NOTES_INPUT_DIR", "output": "CONF_NOTES_OUTPUT_DIR",
            "logs": "CONF_NOTES_LOG_DIR", "work": "CONF_NOTES_WORK_DIR",
        }
        moved = {attr: Path(env[var]).expanduser() for attr, var in folder_vars.items() if env.get(var)}
        paths = replace(paths, **moved)

        models = Models().with_overrides(
            fast=env.get("OPENCODE_MODEL_FAST"), strong=env.get("OPENCODE_MODEL_STRONG"),
            cleanup=env.get("OPENCODE_MODEL_CLEANUP"),
        )
        return cls(
            paths=paths,
            models=models,
            cleanup=CleanupSettings(chunk_words=env_int("CLEANUP_CHUNK_WORDS", CleanupSettings.chunk_words)),
            rate=RateLimitSettings(tpm_budget=env_int("OPENCODE_TPM_BUDGET", RateLimitSettings.tpm_budget)),
            opencode_timeout_seconds=env_int("OPENCODE_TIMEOUT_SECONDS", cls.opencode_timeout_seconds),
        )

    def with_overrides(
        self, *, model_fast: str | None = None, model_strong: str | None = None,
        model_cleanup: str | None = None, whisper_model: str | None = None,
        keep_input: bool | None = None, verbose: bool | None = None,
    ) -> "Config":
        """A copy with CLI flags applied on top (None = leave as is)."""
        updates: dict = {"models": self.models.with_overrides(
            fast=model_fast, strong=model_strong, cleanup=model_cleanup, whisper=whisper_model,
        )}
        if keep_input is not None:
            updates["keep_input"] = keep_input
        if verbose is not None:
            updates["verbose"] = verbose
        return replace(self, **updates)


def configure_environment(cfg: Config) -> None:
    """Process-wide side effects, done explicitly from main() instead of at import time.

    Pin HF_HOME *before* mlx_whisper/huggingface_hub are imported: otherwise the cache
    location depends on $HOME at call time, so running from cron/launchd/a different shell
    can silently point at a different cache dir and make it look like the model is
    re-downloaded every day. (mlx_whisper is imported lazily, see _try_import().)"""
    os.environ.setdefault("HF_HOME", str(cfg.paths.hf_cache))


@functools.lru_cache(maxsize=None)
def _try_import(name: str):
    """(module, None) or (None, error text). Optional backends are imported lazily and once,
    so a missing backend degrades gracefully instead of crashing the script at import.
    WeasyPrint can fail with OSError (not ImportError) when it can't dlopen its native
    libs (pango/cairo/gdk-pixbuf) — see check_environment() for the macOS fix."""
    try:
        return importlib.import_module(name), None
    except (ImportError, OSError) as e:
        return None, str(e)


# =========================================================================================
# 2. ERRORS  (future: conf_notes/errors.py)
# =========================================================================================

class OpencodeRateLimit(RuntimeError):
    """opencode exited non-zero because the provider rejected the call for quota / rate-limit
    reasons. `retry_after` is the delay in seconds the provider asked for, parsed from its
    error message (60 when the message named none)."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after



class StagePermanentError(RuntimeError):
    """A stage failure that re-running the same stage against the same on-disk inputs
    can never fix (e.g. a required upstream file has content the parser can't use).
    Distinct from a plain RuntimeError, which may be a flaky opencode call/timeout
    worth retrying. Raise this instead of RuntimeError for that kind of failure so
    _run_stage_with_retry() fails fast instead of burning 3 retries x 60s on an error
    that will reproduce identically every time."""


# =========================================================================================
# 3. TEXT UTILS  (future: conf_notes/textutils.py)
# =========================================================================================

_ITALIAN_MONTHS = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]

# Date patterns to try, in order, against a recording's filename. Each tuple is
# (regex, group order). Deliberately conservative (4-digit years, non-digit boundaries)
# to avoid false positives on filenames that just happen to contain other numbers.
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
    """Return every topic/entity listed under 'Topics & concepts mentioned' and
    'Named entities' in extract.md.

    The extract pass is itself an opencode call (often on a small/fast model), and
    that model doesn't reliably follow the "- item" bullet format the prompt asks
    for — it sometimes writes one bare item per line instead (no leading "- "/"* "/
    "1. "). Treat any non-empty, non-heading line inside the section as an item
    rather than requiring a bullet marker, so a items list that's really just
    differently formatted isn't mistaken for an empty one.
    """
    items, in_section = [], False
    for line in extract_path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("## Topics & concepts mentioned") or stripped.startswith("## Named entities"):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("## "):
                in_section = False
            elif stripped.startswith("#"):
                continue  # some other heading level inside the section — skip, stay in_section
            elif stripped:
                # Strip a leading bullet/number marker if present, otherwise take the line as-is.
                item = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", stripped).strip()
                if item:
                    items.append(item)
    return items

def derive_date_from_filename(audio: Path) -> str:
    """Best-effort 'Data' value for notes.md's header.

    The speaker rarely states the date out loud, so asking opencode to find it in the
    transcript fails most of the time (extract.md's Session metadata usually says "not
    stated"). We derive it ourselves and hand it to opencode as a fixed value instead of
    a lookup task it would likely get wrong.

    Tries common date patterns in the filename first (recordings are typically named by
    whoever made them, e.g. "2026-03-05_convegno.m4a"), then falls back to the file's
    last-modified time, clearly labelled as a weaker signal (could be a copy/export date
    rather than the talk's date).
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


# Ported from pi's `read` tool, which silently truncated per line at roughly this many
# bytes. opencode's docs (https://opencode.ai/docs/tools) don't publish an equivalent
# per-line cap for its own `read` tool, so this is kept as a cheap, harmless safety net
# rather than a confirmed opencode limit — verify against the docs above if pass 4
# (verify, the only pass still using opencode's `read` tool) ever seems to see a
# truncated notes.md/extract.md/background.md.
OPENCODE_READ_LINE_LIMIT_BYTES = 50_000


def ensure_no_long_lines(path: Path, max_bytes: int = OPENCODE_READ_LINE_LIMIT_BYTES) -> None:
    """Safety net: insert line breaks at whitespace so no line in `path` exceeds
    max_bytes, without changing any word/character in the file — only where the
    newlines fall.

    This exists in addition to transcribe_with_local_whisper() pre-splitting
    transcript-raw.txt, because that only guarantees the *first* file opencode reads is
    safe. Every later stage's output is written by opencode itself (following prompt
    instructions like "keep line breaks as in the original"), and a model — especially a
    small local one — isn't guaranteed to honor that. If e.g. pass 0's cleanup model
    flattens transcript.txt back into one long paragraph while "cleaning" it, pass 1
    would silently see only a truncated prefix of the talk, with no error raised
    anywhere. Called on every stage's output right after _require() confirms it exists.
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
        # Break only at existing spaces, never mid-word.
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
    logger.debug("  rewrapped long line(s) in %s to stay under the ~%dKB read-line safety margin", path.name, max_bytes // 1000)


# =========================================================================================
# 4. LOGGING & FILES  (future: conf_notes/logs.py)
# =========================================================================================

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



def discover_audio_files(input_dir: Path, extensions: frozenset[str] | set[str] = AUDIO_EXTENSIONS) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files, key=lambda p: p.name)


# =========================================================================================
# 5. OPENCODE  (future: conf_notes/opencode.py)
# =========================================================================================

# --- opencode agents --------------------------------------------------------------------
# Every opencode call runs under one of these restricted agents: run_opencode() selects it
# with `--agent <name>` and defines it through OPENCODE_CONFIG_CONTENT (see
# opencode_config_content()). Without that, opencode uses its default `build` agent with
# EVERY tool enabled, and a small model will happily use them (shell, python scripts, git,
# re-reading a file that is already attached, skills, questions...), re-sending the whole
# context on each extra step — that is what burned the token quota in the pass-0 logs.
#
#   allow      the only tools the agent may use; everything else is denied. opencode gates
#              both `write` and `edit` behind the single "edit" permission.
#   steps      cap on the agent's tool loop (opencode then forces a plain-text answer).
#   est_calls  how many model requests one invocation is assumed to make (each re-sends the
#              full context) — only used by TokenThrottle for its estimate.
#   growth     tokens each step adds to the context (web pages / tool output), for that
#              estimate; 0 for agents that never use the web.
ALL_TOOL_KEYS = (
    "read", "edit", "glob", "grep", "list", "bash", "execute", "task", "skill",
    "webfetch", "websearch", "codesearch", "question", "todowrite", "todoread", "lsp",
)

AGENT_PROFILES = {
    "notes-writer": {
        "description": "Writes one fresh output file from the attached input. No read, no bash, no web.",
        "allow": {"edit"},
        "steps": 6,
        "est_calls": 3,
        "growth": 0,
        "system": (
            "You are a text-processing assistant. The complete input is already attached to the message, so "
            "never read, list, search or run anything. Follow the instructions in the message exactly: write "
            "the requested output file with the file-writing tool, containing only the requested text, then "
            "answer with one short line and stop. Do not put commentary, notes or headers of your own in the file."
        ),
    },
    # Pass 4 (pass4-verify.md) tells the model to "search the web if you need" and to rewrite
    # notes.md IN PLACE, so this agent needs web access plus read (opencode's edit/write tools
    # refuse to touch an existing file that wasn't read first).
    "notes-editor": {
        "description": "Fact-checks and edits an existing file in place. Read, edit, web search/fetch only.",
        "allow": {"read", "edit", "webfetch", "websearch"},
        "steps": 40,
        "est_calls": 20,
        "growth": 2000,
        "system": (
            "You are a careful fact-checking editor. Start by reading the file you are asked to check. You may "
            "search the web to check doubtful facts. Apply only the corrections the message asks for, in place, "
            "and never remove content. Do not run commands or read any other local file."
        ),
    },
    # Pass 2 (pass2-enrich.md): web lookups for a batch of topics, one output file.
    # run_opencode(max_steps=, est_calls=) scale "steps"/"est_calls" with the batch size.
    "notes-enrich": {
        "description": "Looks up topics on the web and writes a short background note. Web + edit only.",
        "allow": {"edit", "webfetch", "websearch"},
        "steps": 40,
        "est_calls": 20,
        "growth": 2000,
        "system": (
            "You are a research assistant. Follow the instructions in the message: look up every listed topic "
            "on the web, and write the result to the requested file. Do not run commands or read local files."
        ),
    },
}


def _build_permission(allow: set[str]) -> dict[str, str]:
    """Default-deny permission map for one agent: "*" first, then an explicit entry per
    known tool (opencode applies the last matching rule, and the explicit denies also
    cover opencode versions that ignore the "*" wildcard)."""
    perm = {"*": "deny"}
    for tool in ALL_TOOL_KEYS:
        perm[tool] = "allow" if tool in allow else "deny"
    return perm


def opencode_config_content(steps_override: dict[str, int] | None = None) -> str:
    """JSON for the OPENCODE_CONFIG_CONTENT env var: our agents, merged into any config
    already supplied through that variable. steps_override = {agent: steps} replaces an
    agent's step cap for this one call (enrich scales it with the batch size)."""
    agents = {
        name: {
            "description": p["description"],
            "mode": "primary",
            "prompt": p["system"],
            "permission": _build_permission(p["allow"]),
            "steps": (steps_override or {}).get(name, p["steps"]),
        }
        for name, p in AGENT_PROFILES.items()
    }
    config: dict = {}
    existing = os.environ.get("OPENCODE_CONFIG_CONTENT")
    if existing:
        try:
            config = json.loads(existing)
        except ValueError:
            logger.warning("OPENCODE_CONFIG_CONTENT is set but is not valid JSON — ignoring it")
    config.setdefault("agent", {}).update(agents)
    return json.dumps(config)



class TokenThrottle:
    """Keep estimated input tokens under `budget` per rolling 60 seconds by sleeping before
    a call when needed. Replaces the old fixed `sleep(30/60)` calls between stages: the wait
    now depends on what was actually sent, so small calls aren't delayed and a burst of
    chunk calls is spaced out just enough."""

    def __init__(self, budget_per_minute: int):
        self.budget = budget_per_minute
        self.events: list[tuple[float, int]] = []

    def wait_for(self, tokens: int) -> None:
        if self.budget <= 0:
            return
        tokens = min(tokens, self.budget)  # a single call bigger than the budget can't be helped
        while True:
            now = time.monotonic()
            self.events = [(t, n) for t, n in self.events if now - t < 60]
            used = sum(n for _, n in self.events)
            if used + tokens <= self.budget:
                break
            wait = 60 - (now - self.events[0][0]) + 0.5
            logger.info("  throttle: ~%dK tokens sent in the last 60s (budget %dK) — waiting %.0fs",
                        used // 1000, self.budget // 1000, wait)
            time.sleep(max(wait, 1.0))
        self.events.append((time.monotonic(), tokens))

    def reset(self) -> None:
        """Forget the window (call after having slept out a provider-requested delay)."""
        self.events.clear()



def _estimate_tokens(files: list[Path], prompt_text: str, agent: str, est_calls: int | None, rate: RateLimitSettings) -> int:
    """Rough input-token total for one invocation: every request re-sends the prompt plus
    the files in context, and web-capable agents also accumulate `growth` tokens of tool
    output per step (so cost grows with the square of the number of steps)."""
    profile = AGENT_PROFILES[agent]
    calls = est_calls or profile["est_calls"]
    chars = len(prompt_text) + sum(p.stat().st_size for p in files if p.exists())
    per_call = chars / rate.chars_per_token + rate.agent_base_tokens
    return int(per_call * calls + profile["growth"] * calls * (calls - 1) / 2)


# Lines opencode prints when it uses a tool, mapped to the permission key they correspond
# to. Matched against the (ANSI-stripped, timestamp-stripped) lines of a call's log by
# _audit_log(). Best effort: it only knows the line shapes seen in real logs.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOG_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
_TOOL_LINE_PATTERNS = [
    (re.compile(r"^\$ "), "bash"),
    (re.compile(r"^⚙\s"), "execute"),
    (re.compile(r"^[→✗✓←]\s*Skill\b"), "skill"),
    (re.compile(r"^[→✗✓←]\s*Read\b"), "read"),
    (re.compile(r"^[→✗✓←]\s*Glob\b"), "glob"),
    (re.compile(r"^[→✗✓←]\s*Grep\b"), "grep"),
    (re.compile(r"^[→✗✓←]\s*List\b"), "list"),
    (re.compile(r"^[→✗✓←]?\s*Asked \d+ question"), "question"),
]


def _audit_log(agent: str, log_text: str) -> Counter:
    """Warn if the call used tools its agent is not allowed to — i.e. if opencode did not
    actually apply AGENT_PROFILES (wrong config keys for this opencode version, agent name
    unknown, ...). Returns the Counter of offending tools (empty when all is well)."""
    allowed = AGENT_PROFILES[agent]["allow"]
    used: Counter = Counter()
    for raw in log_text.splitlines():
        line = _LOG_TS_RE.sub("", _ANSI_RE.sub("", raw)).strip()
        for pattern, key in _TOOL_LINE_PATTERNS:
            if pattern.match(line):
                if key not in allowed:
                    used[key] += 1
                break
    if used:
        logger.warning(
            "  agent '%s' used tools it should not have: %s — the permission restrictions were NOT "
            "enforced. Check the opencode version's agent/permission config schema (see "
            "opencode_config_content()) and the call log.",
            agent, ", ".join(f"{k}×{v}" for k, v in used.items()),
        )
    return used


_RETRY_HINT_RES = [
    re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.I),
    re.compile(r"retryDelay\W{1,6}([0-9]+(?:\.[0-9]+)?)s", re.I),
]
_QUOTA_RE = re.compile(r"quota|rate.?limit|RESOURCE_EXHAUSTED|\b429\b|too many requests", re.I)


def _rate_limit_from_log(log_text: str) -> float | None:
    """If the end of the log looks like a quota/rate-limit rejection, return the delay to wait
    in seconds (60 if the provider didn't say), else None."""
    tail = "\n".join(log_text.splitlines()[-40:])
    if not _QUOTA_RE.search(tail):
        return None
    for pattern in _RETRY_HINT_RES:
        m = pattern.search(tail)
        if m:
            return float(m.group(1))
    return 60.0



class OpencodeRunner:
    """Runs opencode calls for one Config. Owns the TokenThrottle, so the rate-limit window
    is shared by every call made through this runner (and by nothing else)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.throttle = TokenThrottle(cfg.rate.tpm_budget)

    def run(
        self, *, attachments: list[Path], prompt_file: Path, model: str | None,
        logfile: Path, cwd: Path, agent: str, substitutions: dict[str, str] | None = None,
        max_steps: int | None = None, est_calls: int | None = None, read_files: list[Path] | None = None,
    ) -> None:
        """Invoke opencode non-interactively under one of the restricted AGENT_PROFILES:
        opencode run --agent <agent> --file f1 --file f2 --model <provider/model> --thinking --auto "<prompt>"

        `agent` is mandatory on purpose: a call that forgets it would silently get opencode's
        default all-tools agent again. Before launching, the call is paced by TokenThrottle
        (hosted models only). After it, the log is audited for forbidden tool use, and a
        quota / rate-limit failure is raised as OpencodeRateLimit carrying the provider's
        "retry in Ns" delay so callers can wait exactly that long.

        We set the subprocess cwd explicitly to `workdir` (opencode resolves relative file
        operations like "write to extract.md" against its own process cwd), and pass --auto
        so a headless run never blocks waiting for an interactive permission prompt. Keep a
        flag between the last --file and the prompt (here: --auto): --file takes several
        values and could otherwise swallow the prompt.

        Optional per-call tuning: max_steps overrides the agent's step cap, est_calls its assumed
        number of requests, and read_files lists files the agent will `read` itself (not attached)
        so the throttle's token estimate still counts them.
        """
        if agent not in AGENT_PROFILES:
            raise ValueError(f"unknown opencode agent {agent!r}; expected one of {sorted(AGENT_PROFILES)}")

        prompt_text = prompt_file.read_text()
        # Fill in values the script already knows (e.g. the talk's date, derived from the
        # filename) instead of leaving opencode to infer them from the transcript, which is
        # unreliable for things speakers rarely state out loud.
        for key, value in (substitutions or {}).items():
            prompt_text = prompt_text.replace(key, value)

        # OpenCode run does not support a --dir flag in the installed CLI version.
        # Run the process with cwd=WORK_DIR and align PWD so OpenCode resolves its
        # project root and all relative paths from the intended working directory.
        cmd = ["opencode", "run", "--agent", agent]
        for p in attachments:
            cmd += ["--file", str(p)]
        if model:
            cmd += ["--model", model]
        cmd += ["--thinking"]
        # Extra flags for opencode itself (e.g. --log-level DEBUG). Varies by opencode
        # version/config, so it's an opt-in env var: export OPENCODE_EXTRA_ARGS="--log-level DEBUG"
        cmd += os.environ.get("OPENCODE_EXTRA_ARGS", "").split()
        cmd += ["--auto", prompt_text]

        env = os.environ.copy()
        # Enable opencode's web search integration for the enrichment pass.
        env.setdefault("OPENCODE_ENABLE_EXA", "1")
        # Define our restricted agents for this call (merged into any config already in the env).
        env["OPENCODE_CONFIG_CONTENT"] = opencode_config_content({agent: max_steps} if max_steps else None)
        # OpenCode resolves its root from PWD in addition to process.cwd(). Keep both
        # aligned so all relative writes land in the requested WORK_DIR/run directory.
        env["PWD"] = str(cwd.resolve())

        if not is_local_model(model):  # local models have no cloud TPM limit
            estimate = _estimate_tokens(attachments + (read_files or []), prompt_text, agent, est_calls, self.cfg.rate)
            if 0 < self.throttle.budget < estimate:
                logger.warning(
                    "  this call alone is estimated at ~%dK input tokens, above the %dK/min budget — "
                    "expect a full-minute wait before it, and a provider rate limit if the real limit is as low. "
                    "Consider a smaller batch (enrich.batch_size) or a higher-quota model.",
                    estimate // 1000, self.throttle.budget // 1000,
                )
            self.throttle.wait_for(estimate)

        logger.info(
            "Running opencode (cwd=%s): opencode run --agent %s %s ...%s",
            cwd, agent, " ".join(f"--file {p.name}" for p in attachments), f" --model {model}" if model else "",
        )

        # Stream instead of subprocess.run(): piping stdout straight into logfile with a raw
        # open() bypasses the logging module (no timestamps), and opencode's real-time
        # activity (tool calls, retries, which provider actually answered) stays invisible
        # until the whole call finishes. Streaming line-by-line lets us timestamp every row
        # and mirror it into the main run log so it's visible live with -v.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env, text=False, bufsize=0,
            # Close stdin immediately rather than passing DEVNULL: several non-interactive
            # CLI agents (pi's -p mode being the original case that prompted this) sit in
            # epoll_wait forever if stdin is /dev/null but exit cleanly once stdin is a pipe
            # that's been closed. Cheap and harmless to keep for opencode too; drop it if you
            # confirm opencode's `run` doesn't need it.
            stdin=subprocess.PIPE,
            start_new_session=True,  # own process group, so we can clean up stragglers below
        )
        proc.stdin.close()

        # Read with a poll/timeout loop instead of `for line in proc.stdout`, which blocks
        # until EOF — requiring every process holding the write end open to close it, not
        # just opencode. A tool opencode invokes (e.g. webfetch rendering a page) could in
        # principle spawn a subprocess that inherits this pipe's fd and never closes it, so
        # opencode finishes and exits cleanly but the pipe never sees EOF because a
        # grandchild still holds it open. Polling treats "opencode's own process has exited"
        # as the real completion signal, independent of anything else holding the pipe.
        timeout = self.cfg.opencode_timeout_seconds
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
                            logger.debug("[opencode] %s", line)
                        continue  # more may be buffered; keep draining first
                # No data ready. If opencode's own process already exited, we're done —
                # anything still holding the pipe open is a leftover grandchild, not opencode.
                if proc.poll() is not None:
                    break

                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    logger.error("opencode timed out after %.0f seconds; aborting pipeline", elapsed)
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
                    raise TimeoutError(f"opencode exceeded the hard timeout of {timeout}s")
            if buf:
                line = buf.decode("utf-8", errors="replace")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lf.write(f"{ts} {line}\n")
                logger.debug("[opencode] %s", line)

        returncode = proc.returncode

        # Best-effort cleanup of leftover descendants (e.g. a headless fetch opencode's
        # webfetch tool spawned) so they don't pile up over many batches.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        log_text = logfile.read_text(errors="replace")
        _audit_log(agent, log_text)

        if returncode != 0:
            retry_after = _rate_limit_from_log(log_text)
            if retry_after is not None:
                raise OpencodeRateLimit(
                    f"opencode call hit a provider quota/rate limit (exit {returncode}, retry in ~{retry_after:.0f}s). "
                    f"See {logfile} for details.", retry_after,
                )
            raise RuntimeError(f"opencode call failed (exit {returncode}). See {logfile} for details.")


# =========================================================================================
# 6. TRANSCRIBE  (future: conf_notes/transcribe.py)
# =========================================================================================

def _whisper_model_is_cached(whisper_model: str) -> bool:
    """Best-effort check of whether whisper_model is already in the local HF cache,
    purely so the log can confirm the caching fix is working instead of guessing from
    timing."""
    try:
        from huggingface_hub import scan_cache_dir
        return whisper_model in {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        return False  # can't tell -> don't claim either way in the log


def transcribe_with_local_whisper(audio: Path, whisper_model: str) -> str:
    mlx_whisper, error = _try_import("mlx_whisper")
    if mlx_whisper is None:
        raise StagePermanentError(f"mlx-whisper could not be imported ({error}). Run: pip install mlx-whisper")

    if _whisper_model_is_cached(whisper_model):
        logger.debug("  Whisper model %s found in cache (%s) — no download needed", whisper_model, os.environ.get("HF_HOME"))
    else:
        logger.info("  Whisper model %s not in cache (%s) — downloading once, will be cached for future runs", whisper_model, os.environ.get("HF_HOME"))

    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=whisper_model)

    # Join whisper's own per-phrase segments with newlines instead of using
    # result["text"] (one giant single-line string for the whole recording): a ~1-hour
    # talk produces a ~50-60KB single line, blowing past the per-line read-limit safety margin
    # (50KB) and silently truncating whatever pass reads this file first (pass 0
    # cleanup) to just the opening minutes, with no error. Segment boundaries are
    # whisper's own phrase breaks, not a rewording, so this changes nothing about the
    # transcribed words, only how they're laid out on disk.
    segments = result.get("segments") or []
    if segments:
        lines = [seg.get("text", "").strip() for seg in segments]
        return "\n".join(line for line in lines if line)
    return result["text"].strip()  # fallback if a whisper backend/version omits segments



def stage_transcribe(ctx: RunContext) -> None:
    model = ctx.cfg.models.whisper
    logger.info("  local mlx-whisper, model=%s", model)
    text = transcribe_with_local_whisper(ctx.audio, model)
    out = ctx.workdir / "transcript-raw.txt"
    out.write_text(text)
    ensure_no_long_lines(out)  # belt-and-suspenders alongside the segment-based split above
    logger.info("  wrote %s (%d words)", out, len(text.split()))


# =========================================================================================
# 7. CLEANUP (pass 0)  (future: conf_notes/cleanup.py)
# =========================================================================================

# --- Pass 0: cleanup ---------------------------------------------------------------------
# The cleanup rules live in prompts/pass0-cleanup.md (single source of truth, edit them
# there). It was written for a whole-file run ("transcript-raw.txt" in, "transcript.txt"
# out), so for each chunk _render_cleanup_prompt() prepends a short note and swaps those two
# file names for the chunk's own input/output files; the rules themselves are untouched.

# Greetings / thanks / fillers named in pass 0's rule 2. Immediate duplicates of these
# ("grazie grazie", "ecco ecco") are collapsed in Python, with no LLM involved.
_FILLER_KEYS = {"grazie", "buongiorno", "buonasera", "salve", "ciao", "ecco", "cioè", "diciamo"}

_EDGE_PUNCT = ".,;:!?…\"'«»“”‘’()[]{}—–-"
_WORD_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)


def _norm_word(word: str) -> str:
    return word.strip(_EDGE_PUNCT).casefold()


_TRAIL_PUNCT_RE = re.compile(r"[.,;:!?…»”’)\]]+$")


def _with_tail_of(token: str, donor: str) -> str:
    """`token` with its trailing punctuation replaced by the donor's ("grazie" + "grazie." -> "grazie.")."""
    tail = _TRAIL_PUNCT_RE.search(token)
    stem = token[:tail.start()] if tail else token
    donor_tail = _TRAIL_PUNCT_RE.search(donor)
    return stem + (donor_tail.group(0) if donor_tail else "")


def _collapse_runs(keys: list[str], max_n: int, min_reps) -> tuple[list[int], dict[int, int]]:
    """(indices to KEEP, {kept index: index of the run's last item}) after collapsing immediately repeated blocks of items in `keys`.

    At each position every block size n (1..max_n) is tried; a block counts if it repeats
    back to back at least min_reps(block) times. Among those, the one covering the most
    items wins (shortest block on ties, so "A A A A" collapses via n=1 and "A B A B A B"
    via n=2), and the whole run is reduced to its first copy. Blocks containing an empty
    key (blank line / bare punctuation) are never treated as repeats.
    """
    keep: list[int] = []
    tail_from: dict[int, int] = {}
    i, total = 0, len(keys)
    while i < total:
        best: tuple[int, int] | None = None  # (items covered, block size)
        for n in range(1, min(max_n, (total - i) // 2) + 1):
            block = keys[i:i + n]
            if any(k == "" for k in block):
                continue
            reps = 1
            while keys[i + reps * n: i + (reps + 1) * n] == block:
                reps += 1
            if reps >= min_reps(block) and (best is None or reps * n > best[0]):
                best = (reps * n, n)
        if best is None:
            keep.append(i)
            i += 1
        else:
            keep.extend(range(i, i + best[1]))  # first copy only ...
            tail_from[i + best[1] - 1] = i + best[0] - 1  # ... but it ends like the LAST copy did
            i += best[0]
    return keep, tail_from


def collapse_repeats(text: str) -> tuple[str, int]:
    """Deterministic removal of Whisper loops — no LLM involved. Returns (text, words_removed).

    Deliberately conservative, because anything removed here is gone for good:
      * whole lines (or groups of up to 4 lines) repeated back to back: collapsed when the
        block has >= 4 words and appears twice; shorter blocks only from 5 copies on;
      * inside a line, a phrase of >= 3 words repeated back to back (2 copies), a 2-word
        phrase (3 copies) or a single word (5 copies) — so ordinary speech patterns like
        "piano piano" or "no, no" are left alone for the LLM / the reader. The one exception
        is the greeting/thanks/filler words of _FILLER_KEYS ("grazie grazie", "salve salve",
        "ecco ecco"...), which collapse from 2 copies on (pass 0's rule 2).
    Comparison ignores case and edge punctuation; the surviving copy keeps its original text.
    """
    lines = text.split("\n")
    words_before = sum(len(line.split()) for line in lines)

    line_keys = [" ".join(_norm_word(w) for w in line.split()) for line in lines]
    keep, _ = _collapse_runs(line_keys, 4, lambda block: 2 if sum(len(k.split()) for k in block) >= 4 else 5)
    lines = [lines[i] for i in keep]

    out = []
    for line in lines:
        tokens = line.split()
        if len(tokens) < 2:
            out.append(line)
            continue
        keep, tail_from = _collapse_runs(
            [_norm_word(t) for t in tokens], 12,
            lambda block: 2 if len(block) == 1 and block[0] in _FILLER_KEYS else {1: 5, 2: 3}.get(len(block), 2),
        )
        if len(keep) == len(tokens):
            out.append(line)
            continue
        # The surviving copy keeps its own words but takes the punctuation that ended the
        # collapsed run, so "grazie grazie. Ecco" becomes "grazie. Ecco", not "grazie Ecco".
        out.append(" ".join(_with_tail_of(tokens[i], tokens[tail_from[i]]) if i in tail_from else tokens[i] for i in keep))

    result = "\n".join(out)
    return result, words_before - sum(len(line.split()) for line in out)


def _split_into_chunks(lines: list[str], target_words: int) -> list[list[str]]:
    """Group consecutive lines into chunks of roughly target_words words, breaking only at
    line boundaries (a pathological single line longer than target is split at spaces)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for line in lines:
        n = len(line.split())
        if n > target_words:
            if current:
                chunks.append(current)
                current, current_words = [], 0
            words = line.split()
            for j in range(0, len(words), target_words):
                chunks.append([" ".join(words[j:j + target_words])])
            continue
        if current and current_words + n > target_words:
            chunks.append(current)
            current, current_words = [], 0
        current.append(line)
        current_words += n
    if current:
        chunks.append(current)
    return chunks


def validate_cleanup_chunk(src: str, out: str, cfg: CleanupSettings) -> list[str]:
    """Reasons why `out` is not an acceptable cleaned version of `src` (empty list = OK)."""
    if not out.strip():
        return ["output is empty"]
    src_words = [w.casefold() for w in _WORD_RE.findall(src)]
    out_words = [w.casefold() for w in _WORD_RE.findall(out)]
    if not src_words:
        return []
    problems = []
    if not out_words:
        return ["output has no words"]

    ratio = len(out_words) / len(src_words)
    if ratio < cfg.min_word_ratio:
        problems.append(f"too much text removed (only {ratio:.0%} of the words kept)")
    elif ratio > cfg.max_word_ratio:
        problems.append(f"text was added ({ratio:.0%} of the input's words)")

    vocab = set(src_words)
    novel = [w for w in out_words if w not in vocab]
    if len(novel) / len(out_words) > cfg.max_novel_ratio:
        problems.append(f"{len(novel)} output words ({len(novel) / len(out_words):.1%}) do not occur in the input, e.g. {sorted(set(novel))[:5]}")

    markers = ("#", "```", "**")
    if not any(l.lstrip().startswith(markers) for l in src.splitlines()):
        if any(l.lstrip().startswith(markers) for l in out.splitlines()):
            problems.append("output contains markdown headers/bold/code fences that are not in the input")
    return problems


def _render_cleanup_prompt(template: str, chunk_no: int, chunk_total: int, in_name: str, out_name: str) -> str:
    """prompts/pass0-cleanup.md adapted to one chunk (see the note above)."""
    body = template.replace("transcript-raw.txt", in_name).replace("transcript.txt", out_name)
    header = (
        f"NOTE FOR THIS CALL: the attached file `{in_name}` is ONE SECTION (part {chunk_no} of {chunk_total}) "
        f"of a longer transcript, not the whole transcript. Clean only this section, keep it in order, and write "
        f"the result to `{out_name}`. Wherever the rules below say \"the transcript\", read it as this section. "
        f"The section may start or end mid-sentence: leave such edges exactly as they are.\n\n"
    )
    return header + body


def _load_cleanup_template(prompts_dir: Path) -> str:
    path = prompts_dir / "pass0-cleanup.md"
    if not path.exists():
        raise StagePermanentError(f"{path} not found — pass 0 reads its rules from that file")
    text = path.read_text()
    if "transcript.txt" not in text:
        logger.warning("  %s doesn't mention `transcript.txt`: the chunk's output file name is only given in the "
                       "note prepended to the prompt", path.name)
    return text


def _cleanup_one_chunk(
    ctx: RunContext, label: str, chunk_no: int, chunk_total: int, src: str, in_path: Path, out_path: Path,
    prompt_path: Path, log_path: Path, cwd: Path, model: str, template: str,
) -> str | None:
    """Run the LLM on one chunk until it returns output that passes validate_cleanup_chunk().
    Returns the cleaned text, or None after ctx.cfg.cleanup.chunk_attempts invalid/failed attempts.
    A provider "retry in Ns" rejection is waited out and does NOT use up an attempt."""
    prompt_path.write_text(_render_cleanup_prompt(template, chunk_no, chunk_total, in_path.name, out_path.name))
    attempts = rate_limit_waits = 0
    while attempts < ctx.cfg.cleanup.chunk_attempts:
        out_path.unlink(missing_ok=True)  # never let a stale file from an earlier attempt pass for this one
        try:
            ctx.runner.run(
                attachments=[in_path], prompt_file=prompt_path, model=model,
                logfile=log_path, cwd=cwd, agent="notes-writer",
            )
        except OpencodeRateLimit as e:
            rate_limit_waits += 1
            if e.retry_after > ctx.cfg.rate.max_wait_seconds or rate_limit_waits > ctx.cfg.rate.max_waits:
                raise StagePermanentError(
                    f"{label}: provider quota exhausted (asked to wait {e.retry_after:.0f}s, "
                    f"{rate_limit_waits} rate-limit waits so far) — retry later; finished chunks are kept in {cwd}"
                ) from e
            wait = e.retry_after + 2
            logger.warning("  %s: rate limited — waiting %.0fs as requested by the provider", label, wait)
            time.sleep(wait)
            ctx.runner.throttle.reset()
            continue
        except (RuntimeError, TimeoutError) as e:
            attempts += 1
            logger.warning("  %s: attempt %d/%d failed (%s)", label, attempts, ctx.cfg.cleanup.chunk_attempts, e)
            continue

        attempts += 1
        if not out_path.exists():
            logger.warning("  %s: attempt %d/%d — opencode finished but did not write %s", label, attempts, ctx.cfg.cleanup.chunk_attempts, out_path.name)
            continue
        out = out_path.read_text()
        problems = validate_cleanup_chunk(src, out, ctx.cfg.cleanup)
        if not problems:
            return out
        logger.warning("  %s: attempt %d/%d rejected — %s", label, attempts, ctx.cfg.cleanup.chunk_attempts, "; ".join(problems))
    return None


def stage_cleanup(ctx: RunContext) -> None:
    """Pass 0: turn transcript-raw.txt into transcript.txt.

    A. collapse_repeats(): exact Whisper loops removed in plain Python (free, deterministic).
       Saved as transcript-deduped.txt so you can diff it against the raw file.
    B. The result is split into ~cfg.cleanup.chunk_words-word chunks; each chunk goes through
       one small opencode call (agent "notes-writer": write-only, no tools to wander with)
       that fixes filler repeats and obvious typos.
    C. Every chunk's output is VALIDATED (non-empty, word count within bounds, no
       invented vocabulary/markdown). A chunk that doesn't validate is retried; if it
       still doesn't, that chunk keeps its de-looped, un-cleaned text. Under-cleaning is
       acceptable, losing content is not.

    Why chunks: the whole-file approach re-sent a ~15K-token transcript on every agent step
    and needed the model to emit all of it in one answer, which is what hit the quota.
    Per-chunk calls are small, predictable, paced by TokenThrottle, and resumable: chunk
    files live in workdir/cleanup/, and a chunk whose input is unchanged and whose output
    still validates is reused on re-runs (delete that folder to force a full redo).
    """
    cfg, workdir = ctx.cfg, ctx.workdir
    model = cfg.models.cleanup
    logger.info("  model=%s", model)

    raw_path = workdir / "transcript-raw.txt"
    final_path = workdir / "transcript.txt"
    final_path.unlink(missing_ok=True)  # a stale transcript from an earlier run must not satisfy _require()

    raw = raw_path.read_text()
    raw_words = len(raw.split())

    deduped, removed = collapse_repeats(raw)
    (workdir / "transcript-deduped.txt").write_text(deduped)
    logger.info("  step A (no LLM): removed %d words of exact repeats/loops (%d -> %d words)", removed, raw_words, raw_words - removed)

    chunks = _split_into_chunks(deduped.split("\n"), cfg.cleanup.chunk_words)
    chunk_dir = workdir / "cleanup"
    chunk_dir.mkdir(exist_ok=True)
    template = _load_cleanup_template(cfg.paths.prompts)
    logger.info("  step B: %d chunk(s) of up to ~%d words", len(chunks), cfg.cleanup.chunk_words)

    pieces: list[str] = []
    fallbacks: list[int] = []
    reused = unchanged = 0
    for idx, chunk_lines in enumerate(chunks):
        label = f"chunk {idx + 1}/{len(chunks)}"
        stem = f"chunk_{idx:03d}"
        in_path, out_path = chunk_dir / f"{stem}.in.txt", chunk_dir / f"{stem}.out.txt"
        src = "\n".join(chunk_lines) + "\n"

        if in_path.exists() and out_path.exists() and in_path.read_text() == src:
            cached = out_path.read_text()
            if not validate_cleanup_chunk(src, cached, cfg.cleanup):
                logger.info("  %s: reusing validated output from an earlier run", label)
                pieces.append(cached.rstrip("\n"))
                reused += 1
                continue

        in_path.write_text(src)
        logger.info("  %s: %d words", label, len(src.split()))
        cleaned = _cleanup_one_chunk(
            ctx, label, idx + 1, len(chunks), src, in_path, out_path, chunk_dir / f"{stem}.prompt.md", chunk_dir / f"{stem}.log",
            chunk_dir, model, template,
        )
        if cleaned is None:
            logger.warning("  %s: no valid LLM output after %d attempts — keeping its de-looped text unchanged (nothing dropped)", label, cfg.cleanup.chunk_attempts)
            fallbacks.append(idx + 1)
            pieces.append(src.rstrip("\n"))
            continue
        if cleaned.strip() == src.strip():
            unchanged += 1
        pieces.append(cleaned.rstrip("\n"))

    final_path.write_text("\n".join(pieces) + "\n")
    final_words = len(final_path.read_text().split())
    logger.info(
        "  cleanup summary: raw %d -> de-looped %d -> final %d words | %d chunk(s): %d reused, %d returned unchanged, %d fell back to un-cleaned text",
        raw_words, raw_words - removed, final_words, len(chunks), reused, unchanged, len(fallbacks),
    )
    if fallbacks:
        logger.warning("  chunk(s) %s were NOT LLM-cleaned (see %s/chunk_NNN.log); re-run '--from-stage cleanup' to retry just those", fallbacks, chunk_dir)
    if removed == 0 and len(fallbacks) + unchanged + reused == len(chunks):
        logger.warning("  pass 0 changed nothing at all — fine if the transcript was already clean, but worth a glance at %s", chunk_dir)

    _require(final_path, "Pass 0")
    logger.info("  wrote %s", final_path)


# =========================================================================================
# 8. PASSES 1-4  (future: conf_notes/passes.py)
# =========================================================================================

def _require(path: Path, stage_name: str) -> None:
    if not path.exists() or not path.read_text().strip():
        raise RuntimeError(f"{stage_name} did not produce {path.name} — check the matching pass log in {path.parent} before continuing.")
    # Belt-and-suspenders: whatever opencode/the model wrote, make sure it doesn't
    # contain a line opencode itself can't fully read back in a later stage (see
    # ensure_no_long_lines() for why this can't be guaranteed by prompt instructions
    # alone).
    ensure_no_long_lines(path)


def stage_extract(ctx: RunContext) -> None:
    workdir = ctx.workdir
    # Agent "notes-writer": transcript.txt is already fully inlined via --file and this
    # stage never opens any other file, so `read` and every other tool stay denied.
    (workdir / "extract.md").unlink(missing_ok=True)  # a stale file must not satisfy _require()
    ctx.runner.run(
        attachments=[workdir / "transcript.txt"], prompt_file=ctx.prompt("pass1-extract.md"),
        model=ctx.cfg.models.extract, logfile=workdir / "pass1.log", cwd=workdir, agent="notes-writer",
    )
    _require(workdir / "extract.md", "Pass 1")
    logger.info("  wrote %s", workdir / "extract.md")


def stage_enrich(ctx: RunContext) -> None:
    """Pass 2: look up each extracted topic/entity and write a short background note
    for it. Batch size adapts to the enrich model (see Models.enrich): a cloud model
    gets enrich.batch_size items per opencode call, a local (ollama/...) model gets
    enrich.batch_size_local (1), because each call's own web-search results otherwise
    blow past a local context window.

    Per-call outputs (one .md per batch, or per item when local) live in their own
    workdir/enrich/ subfolder so a many-batches local run doesn't clutter workdir with
    hundreds of loose files; they're merged into background.md at the end either way.
    """
    cfg, workdir = ctx.cfg, ctx.workdir
    model = cfg.models.enrich
    batch_size = cfg.enrich.batch_size_local if is_local_model(model) else cfg.enrich.batch_size
    logger.info("  batched enrichment (model=%s, batch_size=%d)", model, batch_size)

    extract_path = workdir / "extract.md"
    if not extract_path.exists():
        raise StagePermanentError("extract.md not found — cannot run enrichment")

    items = _parse_extract_items(extract_path)
    if not items:
        raise StagePermanentError(
            "No topics/entities found in extract.md — check its 'Topics & concepts "
            "mentioned' / 'Named entities' sections actually have content under them"
        )

    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    logger.info("  splitting %d items into %d batches of up to %d", len(items), len(batches), batch_size)

    enrich_dir = workdir / "enrich"
    enrich_dir.mkdir(exist_ok=True)

    base_prompt = ctx.prompt("pass2-enrich.md").read_text()
    batch_files = []

    for idx, batch in enumerate(batches):
        batch_file = enrich_dir / f"item_{idx:04d}.md"
        prompt_file = enrich_dir / f"item_{idx:04d}_prompt.md"

        items_block = "\n".join(f"- {item}" for item in batch)
        prompt_text = base_prompt.replace("{{ITEMS}}", items_block).replace("{{OUTPUT_FILE}}", batch_file.name)

        label = batch[0] if batch_size == 1 else f"{len(batch)} items"

        # Resume: if this exact batch (same prompt) already produced output — e.g. in a previous
        # attempt of this stage that failed on a later batch — don't pay for it again.
        if (batch_file.exists() and batch_file.read_text().strip()
                and prompt_file.exists() and prompt_file.read_text() == prompt_text):
            logger.info("  batch %d/%d: %s -> reusing %s from an earlier attempt", idx + 1, len(batches), label, batch_file.name)
            batch_files.append(batch_file)
            continue

        prompt_file.write_text(prompt_text)
        batch_file.unlink(missing_ok=True)  # a stale/empty file must not satisfy _require()
        logger.info("  batch %d/%d: %s -> %s", idx + 1, len(batches), label, batch_file.name)

        # cwd=enrich_dir (not workdir) so opencode's relative "write to {{OUTPUT_FILE}}"
        # lands batch_file.name inside enrich/, matching where batch_file actually is.
        # Agent "notes-enrich": web fetch/search + edit, nothing else. Pacing against the
        # provider's TPM limit is done inside OpencodeRunner.run() (TokenThrottle).
        ctx.runner.run(
            attachments=[], prompt_file=prompt_file, model=model,
            logfile=enrich_dir / f"item_{idx:04d}.log", cwd=enrich_dir, agent="notes-enrich",
            # pass2-enrich.md tries up to 4 sources per topic (it.wikipedia, wiktionary, arXiv,
            # DuckDuckGo): leave room for that so the step cap never cuts a batch off before it
            # writes its file (the default cap of 40 would be too tight for batch_size=8).
            max_steps=5 * len(batch) + 8, est_calls=2 * len(batch) + 3,
        )

        _require(batch_file, f"Pass 2 batch {idx + 1}/{len(batches)}")  # don't proceed until usable output exists
        batch_files.append(batch_file)
        logger.info("  batch %d/%d completed successfully", idx + 1, len(batches))

    combined = []
    for bf in batch_files:
        content = bf.read_text().strip()
        if not content:
            raise RuntimeError(f"Pass 2 produced an empty batch file: {bf.name}")
        combined.append(content)

    if not combined:
        raise RuntimeError("Pass 2 produced no background content in any batch")

    background_md = workdir / "background.md"
    background_md.write_text("\n\n".join(combined))
    logger.info("  wrote %s by combining %d validated batch files", background_md, len(batch_files))
    _require(background_md, "Pass 2")


def stage_synthesize(ctx: RunContext) -> None:
    workdir = ctx.workdir
    date_str = derive_date_from_filename(ctx.audio)
    logger.debug("  using date_str=%r for notes.md header (derived from filename/mtime, not asked of opencode)", date_str)
    # Deliberately NOT attaching transcript.txt: extract.md's "Speaker's claims &
    # arguments" section is already a compressed-but-complete, sentence-by-sentence
    # record of the talk (see pass1-extract.md's no-ellipsis rule), so re-sending the
    # full transcript on top would just burn tokens for content opencode already has.
    # Agent "notes-writer": both attachments are already fully inlined, and this stage
    # never needs to open any other file.
    (workdir / "notes.md").unlink(missing_ok=True)  # a stale file must not satisfy _require()
    ctx.runner.run(
        attachments=[workdir / "extract.md", workdir / "background.md"],
        prompt_file=ctx.prompt("pass3-synthesize.md"), model=ctx.cfg.models.synthesize,
        logfile=workdir / "pass3.log", cwd=workdir, agent="notes-writer", substitutions={"{{DATA}}": date_str},
    )
    _require(workdir / "notes.md", "Pass 3")
    logger.info("  wrote %s", workdir / "notes.md")


def stage_verify(ctx: RunContext) -> None:
    workdir, max_shrink = ctx.workdir, ctx.cfg.verify_max_shrink
    notes = workdir / "notes.md"
    snapshot = workdir / "notes.pre-verify.md"
    shutil.copy2(notes, snapshot)
    words_before = len(snapshot.read_text().split())

    # pass4-verify.md says "search the web if you need" and "rewrite notes.md in place", so the
    # agent "notes-editor" gets read + edit + websearch/webfetch (nothing else). notes.md is
    # deliberately NOT attached: the agent has to `read` it anyway (opencode refuses to edit an
    # existing file it hasn't read), and attaching it too would put two copies of the whole
    # document into the context of every one of its many steps.
    ctx.runner.run(
        attachments=[], read_files=[notes],
        prompt_file=ctx.prompt("pass4-verify.md"), model=ctx.cfg.models.verify,
        logfile=workdir / "pass4.log", cwd=workdir, agent="notes-editor",
    )
    _require(notes, "Pass 4")

    # The prompt's one critical rule is "do not delete content". Enforce it: if the edited
    # file lost more than verify_max_shrink of its words, keep the pre-verify version instead.
    words_after = len(notes.read_text().split())
    if words_after < words_before * (1 - max_shrink):
        rejected = workdir / "notes.verify-rejected.md"
        shutil.copy2(notes, rejected)
        shutil.copy2(snapshot, notes)
        logger.warning(
            "  verify shrank notes.md from %d to %d words (> %d%% loss) — restored the pre-verify version; "
            "the rejected edit is kept as %s", words_before, words_after, int(max_shrink * 100), rejected.name,
        )
    else:
        logger.info("  updated %s (%d -> %d words)", notes, words_before, words_after)


# =========================================================================================
# 9. RENDER  (future: conf_notes/render.py + assets/*.css)
# =========================================================================================

# --- PDF styling (WeasyPrint: pure HTML/CSS -> PDF, no LaTeX install required) ---------
# Margins, font sizes and font family are Config.render values; the CSS below is a template.
# Spacing tightened for density (smaller line-height and margins throughout) so notes.pdf
# fits more per page instead of reading like a print-book layout.
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
    line-height: 1.4;
    color: #1a1a1a;
}}

h1 {{
    string-set: doc-title content();
    font-size: 20pt;
    font-weight: 700;
    color: #0f1f44;
    border-bottom: 3px solid #0f1f44;
    padding-bottom: 6px;
    margin: 0 0 14px 0;
    page-break-after: avoid;
}}

h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: #0f1f44;
    border-left: 4px solid #3762cc;
    padding-left: 9px;
    margin: 16px 0 6px 0;
    page-break-after: avoid;
}}

h3 {{
    font-size: 11.5pt;
    font-weight: 700;
    color: #1e3468;
    margin: 10px 0 4px 0;
    page-break-after: avoid;
}}

p {{
    margin: 0 0 6px 0;
    text-align: justify;
    orphans: 3;
    widows: 3;
}}

ul, ol {{
    margin: 3px 0 8px 0;
    padding-left: 20px;
}}

li {{
    margin-bottom: 2px;
    line-height: 1.35;
}}

blockquote {{
    border-left: 4px solid #3762cc;
    background: #f0f4ff;
    padding: 6px 12px;
    margin: 8px 0;
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
    margin: 8px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}}

th {{
    background: #0f1f44;
    color: white;
    padding: 5px 10px;
    text-align: left;
    font-weight: 600;
}}

td {{
    border: 1px solid #ccd;
    padding: 4px 10px;
    vertical-align: top;
}}

tr:nth-child(even) td {{ background: #f7f8fc; }}

hr {{
    border: none;
    border-top: 1px solid #dde;
    margin: 14px 0;
}}
"""

# --- HTML styling (screen/mobile: separate from PDF_CSS, which is tuned for a printed
# A4 page — @page margins, pt sizes and justified text all work against you on a phone) --
HTML_CSS = """\
:root {
    color-scheme: light dark;
}

* { box-sizing: border-box; }

html { -webkit-text-size-adjust: 100%; }

body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1rem;
    line-height: 1.7;
    color: #1a1a1a;
    background: #ffffff;
    max-width: 46rem;
    margin: 0 auto;
    padding: 1.5rem 1.25rem 3rem;
}

h1 {
    font-size: clamp(1.4rem, 5vw, 1.8rem);
    font-weight: 700;
    color: #0f1f44;
    border-bottom: 3px solid #0f1f44;
    padding-bottom: 0.5rem;
    margin: 0 0 1.5rem 0;
    line-height: 1.3;
}

h2 {
    font-size: clamp(1.1rem, 4vw, 1.25rem);
    font-weight: 700;
    color: #0f1f44;
    border-left: 4px solid #3762cc;
    padding-left: 0.6rem;
    margin: 2rem 0 0.6rem 0;
    line-height: 1.3;
}

h3 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3468;
    margin: 1.2rem 0 0.5rem 0;
}

p {
    margin: 0 0 0.75rem 0;
    text-align: left; /* justify creates ragged word-spacing on narrow phone columns */
}

ul, ol {
    margin: 0.4rem 0 0.9rem 0;
    padding-left: 1.4rem;
}

li {
    margin-bottom: 0.35rem;
    line-height: 1.6;
}

blockquote {
    border-left: 4px solid #3762cc;
    background: #f0f4ff;
    padding: 0.6rem 1rem;
    margin: 1rem 0;
    color: #1e3468;
    font-style: italic;
    border-radius: 0 4px 4px 0;
}

blockquote p { margin: 0; }

code {
    font-family: "Courier New", Courier, monospace;
    font-size: 0.85em;
    background: #f4f4f8;
    padding: 0.1em 0.35em;
    border-radius: 3px;
    color: #c0392b;
    word-break: break-word;
}

strong { font-weight: 700; }
em { font-style: italic; color: #333; }

a {
    color: #3762cc;
    text-decoration: none;
}

/* Tables can be wider than a phone screen; let them scroll horizontally within
   themselves instead of overflowing (breaking) the whole page layout. */
.table-wrap {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    margin: 1rem 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

th {
    background: #0f1f44;
    color: white;
    padding: 0.5rem 0.75rem;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}

td {
    border: 1px solid #ccd;
    padding: 0.45rem 0.75rem;
    vertical-align: top;
}

tr:nth-child(even) td { background: #f7f8fc; }

hr {
    border: none;
    border-top: 1px solid #dde;
    margin: 1.5rem 0;
}

img { max-width: 100%; height: auto; }

@media (prefers-color-scheme: dark) {
    body { background: #14161c; color: #e8e8ea; }
    h1, h2 { color: #a9c1ff; }
    h1 { border-bottom-color: #a9c1ff; }
    h2 { border-left-color: #7fa0ff; }
    h3 { color: #cfd9ff; }
    blockquote { background: #1d2233; color: #c9d3ff; border-left-color: #7fa0ff; }
    code { background: #23262f; color: #ff8a75; }
    th { background: #1d2848; }
    td { border-color: #33384a; }
    tr:nth-child(even) td { background: #1a1c24; }
    a { color: #8fb0ff; }
    hr { border-top-color: #33384a; }
}
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""


def _markdown_to_html(md_text: str) -> str:
    markdown, _ = _try_import("markdown")
    if markdown is None:
        raise RuntimeError("markdown not installed. Run: pip install markdown")
    md = markdown.Markdown(extensions=["markdown.extensions.extra", "markdown.extensions.sane_lists", "markdown.extensions.smarty"])
    return md.convert(md_text)


def _text_to_html(text: str, title: str) -> str:
    """Render plain transcript text as simple justified paragraphs, with a title heading
    so the PDF header (string-set on h1) picks it up.

    Raw transcript text is often one sentence per line with no blank line between them
    (that's just how whisper/opencode wrote it out) — those are NOT paragraph breaks, so
    a single \\n inside a paragraph is collapsed to a space and left to wrap normally.
    Only a blank line (\\n\\s*\\n) starts a new <p>."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    body = [f"<h1>{html.escape(title)}</h1>"]
    for para in paragraphs:
        flat = re.sub(r"\s*\n\s*", " ", para).strip()
        body.append(f"<p>{html.escape(flat)}</p>")
    return "\n".join(body)


def _render_pdf(html_body: str, dest: Path, render: RenderSettings, font_size: str | None = None) -> None:
    weasyprint, error = _try_import("weasyprint")
    if weasyprint is None:
        raise RuntimeError(
            f"weasyprint could not be imported ({error}). This is almost always a "
            f"native-library path issue, not a missing pip install — see check_environment()'s message "
            f"at startup for the fix."
        )
    css = PDF_CSS.format(margin=render.margin, font_size=font_size or render.font_size, font_family=render.font_family)
    weasyprint.HTML(string=HTML_TEMPLATE.format(css=css, body=html_body)).write_pdf(str(dest))
    logger.info("  wrote %s", dest)


def _render_html(html_body: str, dest: Path) -> None:
    """Write a standalone, self-contained HTML file, styled for on-screen/mobile reading
    (HTML_CSS) rather than reusing the print-tuned PDF_CSS: real padding instead of
    @page margins (ignored by browsers), rem-based responsive type instead of fixed pt
    sizes, left-aligned instead of justified text, a max-width for wide screens, a
    prefers-color-scheme dark mode, and horizontally-scrollable tables."""
    html_body = re.sub(r"(<table\b.*?</table>)", r'<div class="table-wrap">\1</div>', html_body, flags=re.DOTALL)
    dest.write_text(HTML_TEMPLATE.format(css=HTML_CSS, body=html_body), encoding="utf-8")
    logger.info("  wrote %s", dest)


def stage_pdf(ctx: RunContext) -> None:
    cfg, workdir, run_name = ctx.cfg, ctx.workdir, ctx.run_name

    notes_html_body = _markdown_to_html((workdir / "notes.md").read_text())
    _render_pdf(notes_html_body, workdir / "notes.pdf", cfg.render)
    _render_html(notes_html_body, workdir / "notes.html")

    transcript_text = (workdir / "transcript.txt").read_text()
    _render_pdf(
        _text_to_html(transcript_text, f"Trascrizione — {run_name}"), workdir / "transcript.pdf",
        cfg.render, font_size=cfg.render.transcript_font_size,
    )

    out_dir = cfg.paths.output
    out_dir.mkdir(parents=True, exist_ok=True)
    dests = {
        "notes.md": out_dir / f"{run_name}-notes.md",
        "notes.pdf": out_dir / f"{run_name}-notes.pdf",
        "notes.html": out_dir / f"{run_name}-notes.html",
        "transcript.pdf": out_dir / f"{run_name}-transcript.pdf",
    }
    for src_name, dest in dests.items():
        shutil.copy2(workdir / src_name, dest)
    logger.info("  copied %d output files to %s", len(dests), out_dir)
    for dest in dests.values():
        logger.info("    %s", dest)


# =========================================================================================
# 10. PIPELINE  (future: conf_notes/pipeline.py)
# =========================================================================================

@dataclass(frozen=True)
class RunContext:
    """Everything one recording's stages need, passed explicitly instead of via globals."""
    cfg: Config
    runner: OpencodeRunner
    audio: Path
    workdir: Path
    run_name: str

    def prompt(self, name: str) -> Path:
        return self.cfg.paths.prompts / name


@dataclass(frozen=True)
class Stage:
    """One pipeline step. `requires` = files (in the run's workdir) that must already exist
    to START at this stage, i.e. what --from-stage needs (irrelevant for a normal full run,
    where the prior stage just wrote them).

    Only "cleanup" and "extract" need the (raw/cleaned) transcript text. From "synthesize"
    onward, extract.md's "Speaker's claims & arguments" section is already a
    compressed-but-complete sentence-by-sentence record of the talk, so the full transcript
    isn't re-attached to later opencode calls (it's long, and re-sending it burns tokens
    for no extra information opencode doesn't already have via extract.md)."""
    name: str
    title: str
    run: Callable[[RunContext], None]
    requires: tuple[str, ...] = ()


# The single source of truth for stage order, CLI choices, resume prerequisites and the
# "[n/N]" progress labels. To add a stage: write stage_xxx(ctx) and add one line here.
STAGES: tuple[Stage, ...] = (
    Stage("transcribe", "Transcribing audio", stage_transcribe),
    Stage("cleanup", "Pass 0: transcript cleanup (de-loop in Python + chunked LLM pass)", stage_cleanup,
          requires=("transcript-raw.txt",)),
    Stage("extract", "Pass 1: extraction (transcript-only, no outside knowledge)", stage_extract,
          requires=("transcript.txt",)),
    Stage("enrich", "Pass 2: multi-source enrichment", stage_enrich,
          requires=("extract.md",)),
    Stage("synthesize", "Pass 3: synthesis into full academic notes (no info dropped)", stage_synthesize,
          requires=("extract.md", "background.md")),
    Stage("verify", "Pass 4: verification (fixes doubtful sentences in place, deletes nothing)", stage_verify,
          requires=("notes.md",)),
    Stage("pdf", "Rendering PDF/HTML outputs (WeasyPrint)", stage_pdf,
          requires=("notes.md", "transcript.txt")),
)
STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in STAGES)
assert len(set(STAGE_NAMES)) == len(STAGE_NAMES), "duplicate stage name in STAGES"


def _run_stage_with_retry(ctx: RunContext, stage: Stage):
    """Run one stage, retrying up to cfg.retry.stage_attempts times with a
    cfg.retry.stage_sleep_seconds pause between attempts if it raises.

    Retries RuntimeError (opencode exited non-zero, or _require() found a stage "did
    not produce" its expected output file) and TimeoutError (opencode exceeded
    cfg.opencode_timeout_seconds) — both can plausibly succeed on a later attempt.
    StagePermanentError is NOT retried: it means the failure is deterministic given
    the current on-disk inputs (e.g. stage_enrich() finding no parseable items in
    extract.md), so retrying without anything changing would just fail identically
    three times in a row. Re-raises the last exception if every attempt fails, so
    process_file()'s own error handling (logging + sys.exit / batch "failures" list)
    still runs exactly as before.
    """
    retry, rate = ctx.cfg.retry, ctx.cfg.rate
    last_exc: Exception | None = None
    for attempt in range(1, retry.stage_attempts + 1):
        try:
            return stage.run(ctx)
        except StagePermanentError:
            raise
        except (RuntimeError, TimeoutError) as e:
            last_exc = e
            if attempt < retry.stage_attempts:
                if isinstance(e, OpencodeRateLimit):
                    # Wait exactly what the provider asked for (+2s margin) instead of a fixed guess.
                    delay = min(e.retry_after + 2, rate.max_wait_seconds)
                else:
                    delay = retry.stage_sleep_seconds
                logger.warning(
                    "  [%s] attempt %d/%d failed (%s) — sleeping %.0fs before retry",
                    stage.name, attempt, retry.stage_attempts, e, delay,
                )
                time.sleep(delay)
                ctx.runner.throttle.reset()
            else:
                logger.error("  [%s] failed after %d attempts", stage.name, retry.stage_attempts)
    raise last_exc


def _check_resume_prereqs(workdir: Path, stage: Stage) -> None:
    """When resuming with --from-stage, fail fast with a clear message if the files that
    stage depends on aren't already in workdir, instead of letting opencode run against
    a missing --file attachment."""
    missing = [f for f in stage.requires if not (workdir / f).exists()]
    if missing:
        raise RuntimeError(f"--from-stage {stage.name} requires {', '.join(missing)} to already exist in {workdir}, but they don't. Run from an earlier stage first.")


def _run_pipeline_stages(ctx: RunContext, from_stage: str) -> None:
    """Run stages from_stage..end for one file. Each stage already retries transient
    failures internally (_run_stage_with_retry); if a stage still fails after that, this
    raises and lets the caller decide whether to restart the whole sequence."""
    start = STAGE_NAMES.index(from_stage)
    if start > 0:
        _check_resume_prereqs(ctx.workdir, STAGES[start])

    for number, stage in enumerate(STAGES, 1):
        if number - 1 < start:
            continue
        logger.info("[%d/%d] %s", number, len(STAGES), stage.title)
        _run_stage_with_retry(ctx, stage)


def process_file(audio: Path, cfg: Config, runner: OpencodeRunner, from_stage: str) -> None:
    run_name = audio.stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    logfile = cfg.paths.logs / f"{timestamp}_{run_name}.log"
    setup_logging(logfile, cfg.verbose)

    logger.info("=== Processing %s ===", audio.name)
    logger.debug("Log file: %s", logfile)

    workdir = cfg.paths.work / run_name
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(cfg=cfg, runner=runner, audio=audio, workdir=workdir, run_name=run_name)

    try:
        _run_pipeline_stages(ctx, from_stage)
    except (RuntimeError, TimeoutError) as e:
        # A stage exhausted its own retries. This doesn't necessarily mean the
        # situation is truly unfixable — re-running the earlier stages from scratch can
        # produce different (working) intermediate output, since these are LLM calls,
        # not deterministic functions (e.g. a re-run of "extract" might phrase its
        # output in a way that parses fine, even though the previous run's didn't). So:
        # go back and redo the whole pipeline for this file from from_stage, one time,
        # before giving up. Only one such restart per file, so a genuinely broken
        # environment (bad credentials, no network, etc.) still fails after 2x the work
        # instead of looping forever.
        logger.warning(
            "  Pipeline failed even after per-stage retries (%s). Restarting the whole "
            "pipeline for %s from '%s' — one retry allowed before giving up.",
            e, audio.name, from_stage,
        )
        restart_from = from_stage
        raw_transcript = workdir / "transcript-raw.txt"
        if from_stage == "transcribe" and raw_transcript.exists() and raw_transcript.read_text().strip():
            restart_from = "cleanup"  # don't redo an hour of Whisper for a failure in a later stage
            logger.info("  transcript-raw.txt already exists — restarting from 'cleanup' instead of re-transcribing")
        time.sleep(cfg.retry.stage_sleep_seconds)
        _run_pipeline_stages(ctx, restart_from)

    if not cfg.keep_input and from_stage == "transcribe":
        try:
            inside_input_dir = audio.parent.resolve() == cfg.paths.input.resolve()
        except OSError:
            inside_input_dir = False
        if inside_input_dir:
            processed_dir = audio.parent / "processed"
            processed_dir.mkdir(exist_ok=True)
            dest = processed_dir / audio.name
            shutil.move(str(audio), str(dest))
            logger.info("Moved %s -> %s", audio.name, dest)
        else:
            logger.debug("Not moving %s to processed/ (outside the input folder)", audio)

    logger.info("Done with %s.", audio.name)
    logger.info("  4 output files in: %s", cfg.paths.output)
    logger.info("  Intermediate files kept in: %s", workdir)


# =========================================================================================
# 11. CLI  (future: conf_notes/cli.py)
# =========================================================================================

def check_environment(cfg: Config) -> None:
    """Ensure the working folders exist, and fail fast with a clear message if a
    required binary/package is missing — rather than discovering it after transcribing a
    1-hour recording."""
    cfg.paths.ensure()

    problems = []
    if shutil.which("opencode") is None:
        problems.append("`opencode` not found on PATH — required for cleanup/extract/enrich/synthesize/verify. Install it, then run `opencode auth login` for your provider(s).")

    weasyprint, weasyprint_error = _try_import("weasyprint")
    if weasyprint is None:
        problems.append(
            f"weasyprint could not be loaded ({weasyprint_error}). On macOS this is almost always "
            "WeasyPrint failing to find its native libraries (pango/cairo/gdk-pixbuf) installed via "
            "Homebrew, not a missing `pip install`. Fix:\n"
            "      brew install cairo pango gdk-pixbuf libffi\n"
            "      export DYLD_FALLBACK_LIBRARY_PATH=\"$(brew --prefix)/lib:$DYLD_FALLBACK_LIBRARY_PATH\"\n"
            "    Add that export to your ~/.zshrc so it's set for every future run. Full troubleshooting: "
            "https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#troubleshooting"
        )
    if _try_import("markdown")[0] is None:
        problems.append("`markdown` not installed — required to render notes.md. Run: pip install markdown")
    if _try_import("mlx_whisper")[0] is None:
        problems.append("mlx-whisper isn't installed — it's the only transcription backend now. Run: pip install mlx-whisper")
    elif shutil.which("ffmpeg") is None:
        problems.append("`ffmpeg` not found on PATH — mlx-whisper needs it to decode audio. Run: brew install ffmpeg")

    if problems:
        print("Setup problem(s) found before starting:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)


def build_parser(defaults: Config) -> argparse.ArgumentParser:
    """`defaults` is the Config built from the environment; it is only used to show the
    effective defaults in --help. CLI values are applied afterwards via Config.with_overrides()."""
    m = defaults.models
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=None, help="Process a single audio file instead of scanning the input folder")
    parser.add_argument(
        "--model-fast", default=None,
        help=f"Model for the extract AND enrich passes, as provider/model (env: OPENCODE_MODEL_FAST; "
             f"current: extract={m.extract}, enrich={m.enrich}).",
    )
    parser.add_argument(
        "--model-cleanup", default=None,
        help=f"Model for the cleanup pass, as provider/model (env: OPENCODE_MODEL_CLEANUP; current: {m.cleanup}). "
             f"Use a local ollama/... model to keep that pass fully offline.",
    )
    parser.add_argument(
        "--model-strong", default=None,
        help=f"Model for the synthesize AND verify passes, as provider/model (env: OPENCODE_MODEL_STRONG; "
             f"current: synthesize={m.synthesize}, verify={m.verify}).",
    )
    parser.add_argument("--whisper-model", default=None, help=f"MLX Whisper model (current: {m.whisper})")
    parser.add_argument("--from-stage", choices=STAGE_NAMES, default="transcribe", help="Resume from this stage (requires --file; reuses existing intermediate files)")
    parser.add_argument("--keep-input", action="store_true", help="Don't move processed files into <input>/processed/")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print debug-level logs to console")
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        base = Config.from_env()
    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(2)

    parser = build_parser(base)
    args = parser.parse_args(argv)
    if args.from_stage != "transcribe" and args.file is None:
        parser.error("--from-stage other than 'transcribe' requires --file")

    cfg = base.with_overrides(
        model_fast=args.model_fast, model_strong=args.model_strong, model_cleanup=args.model_cleanup,
        whisper_model=args.whisper_model, keep_input=args.keep_input or None, verbose=args.verbose or None,
    )
    configure_environment(cfg)   # before anything imports mlx_whisper / huggingface_hub
    check_environment(cfg)
    runner = OpencodeRunner(cfg)

    if args.file is not None:
        audio = args.file.resolve()
        if args.from_stage == "transcribe" and not audio.exists():
            parser.error(f"Audio file not found: {audio}")
        try:
            process_file(audio, cfg, runner, args.from_stage)
        except Exception as e:
            logger.error("%s", e)
            sys.exit(1)
        return

    # Batch mode: scan the input folder, process one file at a time.
    audio_files = discover_audio_files(cfg.paths.input, cfg.audio_extensions)
    if not audio_files:
        print(f"No audio files found in {cfg.paths.input}")
        return

    print(f"Found {len(audio_files)} file(s) in {cfg.paths.input}. Processing one at a time...")
    failures = []
    for audio in audio_files:
        try:
            process_file(audio, cfg, runner, "transcribe")
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