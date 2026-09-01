#!/usr/bin/env python3
"""
conf_notes.py — batch pipeline: scans an input folder for recordings and,
one at a time, turns each into a transcript PDF, a notes markdown file, and
a notes PDF (3 files per recording).

Transcription tries Gemini 3.5 Transcribe (cloud) first if a GEMINI_API_KEY
/ GOOGLE_API_KEY is set, and falls back to local mlx-whisper on any failure
(no key, network issue, rate limit, file too long, etc). That transcript
feeds a 4-pass pi pipeline:
  1. extract     — pull claims/topics/quotes from the transcript only
  2. enrich       — look up each topic across free, no-key sources
  3. synthesize   — combine into full academic notes, nothing dropped
  4. verify       — flag (never delete) anything not traceable to a source

Folder layout (created automatically next to the script if missing):
  input/   — drop recordings here; processed files are moved to
             input/processed/ so reruns don't redo them
  output/  — final "<name>-notes.md", "<name>-notes.pdf", and
             "<name>-transcript.pdf" land here (3 files per recording)
  logs/    — one log file per run, named yyyy-MM-dd_hh-mm_<name>.log

Requirements:
    pip install mlx-whisper google-genai weasyprint markdown
    brew install cairo pango gdk-pixbuf libffi ffmpeg   # WeasyPrint + mlx-whisper native deps
    pi CLI already installed and configured with your provider fallbacks
    export GEMINI_API_KEY=...           # optional: enables cloud transcription

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
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# --- Optional third-party imports, resolved once at startup ----------------
# Kept optional (not a hard `import` at call time) so a missing backend
# degrades gracefully instead of crashing the whole script — e.g. Gemini
# should still work if mlx-whisper isn't installed, and vice versa.
# WeasyPrint in particular can fail with OSError (not ImportError) when it
# can't dlopen its native libraries (pango/cairo/gdk-pixbuf) — see
# check_environment() below for the macOS fix.
try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None

try:
    from google import genai
except ImportError:
    genai = None

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

SCRIPT_DIR = Path(__file__).resolve().parent
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

STAGES = ["transcribe", "extract", "enrich", "synthesize", "verify", "pdf"]
WHISPER_MODEL_DEFAULT = "mlx-community/whisper-large-v3-turbo"
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".mp4", ".aac", ".flac", ".ogg", ".mov"}

# Files each stage needs already present when resuming with --from-stage
# (irrelevant for a normal full run, where the prior stage just wrote them).
STAGE_REQUIRES = {
    "extract": ["transcript.txt"],
    "enrich": ["extract.md"],
    "synthesize": ["transcript.txt", "extract.md", "background.md"],
    "verify": ["notes.md", "transcript.txt", "background.md"],
    "pdf": ["notes.md", "transcript.txt"],
}

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


def run_pi(
    *,
    attachments: list[Path],
    prompt_file: Path,
    tools: str,
    model: str | None,
    logfile: Path,
    cwd: Path,
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
    cmd = ["pi", "-p"]
    cmd += [f"@{p}" for p in attachments]
    cmd += [prompt_text]
    cmd += ["--tools", tools]
    if model:
        cmd += ["--model", model]

    logger.debug("Running (cwd=%s): pi -p %s ... --tools %s%s",
                  cwd, " ".join(f"@{p.name}" for p in attachments),
                  tools, f" --model {model}" if model else "")

    with open(logfile, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd)

    if result.returncode != 0:
        raise RuntimeError(
            f"pi call failed (exit {result.returncode}). See {logfile} for details."
        )


def transcribe_with_gemini(audio: Path) -> str:
    """Transcribe via Gemini 3.5 Transcribe (cloud). Verbatim mode, no
    diarization — so full 1-hour files stay within the free-tier file-length
    limit (diarization is capped at 30 minutes)."""
    if genai is None:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY from env
    audio_file = client.files.upload(file=str(audio))
    # Must match the MIME type the Files API actually stored for this file
    # (audio_file.mime_type) — passing a different string, even a seemingly
    # equivalent one like "audio/mp3" vs "audio/mpeg", is rejected as a
    # mismatch against the uploaded file's parent MIME type.
    mime_type = audio_file.mime_type
    logger.debug("Gemini upload: uri=%s mime_type=%s", audio_file.uri, mime_type)

    # Uploaded files process asynchronously — referencing one before its
    # state reaches ACTIVE is a documented cause of generic 400s. Poll
    # until ready (or FAILED/timeout) before calling interactions.create.
    poll_start = time.monotonic()
    poll_timeout = 180
    while not audio_file.state or audio_file.state.name == "PROCESSING":
        if time.monotonic() - poll_start > poll_timeout:
            raise RuntimeError(
                f"Gemini file {audio_file.name} still PROCESSING after {poll_timeout}s"
            )
        logger.debug("  Gemini file %s state=%s, waiting...", audio_file.name, audio_file.state)
        time.sleep(3)
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file {audio_file.name} failed to process: state={audio_file.state}")

    try:
        interaction = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[{"type": "audio", "uri": audio_file.uri, "mime_type": mime_type}],
            generation_config={
                "transcription_config": {
                    "mode": {"type": "verbatim"},
                }
            },
        )
        text = (interaction.output_text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty transcript")
        return text
    finally:
        # Best-effort cleanup so uploaded audio doesn't accumulate in the project.
        try:
            client.files.delete(name=audio_file.name)
        except Exception:
            logger.debug("Could not delete uploaded Gemini file %s (non-fatal)", audio_file.name)


def transcribe_with_local_whisper(audio: Path, whisper_model: str) -> str:
    if mlx_whisper is None:
        raise RuntimeError("mlx-whisper not installed. Run: pip install mlx-whisper")
    result = mlx_whisper.transcribe(str(audio), path_or_hf_repo=whisper_model)
    return result["text"].strip()


def stage_transcribe(audio: Path, workdir: Path, whisper_model: str) -> None:
    logger.info("[1/6] Transcribing audio")
    text = None
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    if has_key:
        logger.info("  trying Gemini 3.5 Transcribe (cloud)")
        try:
            text = transcribe_with_gemini(audio)
            logger.info("  Gemini transcription succeeded")
        except Exception as e:
            logger.warning("  Gemini transcription failed: %s: %s — falling back to local Whisper",
                           type(e).__name__, e)
    else:
        logger.info("  no GEMINI_API_KEY/GOOGLE_API_KEY set — using local Whisper")

    if text is None:
        logger.info("  transcribing locally (mlx-whisper, model=%s)", whisper_model)
        text = transcribe_with_local_whisper(audio, whisper_model)

    out = workdir / "transcript.txt"
    out.write_text(text)
    logger.info("  wrote %s (%d words)", out, len(text.split()))


def stage_extract(workdir: Path, model_fast: str | None) -> None:
    logger.info("[2/6] Pass 1: extraction (transcript-only, no outside knowledge)")
    run_pi(
        attachments=[workdir / "transcript.txt"],
        prompt_file=PROMPTS_DIR / "pass1-extract.md",
        tools="read,write",
        model=model_fast,
        logfile=workdir / "pass1.log",
        cwd=workdir,
    )
    _require(workdir / "extract.md", "Pass 1")
    logger.info("  wrote %s", workdir / "extract.md")


def stage_enrich(workdir: Path, model_fast: str | None) -> None:
    logger.info("[3/6] Pass 2: multi-source enrichment (no API keys)")
    run_pi(
        attachments=[workdir / "extract.md"],
        prompt_file=PROMPTS_DIR / "pass2-enrich.md",
        tools="read,write,bash",
        model=model_fast,
        logfile=workdir / "pass2.log",
        cwd=workdir,
    )
    _require(workdir / "background.md", "Pass 2")
    logger.info("  wrote %s", workdir / "background.md")


def stage_synthesize(workdir: Path, model_strong: str | None) -> None:
    logger.info("[4/6] Pass 3: synthesis into full academic notes (no info dropped)")
    run_pi(
        attachments=[
            workdir / "transcript.txt",
            workdir / "extract.md",
            workdir / "background.md",
        ],
        prompt_file=PROMPTS_DIR / "pass3-synthesize.md",
        tools="read,write",
        model=model_strong,
        logfile=workdir / "pass3.log",
        cwd=workdir,
    )
    _require(workdir / "notes.md", "Pass 3")
    logger.info("  wrote %s", workdir / "notes.md")


def stage_verify(workdir: Path, model_strong: str | None) -> None:
    logger.info("[5/6] Pass 4: verification (flags unsupported claims, deletes nothing)")
    run_pi(
        attachments=[
            workdir / "notes.md",
            workdir / "transcript.txt",
            workdir / "background.md",
        ],
        prompt_file=PROMPTS_DIR / "pass4-verify.md",
        tools="read,write,edit",
        model=model_strong,
        logfile=workdir / "pass4.log",
        cwd=workdir,
    )
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


def stage_pdf(workdir: Path, run_name: str) -> None:
    logger.info("[6/6] Rendering PDFs (WeasyPrint)")

    notes_md = (workdir / "notes.md").read_text()
    _render_pdf(_markdown_to_html(notes_md), workdir / "notes.pdf")

    transcript_text = (workdir / "transcript.txt").read_text()
    _render_pdf(_text_to_html(transcript_text, f"Trascrizione — {run_name}"), workdir / "transcript.pdf")

    notes_md_dest = OUTPUT_DIR / f"{run_name}-notes.md"
    notes_pdf_dest = OUTPUT_DIR / f"{run_name}-notes.pdf"
    transcript_pdf_dest = OUTPUT_DIR / f"{run_name}-transcript.pdf"
    shutil.copy2(workdir / "notes.md", notes_md_dest)
    shutil.copy2(workdir / "notes.pdf", notes_pdf_dest)
    shutil.copy2(workdir / "transcript.pdf", transcript_pdf_dest)
    logger.info("  copied 3 output files to %s", OUTPUT_DIR)
    logger.info("    %s", notes_md_dest)
    logger.info("    %s", notes_pdf_dest)
    logger.info("    %s", transcript_pdf_dest)


def _require(path: Path, stage_name: str) -> None:
    if not path.exists() or not path.read_text().strip():
        raise RuntimeError(
            f"{stage_name} did not produce {path.name} — check the matching "
            f"pass log in {path.parent} before continuing."
        )


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
    if start <= STAGES.index("extract"):
        stage_extract(workdir, args.model_fast)
    if start <= STAGES.index("enrich"):
        stage_enrich(workdir, args.model_fast)
    if start <= STAGES.index("synthesize"):
        stage_synthesize(workdir, args.model_strong)
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
    logger.info("  3 output files in: %s", OUTPUT_DIR)
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
    if mlx_whisper is not None and shutil.which("ffmpeg") is None:
        problems.append(
            "`ffmpeg` not found on PATH — mlx-whisper needs it to decode audio "
            "(used as the local fallback if Gemini fails/is unavailable). Run: brew install ffmpeg"
        )

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) and mlx_whisper is None:
        problems.append(
            "No transcription backend available: no GEMINI_API_KEY/GOOGLE_API_KEY "
            "set, and mlx-whisper isn't installed. Set one of the env vars, or "
            "run: pip install mlx-whisper"
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
                         help="Model for extract/enrich passes (env: PI_MODEL_FAST)")
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