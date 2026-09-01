# conf-notes — grounded academic notes from conference recordings

Pipeline: recording → transcription (Gemini 3.5 Transcribe, falling back to
local mlx-whisper) → 4-pass pi run → transcript PDF + notes markdown + notes
PDF, three files per recording. PDFs are rendered with WeasyPrint (pure
HTML/CSS → PDF) — no LaTeX distribution required.

## Why 4 passes instead of one prompt

A single "listen and write great notes" prompt is exactly where models blend
transcript content, looked-up facts, and their own training-data guesses into
one indistinguishable paragraph. Splitting it forces each pass to only touch
one kind of information:

1. **Extract** — reads only the transcript. No outside knowledge allowed.
2. **Enrich** — looks up extracted terms across Wikipedia, Wiktionary, arXiv,
   and DuckDuckGo (in that order, no API keys needed). Explicitly forbidden
   from using its own memory; must write "unverified" if nothing is found.
3. **Synthesize** — combines passes 1–2 into the final notes. Every sentence
   must be traceable to either the transcript or a `background.md` citation.
   Nothing from extract.md or background.md may be dropped for brevity.
4. **Verify** — re-reads the finished notes against the transcript and
   background file with no tools/web access, and flags (never deletes)
   anything that isn't actually supported.

## Setup

```bash
pip install mlx-whisper google-genai weasyprint markdown
brew install cairo pango gdk-pixbuf libffi ffmpeg
```

No LaTeX/MacTeX needed — PDFs are rendered directly from HTML/CSS via
WeasyPrint, a much lighter install than a full TeX distribution.

**If WeasyPrint fails to import** with an error about missing/unreachable
libraries: this is almost always Homebrew's `pango`/`cairo`/`gdk-pixbuf`
not being on the dynamic linker's search path, not a missing `pip install`.
Fix (Apple Silicon):

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix)/lib:$DYLD_FALLBACK_LIBRARY_PATH"
```

Add that line to `~/.zshrc` so it's set for every future terminal session,
not just the current one. Full details:
https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#troubleshooting

`pi` should already be configured with your provider/fallback setup.

`input/`, `output/`, and `logs/` are created automatically next to
`conf_notes.py` on first run — no path editing required. If you'd rather
point them somewhere else (iCloud Drive, an external disk), edit the three
constants near the top of the file.

To enable cloud transcription (Gemini 3.5 Transcribe), export an API key —
otherwise every recording transcribes locally via mlx-whisper:

```bash
export GEMINI_API_KEY="your-key-here"
```

Free tier is enough to run this. Two things worth knowing:
- Free-tier requests may be used by Google to improve their products —
  fine for public talks, worth reconsidering for anything under NDA.
- This pipeline requests plain transcription (no speaker diarization), so
  full 1-hour recordings stay within limits — diarization on Gemini's
  free tier is capped at 30 minutes of audio.

If Gemini fails for any reason (no key, network issue, rate limit, file
too long) the pipeline automatically falls back to local mlx-whisper for
that file and continues — no manual intervention needed.

## Usage — batch mode (default)

Drop recordings into `INPUT_DIR`, then just run:

```bash
python conf_notes.py
```

It scans `INPUT_DIR` for audio files, processes them **one at a time** (no
parallelism — keeps RAM predictable on 16GB), and after each file succeeds
moves the source recording into `INPUT_DIR/processed/` so reruns don't
redo it. Three files land in `OUTPUT_DIR` per recording:
`<filename>-transcript.pdf`, `<filename>-notes.md`, `<filename>-notes.pdf`.
One log file per run is written to `LOG_DIR`, named
`yyyy-MM-dd_hh-mm_<filename>.log`.

If one file fails, the batch logs the failure and continues to the next
file rather than aborting the whole run.

## Usage — single file / debugging

```bash
python conf_notes.py --file ~/Recordings/keynote.m4a
```

Optionally pin models per stage (otherwise pi's default fallback chain is
used for every pass):

```bash
python conf_notes.py --model-fast claude-haiku-4-5 --model-strong claude-sonnet-5
# or via env vars:
PI_MODEL_FAST=claude-haiku-4-5 PI_MODEL_STRONG=claude-sonnet-5 python conf_notes.py
```

Resume from a specific stage while tuning a prompt (requires `--file`; reuses
existing intermediate files instead of rerunning transcription/extraction —
the script checks those files actually exist first and errors clearly if not):

```bash
python conf_notes.py --file talk.m4a --from-stage synthesize
```

Valid stages: `transcribe`, `extract`, `enrich`, `synthesize`, `verify`, `pdf`.

Add `--keep-input` to skip moving the source file into `processed/`. Note:
files passed via `--file` from outside `INPUT_DIR` are never auto-moved,
regardless of this flag — that only applies to batch-mode scanning.

## Files

Intermediate/debug files for each run live under `work/<filename>/` next to
the script:
- `transcript.txt` — raw transcription
- `extract.md` — pass 1 output
- `background.md` — pass 2 output (multi-source citations)
- `notes.md` / `notes.pdf` — final verified notes (also copied to OUTPUT_DIR)
- `transcript.pdf` — rendered transcript (also copied to OUTPUT_DIR)
- `pass1.log` … `pass4.log` — full pi transcripts of each pass, kept for audit

## Notes / tuning

- The pass 3 prompt generates a specific, content-based title for each
  talk's notes (not a generic "Session Notes") unless the transcript
  states an official title. If titles come out too generic, that's the
  prompt to tighten first.
- Thin "Background & context" sections for very recent/niche talks (new
  papers, unreleased products) are expected — none of the four sources
  will have coverage yet, and the alternative is the model guessing.
- If pass 4 is flagging a lot as unverified, check `pass4.log` — it's
  usually the fastest way to see where pass 3's attribution was sloppy.
- Each recording gets its own `work/<filename>/` and log file, so
  multi-track days are handled automatically — no manual separation needed.