LANGUAGE: This is an Italian-language conference. Write `notes.md` entirely
in Italian, including all section headings (use the exact Italian headings
given below). Do not translate anything into English.

You are writing final academic-style notes from three attached files:
the speech extracted contents, `extract.md` (strict extraction from the talk), and
`background.md` (external context, already fetched — do not look anything
up yourself in this step).

CRITICAL RULE — MERGE AND ENRICH, DO NOT SUMMARIZE: your job is to combine
and connect the material, not to condense it. This is not a "summary" —
every sentence present in the transcript file should be reported in
notes.md, every topic, named entity, claim, and caveat present in
extract.md must appear in notes.md, and every entry in background.md must
appear in notes.md. You are reorganizing and writing connective prose
around the full content, never compressing it down or leaving things out
because they seem minor or redundant. If two items overlap, include both
rather than merging away detail. If this means the notes run long, that is
correct and expected — length is not a target to minimize, and a shorter
document is not a better one here.
Before finishing, do a completeness pass: check every numbered item in 
extract.md's "Speaker's claims & arguments" and every entry in 
background.md's topic list against notes.md, confirming each one is 
actually present. If something is missing, add it rather than noting 
the gap.

Write `notes.md` with this structure:

# <Titolo>
Generate this title yourself from the actual content of the talk — do not
use a generic placeholder like "Note del convegno" o "Riassunto della
sessione". It should be specific enough that someone scanning a list of
titles could tell this talk apart from any other: name the core topic,
argument, or system the speaker focused on. If the transcript explicitly
states an official session title, use that instead of generating one.
**Data:** {{DATA}}

The {{DATA}} placeholder above is filled in for you before this prompt
reaches you — use it exactly as given, verbatim, and do not try to
re-derive, override, or second-guess it from the transcript. Speakers
rarely state the date out loud, so this value comes from the recording's
filename (or its file metadata) instead, which is far more reliable.

## Sintesi
50–150 words orienting the reader. This is the only section allowed to be
short — everything below it must be complete, not a further summary.

## Conferenza
Every sentence from transcript file rewritten and 
Every extract.md's numbered list sentence, enriched with extra info 
adding them all from background.md as flowing academic prose, 
nothing left out but remove sources (Wikipedia / Wiktionary / arXiv / DuckDuckGo)
and the related URLs. Organize each sentence using sections, this is very 
important to do.
If the speaker's sentence is long, rambling, repetitive, or contains
  false starts and filler, still keep it: you can compress a little,
  tidy up, but do not cut any information.

## Domande aperte e note critiche
Every caveat from extract.md, plus anything where the speaker's claim and 
background.md context appear to be in tension (flag it, don't resolve it).

## Riferimenti
Numbered list of every source URL used in background.md.

Hard rules:
- Never state a fact as true unless it's either directly from the
  extract.md (speaker said it) or cited to a source in background.md.
- Do not blend the two without attribution.
- Do not introduce any fact, statistic, date, or claim that isn't present
  in the three attached files.
- Do not drop, merge-away, or silently shorten any item from extract.md or
  background.md.