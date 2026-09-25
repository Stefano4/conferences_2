LANGUAGE: This is an Italian-language conference. Write `notes.md`
entirely in Italian, including all section headings (use the exact Italian
headings given below). Do not translate anything into English.

You are writing final academic-style notes from two attached files:
`extract.md` (strict extraction from the talk — its "Speaker's claims &
arguments" section is already a compressed-but-complete, sentence-by-
sentence rewrite of everything the speaker said, so treat that section as
your primary source for the speech content itself) and `background.md`
(external context, already fetched — do not look anything up yourself in
this step).

CRITICAL RULE — DO NOT SUMMARIZE OR FILTER: your job is to combine
and connect the material summing up information, not to condense it.
Every sentence in each extract.md's "Speaker's claims & arguments" numbered
item should be reported in notes-draft.md. 

Do not merge two distinct sentences into one to save space, even if they
are closely related so no individual statement is absorbed into another.
Also every topic, named entity, claim, 
and caveat present in extract.md must appear in notes-draft.md, and every
entry in background.md should appear in notes-draft.md. You are reorganizing
and writing connective prose around the full content, never compressing it
down or leaving things out because they seem minor or redundant. If two
items overlap, include both rather than merging away detail. 
If this means the notes run long, that is correct and expected:
length is not a target to minimize, and a shorter document is not 
a better one here.

- Never write "..." anywhere in notes-draft.md, for any reason. "..." is
  always a sign that content is being dropped, and content must never be
  dropped in this pass.
- Length check: extract.md's "Speaker's claims & arguments" section is
  already several thousand words on its own. Once you enrich it with
  background.md and write it out as full prose, "## Conferenza" below
  should come out to a comparable length or longer — never shorter. If
  your draft of "## Conferenza" is noticeably shorter than extract.md's
  "Speaker's claims & arguments" section, that is a sign content was
  dropped during the rewrite, not a sign you wrote a tighter version.
- Before finishing, go back through extract.md's "Speaker's claims &
  arguments" numbered items one by one, and separately through every entry
  in background.md, and confirm each has a clear counterpart somewhere in
  notes-draft.md. If you find one that doesn't, add it before moving on —
  do not treat a missing item as acceptable because the surrounding prose
  already "covers the gist" of it.
- Use Bold, italics, and underlining text as a skilled academic PhD student to make the notes more readable

Write `notes.md` with this structure:

# <Titolo>
Generate this title yourself from the actual content of the talk — do not
use a generic placeholder like "Note del convegno" o "Riassunto della
sessione". It should be specific enough that someone scanning a list of
titles could tell this talk apart from any other: name the core topic,
argument, or system the speaker focused on. If extract.md's "Session
metadata" section states an official session title, use that instead of
generating one.
**Data:** {{DATA}}

The {{DATA}} placeholder above is filled in for you before this prompt
reaches you — use it exactly as given, verbatim, and do not try to
re-derive, override, or second-guess it from extract.md. Speakers
rarely state the date out loud, so this value comes from the recording's
filename (or its file metadata) instead, which is far more reliable.

## Sintesi
50–150 words orienting the reader. This is the only section allowed to be
short — everything below it must be complete, not a further summary.

## Conferenza
Every numbered item and sentence in extract.md's "Speaker's claims & arguments", 
enriched (as additional information, never filter out things ) 
with extra info added from background.md, as flowing
academic prose (not filtering out any information), 
nothing left out but remove sources (Wikipedia / Wiktionary
/ arXiv / DuckDuckGo) and the related URLs. 
Write this as continuous prose, in the same order as extract.md's numbered
items, add titled sections and subsections.
This text must be the extract.md "Speaker's claims & arguments" section
rewritten, enriched, cleaned up, but not filtered down.
Use more or less the same amount of words and information as in the extract.md "Speaker's claims & arguments" section.
If the speaker's sentence is long, rambling, repetitive, or contains
false starts and filler, still keep it: you can tidy up, 
but do not cut any information.
ONLY for the very first and last sentence, if the speaker is clearly talking
about something totally out of topic, before the real conference starts or
after it has clearly eneded, you can drop that sentence.
- Use Bold, italics, and underlining text as a skilled academic PhD student to make the notes more readable
- Do not leave out any of the extract.md's "Speaker's claims & arguments" facts, information or anecdotes.


Hard rule:
- Do not drop, merge-away, or silently shorten any item from extract.md or
  background.md.