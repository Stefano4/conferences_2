LANGUAGE: This is an Italian-language conference. Write `notes.md` entirely
in Italian, including all section headings (use the exact Italian headings
given below). Do not translate anything into English.

You are writing final academic-style notes from three attached files:
the raw transcript, `extract.md` (strict extraction from the talk), and
`background.md` (external context, already fetched — do not look anything
up yourself in this step).

CRITICAL RULE — MERGE AND ENRICH, DO NOT SUMMARIZE: your job is to combine
and connect the material, not to condense it. This is not a "summary" —
every sentence present in the transcript file should be reported in modes.md,
every topic, named entity, claim, and caveat present in extract.md must
appear in notes.md, and every entry in background.md must appear in
notes.md. You are reorganizing and writing connective prose around the
full content, never compressing it down or leaving things out because they
seem minor or redundant. If two items overlap, include both rather than
merging away detail. If this means the notes run long, that is correct and
expected — length is not a target to minimize, and a shorter document is
not a better one here.

Write `notes.md` with this structure:

# <Titolo>
Generate this title yourself from the actual content of the talk — do not
use a generic placeholder like "Note del convegno" or "Riassunto della
sessione". It should be specific enough that someone scanning a list of
titles could tell this talk apart from any other: name the core topic,
argument, or system the speaker focused on. If the transcript explicitly
states an official session title, use that instead of generating one.
**Relatore:** <nome> · **Sede:** <sede> · **Data:** <data>

## Sintesi
50–150 words orienting the reader. This is the only section allowed to be
short — everything below it must be complete, not a further summary.

## Conferenza
Every sentence from transcript file rewritten and enriched with extra info 
(or corrected when there are transcription issues).
Crossing each extract.md's numbered list, as flowing academic prose,
organized by sections and theme, but with nothing left out. 
Each claim stays attributed to the speaker (e.g. "Il relatore sostiene che..."), 
never stated as established fact. Do not add supporting evidence that 
isn't already in background.md.

## Contesto e approfondimenti
Full paragraphs drawing on every background.md entry not already folded
into "Concetti chiave" detail, clearly separated from "Argomentazioni del
relatore", each citing its source.

## Domande aperte e note critiche
Every caveat from extract.md, plus anything where the speaker's claim and
background.md context appear to be in tension (flag it, don't resolve it).

## Riferimenti
Numbered list of every source URL used in background.md.

Hard rules:
- Never state a fact as true unless it's either directly from the
  transcript (speaker said it) or cited to a source in background.md.
- Do not blend the two without attribution.
- Do not introduce any fact, statistic, date, or claim that isn't present
  in the three attached files.
- Do not drop, merge-away, or silently shorten any item from extract.md or
  background.md.
