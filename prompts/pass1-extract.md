LANGUAGE: This is a conference recorded in Italian. Write `extract.md` in
Italian. Do not translate anything into English. Keep verbatim phrases in
the speaker's original Italian wording, exactly as said.

You are doing STRICT EXTRACTION ONLY from the attached transcript. Do not add
any information you were not given. Do not use outside knowledge, do not
guess, do not elaborate, do not summarize.

Read the attached transcript (a conference talk transcription, roughly one
hour / ~11,000 words). Produce a structured extraction and write it to
`extract.md` with these sections:

## Session metadata
Title, speaker name(s), venue/event, date — only if explicitly stated in the
transcript. Otherwise write "not stated".

## Classificazione
Classify the talk into EXACTLY ONE of these four categories, based on its
actual subject matter: Arte, Letteratura, Scienza, Altro. Write just the
single chosen label, nothing else. Use "Scienza" for technical, scientific,
engineering, medical, or research-focused talks; "Arte" for visual arts,
music, film, design, or performance; "Letteratura" for literature, writing,
publishing, or literary criticism; "Altro" for anything that doesn't
clearly fit the other three (business, politics, general culture, etc).

## Topics & concepts mentioned
Bullet list of every named concept, technology, method, or term the speaker
discusses that is not common knowledge (specific algorithms, product names,
frameworks, studies, historical events, etc). One line each, just the term as
spoken — no explanation.

## Named entities
People, organizations, papers, products mentioned by name.

## Speaker's claims & arguments
This section is a full rewrite of the talk, not a selection from it. Later
steps in this pipeline will work from this section as their only record of
what was said — they will not have the transcript itself — so it must carry
the same information as the transcript, simply rewritten: cleaned up,
reorganized into numbered paragraphs, but not filtered down.

Numbered list of paragraphs, grouped the way the speaker naturally grouped
their own points. Within each paragraph, rewrite the speaker's speech
sentence by sentence, in order, covering essentially every sentence in the
transcript — treat "I skipped a sentence because it seemed minor/off-topic/
already implied" as not a valid reason to skip it. The only sentences that
may be left out are exact or near-exact repeats of a sentence already
captured (the speaker restating the same point in the same words) and pure
disfluency with zero informational content (e.g. a lone "ehm", a false
start abandoned mid-word before the real sentence begins). Everything else
— including asides, digressions, jokes, anecdotes, audience interaction,
transitions like "detto questo" or "quindi", and material that seems
redundant but is phrased differently or adds any nuance — gets its own
place in the list.
Use the closest verbatim phrase from the transcript it's based on.

- Never write "..." anywhere in this section, for any reason. "..." is
  always a sign that content is being dropped, and content must never be
  dropped in this pass.
- If the speaker's sentence is long, rambling, repetitive, or contains
  false starts and filler, still keep it: you can compress a little,
  tidy up, but do not cut any information.
- Do not soften, generalize, summarize, or add nuance the speaker didn't
  state. This is a transcription-level extraction, not a summary — the
  test is "could someone reconstruct what was actually said from this
  list, sentence for sentence", not "did I capture the gist".
- Do not merge two distinct sentences into one to save space, even if they
  are closely related — give each its own entry (or its own clause within
  a paragraph) so no individual statement is absorbed into another.
- Before finishing this section, go back to the transcript and check it
  against your draft passage by passage (not just skimming for "...").
  For each stretch of transcript, confirm there is a corresponding item in
  your list. If you find a gap — a sentence, aside, or point with no
  counterpart in the draft — add it before moving on. A shorter list is
  not a sign you did a better job; a list that is missing something the
  speaker said is a failure of this pass.

If this makes the section very long, that is correct and expected — do not
shorten it to keep the file a manageable size. Length is not a concern in
this pass. Given the transcript is roughly one hour / ~11,000 words, this
section rewriting essentially all of it should itself run to several
thousand words — a short version means content is missing.

## Explicit open questions / caveats
Anything the speaker themselves flagged as uncertain, contested, or
unresolved. Same no-ellipsis rule applies: quote the full relevant passage,
not a fragment of it.

Rules:
- Every item must trace back to something actually said in the transcript.
- If you are unsure whether something was said, omit it rather than guess.
- Do not research, define, or explain anything here — that happens later.