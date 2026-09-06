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
Numbered list of paragraphs. Rewrite, sentence by sentence, the speaker speech.
Do not skip any sentence, unless there are clear repetitions.
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
  list", not "did I capture the gist". 
- Before finishing this section, reread your own draft specifically to find 
  any "..." or any place where you can tell you left out some statement 
  rather than transcribe. If you find one, go back to the transcript and 
  expand it into full sentence(s) instead, just compressing repetitions.

If this makes the section very long, that is correct and expected — do not
shorten it to keep the file a manageable size. Length is not a concern in
this pass.

## Explicit open questions / caveats
Anything the speaker themselves flagged as uncertain, contested, or
unresolved. Same no-ellipsis rule applies: quote the full relevant passage,
not a fragment of it.

Rules:
- Every item must trace back to something actually said in the transcript.
- If you are unsure whether something was said, omit it rather than guess.
- Do not research, define, or explain anything here — that happens later.