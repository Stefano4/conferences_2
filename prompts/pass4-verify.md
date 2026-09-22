LANGUAGE: `notes.md` is written in Italian. Keep all your edits and
annotations in Italian too.

You are fact-checking `notes.md` against `extract.md` (ground truth for
what the speaker said) and `background.md` (ground truth for external
facts). You do NOT have web access in this step, and you must not use it —
only compare the three attached files against each other.

CRITICAL RULE — DO NOT DELETE CONTENT: your job is to label reliability,
not to shorten the notes. Never remove a sentence just because it's
unsupported — flag it instead. All information stays in the document.

For every factual sentence in notes.md, check:
- If it's presented as the speaker's view: does it match something
  actually in extract.md?
- If it's presented as background/context: is it backed by an entry in
  background.md?
- If neither applies: it is UNSUPPORTED.

ADDITIONAL CHECK — TRUNCATION: separately scan notes.md for any "..." or
any place a sentence looks cut short / paraphrased down from something
longer. For each one you find:
- Do not silently fix it yourself with outside knowledge.
- Look up the corresponding moment in extract.md and, if you can find
  the full original passage, replace the truncated text with the complete
  wording (this is filling a real gap from the ground-truth source, not
  adding new content, so it's allowed here).

Rewrite `notes.md` in place:
- Leave supported sentences exactly as they are.
- For unsupported sentences, keep the sentence but append inline:
  "**[non verificato]**". Do not delete or shorten it.

Do not add any new facts while doing this pass beyond recovering
truncated passages from extract.md as described above, and do not
remove any existing content — only annotate.