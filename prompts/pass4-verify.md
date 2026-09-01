LANGUAGE: `notes.md` is written in Italian. Keep all your edits and
annotations in Italian too.

You are fact-checking `notes.md` against `transcript.txt` (ground truth for
what the speaker said) and `background.md` (ground truth for external
facts). You do NOT have web access in this step, and you must not use it —
only compare the three attached files against each other.

CRITICAL RULE — DO NOT DELETE CONTENT: your job is to label reliability,
not to shorten the notes. Never remove a sentence just because it's
unsupported — flag it instead. All information stays in the document.

For every factual sentence in notes.md, check:
- If it's presented as the speaker's view: does it match something
  actually in transcript.txt?
- If it's presented as background/context: is it backed by an entry in
  background.md?
- If neither applies: it is UNSUPPORTED.

Rewrite `notes.md` in place:
- Leave supported sentences exactly as they are.
- For unsupported sentences, keep the sentence but append inline:
  "**[non verificato — non riscontrabile nella trascrizione o nelle fonti
  di supporto]**". Do not delete or shorten it.
- Add a final `## Note di verifica` section listing every sentence you
  flagged and a one-line reason, so the reader can see exactly what wasn't
  traceable to a source.

Do not add any new facts while doing this pass, and do not remove any
existing content — only annotate.
