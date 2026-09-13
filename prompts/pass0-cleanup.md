LANGUAGE: This is a conference recorded in Italian. Write `transcript.txt`
in Italian. Do not translate anything into English. Keep the speaker's
original Italian wording exactly as said, except for the specific fixes
described below.

You are doing LIGHT CLEANUP ONLY on the attached raw transcript
(`transcript-raw.txt`), which is the unedited output of an automatic
speech-recognition model (whisper) and contains the usual whisper
artifacts: duplicated lines, looping/hallucinated phrases, and
transcription typos. This is not an extraction or summarization pass —
you are not analyzing the content, just cleaning the text itself. Every
piece of actual information and every real sentence the speaker said must
survive this pass untouched.

Do exactly these three things, and nothing else:

1. Remove repeated and hallucinated text.
   Whisper sometimes emits the same sentence, phrase, or clause two or
   more times in a row (immediate repetition), or drifts into a looping
   phrase that is clearly a transcription artifact rather than something
   the speaker actually said (e.g. the same short phrase repeated many
   times with no plausible reason a speaker would do that). Collapse each
   such run down to a single occurrence. Only remove a repetition when it
   is clearly a machine artifact — if the speaker plausibly repeated
   something themselves for emphasis (e.g. said the same short phrase
   twice with different intonation implied by context, or it's a
   deliberate rhetorical repetition), leave it as-is; when in doubt, keep
   it rather than remove it.

2. Remove useless filler repetitions of greetings/thanks.
   Remove pleasantries such as "buongiorno", "grazie grazie", 
   "salve salve" and similar filler words (like "ecco", "cioè", "diciamo").

3. Correct obvious transcription typos.
   Fix clear, obvious errors that are plainly created by the transcription
   affecting grammar. Only fix something here if you
   are confident it's a transcription error and that makes the sentence
   grammatically incorrect.
   - Do NOT "correct" anything you are unsure about. Unknown or
     unfamiliar words, proper names, foreign terms, technical jargon,
     acronyms, or anything that could plausibly be a real word/name you
     simply don't recognize must be left exactly as transcribed — do not
     guess a "cleaner" spelling for it. Those get handled later.
   - Just correct transcription error so clear that would affect grammar.

Hard rules:
- Do not summarize, paraphrase, compress, or reword anything beyond the
  three fixes above. If you're not fixing a duplication/filler-repeat/typo
  in a given sentence, that sentence should come out character-for-character
  the same as it went in.
- Do not drop any sentence or any piece of information. The only things
  allowed to disappear are exact/near-exact duplicate runs (rule 1) and
  duplicate greetings/thanks (rule 2) — and even then only the extra
  copies, never the first occurrence.
- Do not add anything: no headers, no commentary, no notes about what you
  changed, no summary at the top or bottom. The output is plain transcript
  text only, in the same shape as the input (paragraphs/line breaks as in
  the original, minus the removed duplicates).
- If you are unsure whether something is a whisper artifact or something
  the speaker actually said, leave it in. Under-cleaning is fine;
  over-cleaning (removing real content) is not.

Write the cleaned result to `transcript.txt`, similar output number of words (besides hallucinated repetitions).
