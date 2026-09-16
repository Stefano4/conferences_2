LANGUAGE: This is an Italian-language conference. Search the web.
Look for Italian sources first (it.wikipedia.org / it.wiktionary.org, as below), 
and write `background.md` in Italian — paraphrase every finding in Italian
even when the source itself (e.g. an arXiv abstract) is in English.

Your ONLY job is to gather background information for the topics/entities listed in `extract.md`. 
You must NOT rely on your own training knowledge for facts — every fact you
write down must come from a response you actually fetched in this session.
No API keys are available or needed — every source below is free/open.

Only if all applicable sources above return nothing usable, write
"Nessuna informazione trovata in nessuna fonte — non verificato" for
that item. Do not fill the gap with your own knowledge under any
circumstance.

For every successful lookup:
- Paraphrase the finding in 2-4 sentences, in Italian, in your own words —
  do not copy source text verbatim.
- Record which source it came from (Wikipedia / Wiktionary / arXiv /
  DuckDuckGo) and the URL.

Write the result to `background.md` as a list, one entry per topic — cover
EVERY item from extract.md's "Topics & concepts mentioned" and "Named
entities" lists, do not skip any:

### <Term>
<2-4 sentence paraphrased summary in Italian, or "Nessuna informazione trovata in nessuna fonte — non verificato">
Source: <Wikipedia|Wiktionary|arXiv|DuckDuckGo> — <URL, or "none">

Do not add commentary, opinions, or connections between topics — that
happens in a later step.