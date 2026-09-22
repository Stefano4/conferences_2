LANGUAGE: This is an Italian-language conference. Search the web using your 
tools and look for Italian sources first ( it.wikipedia.org / it.wiktionary.org ), 
and write `{{OUTPUT_FILE}}` in Italian — paraphrase every finding in Italian
even when the source itself (e.g., an arXiv abstract) is in English.

Your ONLY job is to gather a definition and some information for the topics/entities listed below, similarly to wikipedia.
You must NOT rely only on your own training knowledge for facts: every fact you
write down must come from a response you actually fetched in this session.
No API keys are available or needed — every source below is free/open.

For every successful lookup:
- Paraphrase the finding in 2-4 sentences, in Italian,  —
  do not copy source text verbatim.
- Record which source it came from (Wikipedia / Wiktionary / arXiv /
  DuckDuckGo) and the URL.

For every failed lookup, where you could not get any info at all from the web:
- If all applicable sources above return nothing usable, try to fill the gap with 
  your own knowledge and list as a source "Non verificato". 

Write the result to `{{OUTPUT_FILE}}` as a list, one entry per topic — cover
EVERY item in the list below, do not skip any:

{{ITEMS}}

Do not add commentary, opinions, or connections between topics — that
happens in a later step.