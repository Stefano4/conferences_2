LANGUAGE: This is an Italian-language conference. Search Italian sources
first (it.wikipedia.org / it.wiktionary.org, as below), and write
`background.md` in Italian — paraphrase every finding in Italian even when
the source itself (e.g. an arXiv abstract) is in English.

You have a `bash` tool with network access. Your ONLY job is to gather
background information for the topics/entities listed in `extract.md`. You
must NOT rely on your own training knowledge for facts — every fact you
write down must come from a response you actually fetched in this session.
No API keys are available or needed — every source below is free/open.

Technical notes for the curl calls below:
- URL-encode each term before inserting it into a query string (e.g.
  replace spaces with `%20` or `+`) — an unencoded multi-word term will
  break the request.
- Send a descriptive User-Agent header on every request (e.g.
  `-A "conf-notes-pipeline/1.0"`) — Wikipedia and DuckDuckGo may throttle
  or reject requests with no/default user agents.

For EACH item under "Topics & concepts mentioned" and "Named entities" in
extract.md, work through this fallback chain and stop at the first source
that returns real content:

1. Italian Wikipedia full-text search (more forgiving than title-prefix
   matching):
   curl -s "https://it.wikipedia.org/w/api.php?action=query&list=search&srsearch=<TERM>&format=json&srlimit=1"
   Take the top hit's title, then fetch the summary:
   curl -s "https://it.wikipedia.org/api/rest_v1/page/summary/<TITLE>"
   If the term has no Italian Wikipedia coverage but clearly would on
   English Wikipedia (e.g. a very new or niche technical term), you may
   retry against en.wikipedia.org instead — but still paraphrase the
   result into Italian.

2. If no Wikipedia match, try Wiktionary (good for terminology/jargon),
   Italian first:
   curl -s "https://it.wiktionary.org/api/rest_v1/page/summary/<TERM>"

3. If the term looks like a paper, method, or research topic, try arXiv
   (abstracts will typically be in English — paraphrase into Italian):
   curl -s "http://export.arxiv.org/api/query?search_query=all:<TERM>&max_results=1"
   Extract the <title>, <summary>, and <id> (URL) from the returned XML.

4. If still nothing, try DuckDuckGo's no-key HTML search as a general
   fallback, biased toward Italian results, and read the first 1-2 result
   snippets:
   curl -s "https://html.duckduckgo.com/html/?q=<TERM>&kl=it-it"
   Extract the top result's title, snippet text, and linked URL from the
   HTML. Only use this if you can identify an actual result with real
   content — do not fabricate a summary from a thin/empty response.

5. Only if all four sources above return nothing usable, write "Nessuna
   informazione trovata in nessuna fonte — non verificato" for that item.
   Do not fill the gap with your own knowledge under any circumstance.

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
