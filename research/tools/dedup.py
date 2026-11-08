#!/usr/bin/env python3
"""Merge literature/raw/records.jsonl into literature/papers.jsonl, deduplicated by arXiv id, DOI, normalized title."""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "literature"

def norm(t): return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()

papers, by_key = [], {}
for line in (ROOT / "raw/records.jsonl").open():
    r = json.loads(line)
    if not r.get("title"): continue
    t = norm(r["title"])
    if len(t) < 12: continue
    keys = [k for k in (f"a:{r.get('arxiv')}" if r.get("arxiv") else None,
                        f"d:{r['doi'].lower()}" if r.get("doi") else None, f"t:{t}") if k]
    p = next((by_key[k] for k in keys if k in by_key), None)
    if p is None:
        p = {"id": None, "title": r["title"], "abstract": "", "authors": [], "year": None, "date": None, "venue": None,
             "arxiv": None, "doi": None, "openalex": None, "cited_by": None, "refs": [], "sources": [], "clusters": [], "queries": []}
        papers.append(p)
    for f in ("arxiv", "doi", "openalex", "year", "date", "venue", "cited_by"):
        if r.get(f) and not p[f]: p[f] = r[f]
    if len(r.get("abstract") or "") > len(p["abstract"]): p["abstract"] = r["abstract"]
    if len(r.get("authors") or []) > len(p["authors"]): p["authors"] = r["authors"]
    if len(r.get("refs") or []) > len(p["refs"]): p["refs"] = r["refs"]
    for f, v in (("sources", r["source"]), ("clusters", r["cluster"]), ("queries", r["query"])):
        if v not in p[f]: p[f].append(v)
    for k in keys: by_key[k] = p

for i, p in enumerate(papers):
    p["id"] = f"arxiv:{p['arxiv']}" if p["arxiv"] else f"doi:{p['doi']}" if p["doi"] else f"oa:{(p['openalex'] or '').rsplit('/',1)[-1]}" if p["openalex"] else f"p{i}"
with (ROOT / "papers.jsonl").open("w") as f:
    for p in papers: f.write(json.dumps(p, ensure_ascii=False) + "\n")
ab = sum(1 for p in papers if len(p["abstract"]) > 100)
print(f"{len(papers)} unique papers, {ab} with abstracts, {sum(1 for p in papers if p['arxiv'])} with arXiv ids")
