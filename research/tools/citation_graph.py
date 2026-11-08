#!/usr/bin/env python3
"""Citation graph from OpenAlex reference lists in papers.jsonl. Writes citation_graph/edges.jsonl and report.md.
Backward: works most cited by the high-relevance set, inside and outside the corpus. Forward: corpus papers that cite the works we read.
Counts locate work to inspect; they are never treated as quality (research.md section 3)."""
import json, collections
from pathlib import Path
L = Path(__file__).resolve().parents[1] / "literature"; G = L / "citation_graph"; G.mkdir(exist_ok=True)
P = [json.loads(l) for l in (L / "papers.jsonl").open()]
rel = {}
if (L / "triage.jsonl").exists():
    for l in (L / "triage.jsonl").open():
        r = json.loads(l); rel[r["id"]] = r["rel"]
byoa = {p["openalex"]: p for p in P if p.get("openalex")}
edges = [(p["openalex"], r) for p in P if p.get("openalex") for r in p["refs"]]
with (G / "edges.jsonl").open("w") as f:
    for a, b in edges: f.write(json.dumps([a, b]) + "\n")
thr = sorted(rel.values())[int(len(rel) * .9)] if rel else 0
hi = [p for p in P if rel.get(p["id"], 0) >= thr and p["refs"]]
back = collections.Counter(r for p in hi for r in p["refs"])
indeg = collections.Counter(b for _, b in edges)
read = set()
for f in (L / "reading_notes").glob("*.jsonl"):
    for l in f.open():
        a = json.loads(l).get("arxiv")
        if a: read.add(a)
read_oa = {p["openalex"]: p for p in P if p.get("openalex") and p.get("arxiv") in read}
citers = collections.defaultdict(list)
for a, b in edges:
    if b in read_oa and a in byoa: citers[b].append(a)
unread_forward = collections.Counter()
for b, cs in citers.items():
    for c in cs:
        if byoa[c].get("arxiv") not in read: unread_forward[c] += 1
out = [f"# Citation graph report\n\nPapers with reference lists: {sum(1 for p in P if p['refs'])} of {len(P)}. Edges: {len(edges)}. Inside-corpus citations: {sum(1 for _, b in edges if b in byoa)}.\n"]
out.append("## Backward: most referenced by the top-decile relevance set (in corpus)\n")
for w, c in [x for x in back.most_common(400) if x[0] in byoa][:40]:
    p = byoa[w]; out.append(f"- {c} | {p['year']} | {'READ' if p.get('arxiv') in read else 'unread'} | {p.get('arxiv') or p.get('doi') or w} | {p['title'][:90]}")
out.append("\n## Backward: referenced but NOT in our corpus (OpenAlex ids, to resolve)\n")
missing = [x for x in back.most_common(400) if x[0] not in byoa][:40]
for w, c in missing: out.append(f"- {c} | {w}")
out.append("\n## Forward: unread corpus papers citing several papers we read\n")
for w, c in unread_forward.most_common(40):
    p = byoa[w]; out.append(f"- {c} | {p['year']} | rel {rel.get(p['id'])} | {p.get('arxiv') or p.get('doi') or w} | {p['title'][:90]}")
(G / "report.md").write_text("\n".join(out)); print("\n".join(out))
