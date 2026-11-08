#!/usr/bin/env python3
"""Rank papers.jsonl for reading. Score = relevance by embedding similarity to research questions (local nomic-embed-text),
plus per-cluster quotas so that a popular cluster cannot crowd out an adjacent field. Citation count is NOT used as a score
(research.md section 3). Writes literature/triage.jsonl and literature/reading_queue.jsonl. Embeddings are cached."""
import json, sys, urllib.request, concurrent.futures as cf
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1] / "literature"
import os; URL = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/embed"; MODEL = "nomic-embed-text"
QUESTIONS = [
 "closed-form derivation of a small modification to a pretrained neural network that changes its behaviour without gradient training",
 "conditional activation steering that only activates on relevant inputs, gating from internal states of the model",
 "rank-one edit to transformer MLP weights computed analytically from forward passes",
 "composing many independent interventions on a language model and predicting interference between them",
 "failure modes and unreliability of steering vectors, generalization to unseen prompts",
 "compiling a natural language instruction into model parameters or activation interventions",
 "making a language model call tools or retrieve memory more reliably through internal intervention",
 "fundamental limits of what can be changed in a neural network without training, new capabilities versus existing ones",
 "linear constrained estimators such as beamforming or minimum variance filters applied to neural activations",
 "gain modulation, context-dependent gating and subspace routing of computation in neural populations",
 "closed-form regression between activation distributions, optimal transport or low-rank linear maps on hidden states",
]
def embed(batch):
    req = urllib.request.Request(URL, json.dumps({"model": MODEL, "input": batch, "truncate": True}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)["embeddings"]
def embed_all(texts, bs=64, workers=3):
    parts = [texts[i:i+bs] for i in range(0, len(texts), bs)]
    with cf.ThreadPoolExecutor(workers) as ex: res = list(ex.map(embed, parts))
    return np.array([v for p in res for v in p], dtype=np.float32)
papers = [json.loads(l) for l in (ROOT / "papers.jsonl").open()]
papers = [p for p in papers if len(p["abstract"]) > 200]
cache = ROOT / "raw/emb.npz"; ids = [p["id"] for p in papers]; known = {}
if cache.exists():
    z = np.load(cache, allow_pickle=True); known = dict(zip(z["ids"].tolist(), z["vecs"]))
todo = [p for p in papers if p["id"] not in known]
if todo:
    vecs = embed_all([f"search_document: {p['title']}. {p['abstract'][:1500]}" for p in todo])
    known.update(zip([p["id"] for p in todo], vecs))
    np.savez(cache, ids=np.array(list(known), dtype=object), vecs=np.stack(list(known.values())))
E = np.stack([known[i] for i in ids]); E /= np.linalg.norm(E, axis=1, keepdims=True)
Q = embed_all([f"search_query: {q}" for q in QUESTIONS]); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
S = E @ Q.T; best = S.max(1); which = S.argmax(1)
for p, b, w in zip(papers, best, which): p["rel"] = round(float(b), 4); p["rel_q"] = int(w)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
order = sorted(range(len(papers)), key=lambda i: -best[i])
queue, seen, per = [], set(), {}
QUOTA = max(8, N // 40)
for i in order:                                  # quota pass: top of each cluster
    for c in papers[i]["clusters"]:
        if per.get(c, 0) < QUOTA and i not in seen:
            seen.add(i); per[c] = per.get(c, 0) + 1; queue.append(i); break
for i in order:                                  # remainder: global relevance
    if len(queue) >= N: break
    if i not in seen: seen.add(i); queue.append(i)
with (ROOT / "triage.jsonl").open("w") as f:
    for i in order:
        p = papers[i]; f.write(json.dumps({k: p.get(k) for k in ("id","title","year","rel","rel_q","clusters","arxiv","doi")}, ensure_ascii=False) + "\n")
with (ROOT / "reading_queue.jsonl").open("w") as f:
    for i in sorted(queue, key=lambda i: -best[i]):
        p = papers[i]; f.write(json.dumps({k: p.get(k) for k in ("id","title","year","rel","clusters","arxiv","doi","abstract")}, ensure_ascii=False) + "\n")
print(f"triaged {len(papers)}; queue {len(queue)}; with arXiv {sum(1 for i in queue if papers[i]['arxiv'])}; rel p50={np.median(best):.3f} p99={np.percentile(best,99):.3f}")
