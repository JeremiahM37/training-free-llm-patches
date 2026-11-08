#!/usr/bin/env python3
"""Harvest paper metadata per research cluster. Resumable: finished (source, query) pairs are logged.

Honours each service's limits: on 429 it backs off, and after repeated refusals it stops using that
source for this run and says so. It never rotates identity. Rerun later to continue.
Output: literature/raw/records.jsonl (one record per source hit); dedup.py builds papers.jsonl.
"""
import json, re, sys, time, urllib.parse, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "literature"
RAW = ROOT / "raw"; RAW.mkdir(parents=True, exist_ok=True)
OUT = RAW / "records.jsonl"; DONE = RAW / "done.txt"; LOG = RAW / "harvest.log"
UA = {"User-Agent": "llm-modification-literature-harvest/0.1 (independent research)"}

CLUSTERS = {
 "model_editing": ["knowledge editing language model", "model editing locating and editing factual associations", "ROME MEMIT mass editing memory transformer", "locality generality specificity knowledge editing", "closed-form model editing", "unlearning language model weights editing", "sequential knowledge editing catastrophic failure"],
 "lora_adapters": ["LoRA low-rank adaptation large language model", "QLoRA quantized adapter", "adapter fine-tuning parameter efficient", "prefix tuning prompt tuning", "low-rank fine-tuning transformer memory efficient", "adapter merging composition LoRA", "LoRA hypernetwork generation"],
 "steering": ["activation steering language model", "activation addition steering vectors", "contrastive activation addition", "representation engineering", "inference-time intervention truthful", "steering vector reliability generalization", "conditional activation steering", "representation fine-tuning intervention", "refusal direction single direction", "optimal transport activation transport steering", "concept erasure linear closed form", "AxBench steering sparse autoencoder benchmark", "steering vector side effects", "persona vectors character", "function vectors task vectors in-context learning"],
 "mech_interp": ["mechanistic interpretability circuit discovery", "causal tracing activation patching", "causal abstraction distributed alignment search", "attribution patching", "transcoders circuit tracing", "attention head causal intervention language model", "linear representation hypothesis geometry", "logit lens tuned lens", "patchscopes"],
 "sae": ["sparse autoencoder language model features", "sparse dictionary learning feature disentanglement", "feature steering sparse autoencoder", "superposition toy models", "sparse autoencoder limitations negative result", "crosscoders model diffing"],
 "hypernetworks": ["hypernetwork generate parameters language model", "generate adapter weights from description", "instruction to steering vector hypernetwork", "neural network parameter generation diffusion", "task embedding generated model parameters", "weight space learning neural functionals"],
 "meta_learning": ["meta-learning fast adaptation transformer", "amortized inference task adaptation", "model-agnostic meta-learning large language models", "learning to learn without gradient adaptation", "conditional neural processes", "implicit gradients meta-learning"],
 "tta": ["test-time adaptation neural network", "inference-time learning language model", "online adaptation without labels", "batch normalization statistics adaptation", "feedback control language model generation"],
 "memory": ["associative memory neural network language model", "differentiable neural memory", "Hopfield network capacity modern", "memory tool retrieval agent language model decides", "when to retrieve adaptive retrieval language model", "agent memory tool call reliability", "tool calling activation steering", "kanerva sparse distributed memory", "vector symbolic architecture hyperdimensional composition"],
 "fast_weights": ["fast weights attend to the recent past", "Hebbian plasticity neural network learned", "differentiable plasticity", "biologically inspired synaptic plasticity transformer", "neuromodulation artificial neural network gating", "one-shot Hebbian update deep network"],
 "reservoir": ["reservoir computing readout closed form", "echo state network transformer", "dynamical systems view recurrent language model", "extreme learning machine ridge regression readout", "koopman operator neural network"],
 "linearized": ["neural tangent kernel finite width linearization", "linearized neural network fine-tuning", "last-layer Laplace kernel fine-tuning", "parameter space geometry loss landscape linear mode connectivity", "fisher information natural gradient geometry neural network"],
 "second_order": ["approximate second-order optimization K-FAC", "influence functions large language models", "Gauss-Newton one-step update neural network", "closed form update deep network layer", "Sherman-Morrison recursive least squares neural network"],
 "grad_free": ["gradient-free optimization language model", "forward-forward algorithm", "evolution strategies neural network weights", "forward gradient without backpropagation", "closed-form solution deep learning layer", "ridge regression activation fine-tune transformer"],
 "merging": ["task arithmetic model merging", "editing models with task arithmetic", "model merging language models", "weight averaging linear interpolation adapters", "vector arithmetic word analogy transformer behaviour"],
 "conditional_comp": ["conditional computation neural network gating", "dynamic neural networks routing", "feature-wise linear modulation FiLM", "mixture of experts routing gating", "gated modulation conditional layer normalization", "conditional intervention trigger language model"],
 "info_theory": ["information bottleneck deep learning", "minimum description length neural network", "compressed representation neural network theory", "PAC-Bayes compression bound", "lower bound model editing capacity", "information-theoretic limit adaptation", "compressed sensing neural activations"],
 "neuro": ["distributed representations neural circuits", "neural population manifold low dimensional control", "optogenetic steering neural population control theory", "neuromodulatory gain modulation cortex context-dependent computation", "mixed selectivity prefrontal context dependent", "orthogonal subspaces neural population multitasking"],
 "continual": ["continual learning catastrophic interference", "elastic weight consolidation", "orthogonal gradient descent subspace continual learning", "context-dependent gating continual learning", "superposition of many models into one", "continual learning language model adapter interference"],
 "compiler": ["program synthesis neural transformation", "compiling programs to neural networks", "Tracr compiled transformer", "natural language instruction compiled to neural modification", "system prompt compile activations", "prompt to weight translation"],
 "control": ["control theory language model activation closed-loop", "linear quadratic control neural activations", "feedback controller steering vector", "model predictive control activation steering"],
 "prompting_baseline": ["system prompt instruction following tool use", "prompt caching prefix cost inference", "instruction hierarchy system prompt", "prompt versus fine-tuning behaviour comparison", "emulating system prompt without tokens"],
 "low_rank_ops": ["low-rank linear map activations language model", "reduced rank regression neural representations", "affine concept editing", "linear relational embedding transformer", "linear relation decoding lre", "matrix steering operator activations", "projection erasure subspace transformer"],
}

ARXIV_CATS = "(cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:stat.ML OR cat:cs.NE OR cat:q-bio.NC OR cat:cs.IT)"
state = {"arxiv": 0.0, "openalex": 0.0, "hf": 0.0, "crossref": 0.0}
GAP = {"arxiv": 3.2, "openalex": 0.25, "hf": 1.0, "crossref": 1.0}
blocked = {}
failed = False   # set when a request in the current query was refused, so the query is retried next run

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f: f.write(line + "\n")

def get(source, url):
    global failed
    for attempt in range(5):
        wait = state[source] + GAP[source] - time.time()
        if wait > 0: time.sleep(wait)
        state[source] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(min(float(e.headers.get("Retry-After") or 0) or 20 * 2 ** attempt, 300))
                continue
            log(f"{source} HTTP {e.code} {url[:100]}"); failed = True; return None
        except Exception as e:
            log(f"{source} error {e}"); time.sleep(10)
    blocked[source] = True; failed = True
    log(f"{source}: repeated refusals, skipping this source for the rest of this run")
    return None

def emit(rec, cluster, query):
    rec.update(cluster=cluster, query=query)
    with OUT.open("a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def arxiv(cluster, q):
    terms = " AND ".join(f'all:"{w}"' if " " in w else f"all:{w}" for w in q.split()[:6])
    n = 0
    for start in (0, 200):
        u = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": f"({terms}) AND {ARXIV_CATS}", "start": start, "max_results": 200, "sortBy": "relevance"})
        b = get("arxiv", u)
        if not b: break
        ns = {"a": "http://www.w3.org/2005/Atom"}
        es = ET.fromstring(b).findall("a:entry", ns)
        for e in es:
            aid = re.sub(r"v\d+$", "", e.findtext("a:id", "", ns).rsplit("/abs/")[-1])
            emit({"source": "arxiv", "arxiv": aid, "title": " ".join(e.findtext("a:title", "", ns).split()),
                  "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
                  "year": int(e.findtext("a:published", "0", ns)[:4]), "date": e.findtext("a:published", "", ns)[:10],
                  "authors": [x.findtext("a:name", "", ns) for x in e.findall("a:author", ns)]}, cluster, q)
        n += len(es)
        if len(es) < 200: break
    return n

def oa_abstract(idx):
    if not idx: return ""
    pos = [(p, w) for w, ps in idx.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))

def openalex(cluster, q):
    n = 0; cursor = "*"
    for _ in range(2):
        u = "https://api.openalex.org/works?" + urllib.parse.urlencode({
            "filter": f"title_and_abstract.search:{q},from_publication_date:2012-01-01",
            "per-page": 200, "cursor": cursor,
            "select": "id,doi,title,publication_date,publication_year,authorships,primary_location,abstract_inverted_index,cited_by_count,referenced_works,ids,type"})
        b = get("openalex", u)
        if not b: break
        d = json.loads(b)
        for w in d.get("results", []):
            loc = (w.get("primary_location") or {}).get("source") or {}
            aid = None
            for v in [(w.get("primary_location") or {}).get("landing_page_url") or "", w.get("doi") or ""]:
                m = re.search(r"arxiv[./:](?:org/abs/)?(\d{4}\.\d{4,5})", v, re.I)
                if m: aid = m.group(1)
            emit({"source": "openalex", "openalex": w["id"], "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
                  "arxiv": aid, "title": w.get("title") or "", "abstract": oa_abstract(w.get("abstract_inverted_index")),
                  "year": w.get("publication_year"), "date": w.get("publication_date"), "venue": loc.get("display_name"),
                  "authors": [a["author"].get("display_name") for a in (w.get("authorships") or [])][:30],
                  "cited_by": w.get("cited_by_count"), "refs": w.get("referenced_works") or []}, cluster, q)
            n += 1
        cursor = (d.get("meta") or {}).get("next_cursor")
        if not cursor or len(d.get("results", [])) < 200: break
    return n

def hf(cluster, q):
    b = get("hf", "https://huggingface.co/api/papers/search?" + urllib.parse.urlencode({"q": q}))
    if not b: return 0
    n = 0
    for it in json.loads(b):
        p = it.get("paper") or it
        aid = p.get("id") or it.get("id")
        emit({"source": "hf", "arxiv": aid, "title": p.get("title") or it.get("title") or "",
              "abstract": p.get("summary") or "", "date": (p.get("publishedAt") or "")[:10],
              "year": int((p.get("publishedAt") or "0")[:4] or 0) or None,
              "authors": [a.get("name") for a in p.get("authors") or []]}, cluster, q)
        n += 1
    return n

def crossref(cluster, q):
    b = get("crossref", "https://api.crossref.org/works?" + urllib.parse.urlencode({
        "query.bibliographic": q, "rows": 100, "filter": "from-pub-date:2012,type:proceedings-article",
        "select": "DOI,title,author,issued,container-title,abstract,is-referenced-by-count"}))
    if not b: return 0
    n = 0
    for w in json.loads(b)["message"]["items"]:
        emit({"source": "crossref", "doi": w.get("DOI"), "title": (w.get("title") or [""])[0],
              "abstract": re.sub(r"<[^>]+>", "", w.get("abstract") or ""),
              "year": ((w.get("issued") or {}).get("date-parts") or [[None]])[0][0],
              "venue": (w.get("container-title") or [None])[0],
              "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in w.get("author") or []][:30],
              "cited_by": w.get("is-referenced-by-count")}, cluster, q)
        n += 1
    return n

SOURCES = {"arxiv": arxiv, "openalex": openalex, "hf": hf, "crossref": crossref}

def main():
    global failed
    wanted = sys.argv[1:] or list(SOURCES)
    done = set(DONE.read_text().splitlines()) if DONE.exists() else set()
    for s in wanted:
        for cluster, qs in CLUSTERS.items():
            for q in qs:
                key = f"{s}|{q}"
                if key in done or blocked.get(s): continue
                failed = False
                try: n = SOURCES[s](cluster, q)
                except Exception as e:
                    log(f"{s} failed on {q!r}: {e}"); continue
                if failed: continue
                log(f"{s} {cluster}: {q!r} -> {n}")
                with DONE.open("a") as f: f.write(key + "\n")
    log(f"HARVEST_DONE blocked={list(blocked)}")

if __name__ == "__main__":
    main()
