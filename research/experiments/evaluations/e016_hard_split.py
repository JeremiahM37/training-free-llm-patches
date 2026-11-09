"""E016: the frozen E011 recipes on a harder, freshly written held-out split (data/hard2.jsonl, never used for any selection):
80 matched pairs in new domains, four kinds: in_context (the fact is or is not already in this conversation's history),
multilingual (es/fr/de/pt), buried (one reference inside a 60-120 word message) and decoy (memory words without a memory need).
Same configurations, fit pool and layer map as E012; history turns are rendered before the final message for every method.
The TF-IDF control sees only the final message."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"; torch.manual_seed(0)
model, tok = load(NAME)
NL = len(model.model.layers); MAP = lambda l: min(NL - 1, round(l / 36 * NL)); print("layers", NL, {l: MAP(l) for l in (21, 25, 34)}, flush=True)
SPLIT = os.environ.get("SPLIT", "hard2"); test = read(SPLIT); HX = lambda r: r.get("history", ()); Yt = np.array([r["label"] for r in test])
tr = read("train"); ytr = np.array([r["label"] for r in tr]); ptr = [r["prompt"] for r in tr]
pool = list(dict.fromkeys(r["prompt"] for s in ("train", "dev", "adversarial", "contrast", "style2") for r in read(s)))
RULES = "Rules for recall_memory. Call it only if answering requires a personal fact from an EARLIER conversation with this user that is not in the current message (their preferences, past plans, things they told you). Never call it for general knowledge, math, code, writing, or when the message already contains the needed facts. If you call it, output only the tool call."
L21, L25, L34 = MAP(21), MAP(25), MAP(34); LAYERS = sorted({L21, L25, L34})
_, _, bg = collect(model, tok, background_texts(model, tok, [render(tok, p) for p in ptr]), LAYERS, want_all=True, max_rows=6000)
A, H, _ = collect(model, tok, [render(tok, p) for p in ptr], LAYERS); _, Hi, _ = collect(model, tok, [render_cond(tok, p, "instruction") for p in ptr], LAYERS)
Ap, Hp, _ = collect(model, tok, [render(tok, p) for p in pool], LAYERS); _, Htp, _ = collect(model, tok, [render_cond(tok, p, "fewshot") for p in pool], LAYERS)
_, _, bgp = collect(model, tok, background_texts(model, tok, [render(tok, p) for p in pool[:150]], n_gen=60), LAYERS, want_all=True, max_rows=4000)
class Add:
    def __init__(s, l, E): s.w, s.E = down(model, l).weight, E.to(model.device)
    def __enter__(s): s.old = s.w.data.clone(); s.w.data = (s.w.data.float() + s.E).to(s.old.dtype)
    def __exit__(s, *a): s.w.data = s.old
class Hook:
    def __init__(s, mod, fn): s.mod, s.fn = mod, fn
    def __enter__(s): s.h = s.mod.register_forward_hook(s.fn)
    def __exit__(s, *a): s.h.remove()
def nothing(): return Hook(down(model, 0), lambda m, i, o: o)
M = {}
M["no patch"] = (lambda r: render(tok, r["prompt"], extra=HX(r)), nothing, "-")
M["few-shot prompt"] = (lambda r: render_cond(tok, r["prompt"], "fewshot", HX(r)), nothing, "E001/E007 teacher")
M["rules prompt"] = (lambda r: render(tok, r["prompt"], DEFAULT_SYS + " " + RULES, HX(r)), nothing, "best of 4 on dev, E003")
# TFTV and Steer2Edit, frozen from E005 selection
def sdir(l): d = (Hi[l][ytr == 1] - H[l][ytr == 1]).mean(0); return d / d.norm()
W34 = down(model, L34).weight.detach().float().cpu(); _, S_, Vh = torch.linalg.svd(W34, full_matrices=False)
E_tftv = 0.05 * sdir(L34)[:, None] @ ((torch.sign(Vh @ A[34].mean(0)) * S_)[None] @ Vh)
W25 = down(model, L25).weight.detach().float().cpu(); k = W25.T @ sdir(L25); E_s2e = -32 * sdir(L25)[:, None] @ (k / k.norm())[None]
M["TFTV rank-one (L34, a=.05)"] = (lambda r: render(tok, r["prompt"], extra=HX(r)), lambda: Add(L34, E_tftv), "E005 dev selection")
M["Steer2Edit rank-one (L25, a=-32)"] = (lambda r: render(tok, r["prompt"], extra=HX(r)), lambda: Add(L25, E_s2e), "E005 dev selection")
# Gated rank-one (DKE), frozen from E002 Qwen3 selection: lda, L34, alpha 1, value dL
v = fit_gate(bg[L34], A[L34][ytr == 1], A[L34][ytr == 0]); dL = H[L34][ytr == 1].mean(0) - H[L34][ytr == 0].mean(0); D = dL / dL.norm() * H[L34].norm(dim=1).mean()
M["gated rank-one (L34, a=1)"] = (lambda r: render(tok, r["prompt"], extra=HX(r)), lambda: Edit(model, L34, D, v, 1.0), "E002 dev selection")
# Closed-form LoRA (E010 config rank 16, L21, lam .1, bg weight .1); fit on the unlabelled pool
def rr(l, lam_rel, bgw, r, Xk, Yt_):
    Xb = bgp[l].double() * bgw; Xa = torch.cat([Xk.double(), Xb]); Ya = torch.cat([Yt_.double(), torch.zeros(len(Xb), Yt_.shape[1]).double()])
    K = Xa @ Xa.T; Al = torch.cholesky_solve(Ya, torch.linalg.cholesky(K + lam_rel * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype)))
    _, _, Vt = torch.linalg.svd(K @ Al, full_matrices=False); return ((Xa.T @ Al) @ (Vt[:r].T @ Vt[:r])).float()
B = rr(L21, .1, .1, 16, Ap[L21], Htp[L21] - Hp[L21]).to(model.device)
M["closed-form LoRA, no labels (r16, L21)"] = (lambda r: render(tok, r["prompt"], extra=HX(r)), lambda: Hook(down(model, L21), lambda m, i, o: o + (i[0].float() @ B).to(o.dtype)), "E010 holdout-R2 selection")
# Reduced-rank map at the decision token (E007 config rank 4, L34), position-aware hook
X = Hp[L34].double(); Y = Htp[L34].double() - X; mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; Kx = Xc @ Xc.T
Al = torch.linalg.solve(Kx + .1 * Xc.pow(2).sum(1).mean() * torch.eye(len(Kx), dtype=Kx.dtype), Yc); _, _, Vt = torch.linalg.svd(Kx @ Al, full_matrices=False)
Br = (Xc.T @ Al @ Vt[:4].T @ Vt[:4]).float().to(model.device); mxg, myg = mx.float().to(model.device), my.float().to(model.device)
def last_only(m, i, o):
    x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[:, -1] += ((x[:, -1].float() - mxg) @ Br + myg).to(x.dtype); return (x, *o[1:]) if isinstance(o, tuple) else x
M["reduced-rank map at decision token, hook (r4, L34)"] = (lambda r: render(tok, r["prompt"], extra=HX(r)), lambda: Hook(model.model.layers[L34], last_only), "E007 config")
out = {"meta": provenance(NAME, exp="E016", split=SPLIT, note="frozen E011 recipes on the hard2 split; history rendered before the final message", layer_map={21: L21, 25: L25, 34: L34}, load_8bit=bool(os.environ.get("LOAD_8BIT")), keep_bf16_layers=os.environ.get("KEEP_BF16_LAYERS", "")), "methods": {}}
pos = [r for r in test if r["label"] == 1]
for name, (rf, ctx, src) in M.items():
    with ctx():
        s = decide(model, tok, [rf(r) for r in test]); res = rates(s, Yt); KIND = np.array([r["kind"] for r in test]); res["by_kind"] = {k: rates(s[KIND == k], Yt[KIND == k]) for k in sorted(set(KIND))}
        g = generate(model, tok, [rf(r) for r in pos], max_new=48); fol, idx = [], []; called = ex = use = 0
        for r, t in zip(pos, g):
            c = parse_call(t); nm = c.get("name") if c and not c.get("malformed") else None; q = str((c.get("arguments") or {}).get("query", "")) if nm else ""; called += bool(c)
            if nm == "recall_memory" and q:
                ex += any(w in q.lower() for w in r["query_terms"])
                fol.append(followup(tok, r["prompt"], q, t, r["memory_record"], extra=HX(r))); idx.append(r)
        for r, a in zip(idx, generate(model, tok, fol, max_new=64) if fol else []): use += r["answer_substr"].lower() in a.lower() and "<tool_call>" not in a
    n = len(pos); res.update(called=round(called / n, 4), executed=round(ex / n, 4), used=round(use / n, 4), use_ci=wilson(use, n), frozen_from=src)
    out["methods"][name] = res; print("E016", f"{name:52s}", res["tpr"], res["fpr"], res["auroc"], {k: (v["tpr"], v["fpr"]) for k, v in res["by_kind"].items()}, "e2e", res["called"], res["executed"], res["used"], flush=True)
wv, cv = TfidfVectorizer(ngram_range=(1, 2)), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)); wv.fit(ptr); cv.fit(ptr); F = lambda q: hstack([wv.transform(q), cv.transform(q)])
st = LogisticRegression(C=100, max_iter=3000).fit(F(ptr), ytr).decision_function(F([r["prompt"] for r in test]))
out["methods"]["TF-IDF classifier (decision only, separate model)"] = {**rates(st, Yt), "by_kind": {k: rates(st[KIND == k], Yt[KIND == k]) for k in sorted(set(KIND))}, "frozen_from": "E003 config, fit on train"}; print("TEST tfidf", out["methods"]["TF-IDF classifier (decision only, separate model)"]["tpr"], out["methods"]["TF-IDF classifier (decision only, separate model)"]["fpr"], flush=True)
save(f"E016_{SPLIT}_" + NAME.split("/")[-1], out); print("E016_DONE")
