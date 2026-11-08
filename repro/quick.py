"""Tier 1: CPU-friendly reproduction of the core mechanisms on a small model (default Qwen2.5-0.5B-Instruct, about 1 GB).
No GPU, no gradient, no API keys. Prints a table and writes repro/out/quick_<model>.json.
It reproduces mechanisms and directions, not the exact headline numbers, which came from larger models (Tier 2).

  python repro/quick.py                  # about 10-20 min on an 8-core laptop
  python repro/quick.py --skip-clinc     # skip the 150-intent capacity curve (needs a small dataset download)
"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "research" / "experiments"))
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", 8)))
ap.add_argument("--skip-clinc", action="store_true"); args = ap.parse_args()
import torch; torch.set_num_threads(args.threads); torch.manual_seed(0)
from patchlib import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from scipy.sparse import hstack
T0 = time.time(); out = {"model": args.model, "threads": args.threads, "device": "cuda" if torch.cuda.is_available() else "cpu"}
import resource, ctypes, gc
try: _libc = ctypes.CDLL("libc.so.6")
except OSError: _libc = None
import threading
_mem = {"max": 0}
def _sample():
    while True:
        try: _mem["max"] = max(_mem["max"], int(open("/proc/self/statm").read().split()[1]) * 4096 // 2**20)
        except OSError: return
        time.sleep(.2)
threading.Thread(target=_sample, daemon=True).start()
def step(msg):
    msg = f"{msg}  (max RSS since last step {_mem['max']} MB)"; _mem["max"] = 0
    gc.collect()
    if _libc: _libc.malloc_trim(0)          # hand freed memory back to the OS (glibc); keeps peak RSS low on CPU
    print(f"[{time.time() - T0:6.0f}s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB] {msg}", flush=True)
model, tok = load(args.model); nl = model.config.num_hidden_layers; L = sorted({int(nl * .6), int(nl * .8), int(nl * .9)})
step(f"loaded {args.model}: {nl} layers, testing layers {L}")
tr = read("train"); ytr = np.array([r["label"] for r in tr]); ptr = [r["prompt"] for r in tr]
base = [render(tok, p) for p in ptr]; few = [render_cond(tok, p, "fewshot") for p in ptr]
_, _, bg = collect(model, tok, base, L, want_all=True, bs=8, max_rows=6000); step(f"background keys: {[tuple(bg[l].shape) for l in L]}")
A, H, _ = collect(model, tok, base, L, bs=8); _, Hf, _ = collect(model, tok, few, L, bs=8); step("fit activations collected")

# 1. The identity behind every weight patch: editing down_proj by alpha*D*v' equals adding (v'a)*alpha*D at the block output.
l = L[1]; v = fit_gate(bg[l], A[l][ytr == 1], A[l][ytr == 0]); d = (Hf[l][ytr == 1] - H[l][ytr == 1]).mean(0); D = d / d.norm() * H[l].norm(dim=1).mean()
enc = tok(base[:4], return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    lb = model(**enc).logits[:, -1].float()
    h = down(model, l).register_forward_hook(lambda m, i, o: o + .5 * (i[0].float() @ v.to(o.device))[..., None].to(o.dtype) * D.to(o.device).to(o.dtype))
    lh = model(**enc).logits[:, -1].float(); h.remove()
    with Edit(model, l, D, v, .5): le = model(**enc).logits[:, -1].float()
out["identity"] = {"max_abs_logit_diff_hook_vs_weight_edit": float((lh - le).abs().max()), "max_abs_patch_effect": float((lh - lb).abs().max()), "dtype": str(next(model.parameters()).dtype)}
step(f"identity: hook vs weight edit differ by {out['identity']['max_abs_logit_diff_hook_vs_weight_edit']:.2e} on an effect of {out['identity']['max_abs_patch_effect']:.2f}")

# 2. How much of a prompt's effect at the decision token is a constant shift, versus a low-rank function of the state? No labels used.
pool = [r["prompt"] for s in ("train", "dev", "style2") for r in read(s)]          # 280 utterances, labels unused
_, Hb, _ = collect(model, tok, [render(tok, p) for p in pool], L, bs=8); step("pool: unprompted done")
_, Ht, _ = collect(model, tok, [render_cond(tok, p, "fewshot") for p in pool], L, bs=8); step("pool: prompted done")
sd = {}
for l in L:
    X = Hb[l].double(); Y = Ht[l].double() - X; n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)); te, trn = idx[: n // 4], idx[n // 4:]
    mx, my = X[trn].mean(0), Y[trn].mean(0); Xc, Yc = X[trn] - mx, Y[trn] - my; K = Xc @ Xc.T; Al = torch.linalg.solve(K + .1 * K.diagonal().mean() * torch.eye(len(trn), dtype=K.dtype), Yc)
    _, _, Vt = torch.linalg.svd(K @ Al, full_matrices=False); r2 = lambda p, t: float(1 - (t - p).pow(2).sum() / (t - t.mean(0)).pow(2).sum())
    row = {"constant_shift": round(r2(my.expand_as(Y[te]), Y[te]), 3)}
    for k in (1, 4, 16): row[f"rank{k}"] = round(r2((X[te] - mx) @ (Xc.T @ Al @ Vt[:k].T @ Vt[:k]) + my, Y[te]), 3)
    sd[f"L{l}"] = row
out["state_dependence_heldout_R2"] = sd; step(f"state dependence (held-out R^2): {sd}")

# 3. Do activations carry signal beyond the words? Same folds for a TF-IDF classifier and the closed-form gate.
P = []
for s in ("train", "adversarial", "contrast", "cantanswer"):
    for i, r in enumerate(read(s)): P.append((r["prompt"], r["label"], f"{s}{r.get('pair', i)}"))
ps, y, g = [p for p, _, _ in P], np.array([q for _, q, _ in P]), np.array([z for _, _, z in P])
Ap, _, _ = collect(model, tok, [render(tok, p) for p in ps], L, bs=8); folds = list(GroupKFold(5).split(ps, y, g)); oof_t = np.zeros(len(y)); oof = {l: np.zeros(len(y)) for l in L}
for fi, ev in folds:
    wv, cv = TfidfVectorizer(ngram_range=(1, 2)), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)); F = [ps[i] for i in fi]; wv.fit(F); cv.fit(F); X = lambda q: hstack([wv.transform(q), cv.transform(q)])
    oof_t[ev] = LogisticRegression(C=100, max_iter=3000).fit(X(F), y[fi]).decision_function(X([ps[i] for i in ev]))
    for l in L:
        f_ = torch.tensor(fi); yf = torch.tensor(y[fi]); vv = fit_gate(bg[l], Ap[l][f_][yf == 1], Ap[l][f_][yf == 0]); oof[l][ev] = (Ap[l][torch.tensor(ev)] @ vv).numpy()
au = lambda s: round(auroc(s[y == 1], s[y == 0]), 3)
out["words_vs_activations_cv_auroc"] = {"tfidf": au(oof_t), **{f"gate_L{l}": au(oof[l]) for l in L}, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
step(f"words vs activations (5-fold AUROC): {out['words_vs_activations_cv_auroc']}")

# 4. Decision table: no patch, few-shot prompt, closed-form rank-one patch (layer/strength picked on dev), text classifier.
sets = {k: read(k) for k in ("dev", "style2", "contrast", "cantanswer")}; tab = {}
def run(texts_fn, ctx=None):
    o = {}
    for k, d in sets.items():
        if ctx: ctx.__enter__()
        try: s = decide(model, tok, [texts_fn(r["prompt"]) for r in d], bs=16)
        finally:
            if ctx: ctx.__exit__()
        o[k] = {"fpr": round(float((s > 0).mean()), 3)} if k == "cantanswer" else {x: rates(s, [r["label"] for r in d])[x] for x in ("tpr", "fpr", "auroc")}
    return o
tab["no_patch"] = run(lambda p: render(tok, p)); tab["fewshot_prompt"] = run(lambda p: render_cond(tok, p, "fewshot"))
dev = sets["dev"]; Td = [render(tok, r["prompt"]) for r in dev]; Yd = [r["label"] for r in dev]; best = None
for l in L:
    vv = fit_gate(bg[l], A[l][ytr == 1], A[l][ytr == 0]); dd = (Hf[l][ytr == 1] - H[l][ytr == 1]).mean(0); DD = dd / dd.norm() * H[l].norm(dim=1).mean()
    for a in (.25, .5, 1.):
        with Edit(model, l, DD, vv, a): r = rates(decide(model, tok, Td, bs=16), Yd)
        if best is None or r["tpr"] - r["fpr"] > best[0]: best = (r["tpr"] - r["fpr"], l, a, DD, vv)
_, l, a, DD, vv = best; tab[f"rank1_patch_L{l}_a{a}"] = run(lambda p: render(tok, p), Edit(model, l, DD, vv, a))
out["decision_table"] = tab; step("decision table done")

# 5. Capacity of K joint closed-form gates on 150 human-written CLINC intents: exact noise gain, no generation.
if not args.skip_clinc:
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus"); names = ds["train"].features["intent"].names; oos = names.index("oos"); ALL = [i for i in range(len(names)) if i != oos]
    txt = [(i, r["text"]) for i in ALL for r in [x for x in ds["train"] if x["intent"] == i][:6]]
    sysr = lambda t: tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYS}, {"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
    lc = L[1]; Ac, _, bgc = collect(model, tok, [sysr(t) for _, t in txt], [lc], want_all=True, bs=16, max_rows=3000); lab = np.array([i for i, _ in txt])
    Xb = bgc[lc].double(); Xb = Xb[torch.randperm(len(Xb), generator=torch.Generator().manual_seed(0))[:3000]]; lam = 1e-2 * Xb.pow(2).sum(1).mean() / Xb.shape[1]
    Lc = torch.linalg.cholesky(Xb @ Xb.T / len(Xb) + lam * torch.eye(len(Xb), dtype=Xb.dtype)); Rinv = lambda B: (B - Xb.T @ torch.cholesky_solve((Xb @ B) / len(Xb), Lc)) / lam
    a_ = Ac[lc].double(); mu = a_.mean(0); M = torch.stack([a_[lab == i].mean(0) - mu for i in ALL], 1); G = M.T @ Rinv(M); curve = []
    for K_ in (1, 10, 40, 80, 120, 140):
        gs = [float((torch.linalg.inv(G[ix][:, ix]).diag() * G[ix][:, ix].diag()).mean()) for ix in [np.random.default_rng(s + K_).choice(len(ALL), K_, replace=False) for s in range(5)]]
        curve.append({"K": K_, "noise_gain_vs_alone": round(float(np.mean(gs)), 3)})
    out["capacity_curve"] = {"layer": lc, "rows": curve, "note": "149 intents; K=150 is excluded because 150 centred means are linearly dependent"}; step(f"capacity curve: {curve}")
od = ROOT / "repro" / "out"; od.mkdir(exist_ok=True); f = od / f"quick_{args.model.split('/')[-1]}.json"; f.write_text(json.dumps(out, indent=1, default=float))
print("\n== decision table (TPR / FPR, AUROC) =="); 
for k, v in tab.items(): print(f"{k:28s}", "  ".join(f"{s}: {x.get('tpr','-')}/{x['fpr']}" + (f" ({x['auroc']})" if x.get('auroc') is not None else "") for s, x in v.items()))
print(f"\nwrote {f.relative_to(ROOT)} in {time.time() - T0:.0f}s")
