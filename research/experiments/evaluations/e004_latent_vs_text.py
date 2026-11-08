"""E004: is there signal in activations beyond the words? Pool all hard data, 5-fold CV with pairs kept together, gate versus TF-IDF on identical folds.
Then leave-one-negative-kind-out: unseen kinds of negatives. Then behaviour with a hard-negative fit, real weight edit. Test split stays sealed."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from scipy.sparse import hstack
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
model, tok = load(NAME); nl = model.config.num_hidden_layers
LAYERS = sorted({int(nl * f) for f in (.25, .4, .5, .6, .7, .75, .8, .85, .9)} | {nl - 2, nl - 1})
pool = []
for s in ("train", "adversarial", "contrast", "cantanswer"):
    for i, r in enumerate(read(s)): pool.append({**r, "split": s, "group": f"{s}{r['pair']}" if "pair" in r else f"{s}{i}", "kind": r.get("kind") or "pos"})
y = np.array([r["label"] for r in pool]); ps = [r["prompt"] for r in pool]; groups = np.array([r["group"] for r in pool])
texts = [render(tok, p) for p in ps]; base = decide(model, tok, texts)
A, H, _ = collect(model, tok, texts, LAYERS); _, Hi, _ = collect(model, tok, [render_cond(tok, p, "instruction") for p in ps], LAYERS)
trn = [render_cond(tok, r["prompt"], "schema") for r in read("train")]
_, _, bg = collect(model, tok, background_texts(model, tok, trn), LAYERS, want_all=True)
def text_scores(fit, ev):
    wv, cv = TfidfVectorizer(ngram_range=(1, 2)), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    P = [ps[i] for i in fit]; wv.fit(P); cv.fit(P); F = lambda q: hstack([wv.transform(q), cv.transform(q)])
    return LogisticRegression(C=100, max_iter=3000).fit(F(P), y[fit]).decision_function(F([ps[i] for i in ev]))
folds = list(GroupKFold(5).split(ps, y, groups))
oof_t = np.zeros(len(y)); oof = {(l, k): np.zeros(len(y)) for l in LAYERS for k in ("lda", "lcmv")}
for fit, ev in folds:
    oof_t[ev] = text_scores(fit, ev)
    fi = torch.tensor(fit); yf = torch.tensor(y[fit])
    for l in LAYERS:
        for k in ("lda", "lcmv"):
            v = fit_gate(bg[l], A[l][fi][yf == 1], A[l][fi][yf == 0], k)
            s = A[l][torch.tensor(ev)] @ v
            sf = A[l][fi] @ v; oof[l, k][ev] = (s - (sf[yf == 1].mean() + sf[yf == 0].mean()) / 2).numpy()
hard = np.array([r["split"] != "train" for r in pool])
def au(s, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return round(auroc(s[m & (y == 1)], s[m & (y == 0)]), 4)
out = {"meta": provenance(NAME, exp="E004", layers=LAYERS, n=len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
out["cv"] = {"base_first_token": au(base), "tfidf": au(oof_t), "tfidf_hard_only": au(oof_t, hard), "layers": {f"L{l}/{k}": {"all": au(oof[l, k]), "hard": au(oof[l, k], hard)} for l in LAYERS for k in ("lda", "lcmv")}}
# bootstrap by group: is the best layer really above text?
best = max(LAYERS, key=lambda l: au(oof[l, "lda"])); ug = np.unique(groups); rng = np.random.default_rng(0); diffs = []
idx = {g: np.where(groups == g)[0] for g in ug}
for _ in range(2000):
    samp = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); yy = y[samp]
    if yy.min() == yy.max(): continue
    diffs.append(auroc(oof[best, "lda"][samp][yy == 1], oof[best, "lda"][samp][yy == 0]) - auroc(oof_t[samp][yy == 1], oof_t[samp][yy == 0]))
out["cv"]["best_layer"] = best; out["cv"]["auroc_diff_best_minus_tfidf"] = {"mean": round(float(np.mean(diffs)), 4), "ci95": [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])], "note": "best layer chosen on the same data, so biased upward"}
print("CV", json.dumps(out["cv"]), flush=True)
# unseen negative kinds
lk = {}
negk = sorted({r["kind"] for r in pool if r["label"] == 0})
for kk in negk:
    ev = np.array([i for i, r in enumerate(pool) if r["kind"] == kk]); fitm = np.array([r["kind"] != kk for r in pool])
    posidx = np.where(y == 1)[0]; fit = np.where(fitm)[0]
    # score the unseen negatives against out-of-fold positives by fitting on everything except this kind; positives scored by CV-free split
    ph = posidx[np.arange(len(posidx)) % 3 == 0]; fit2 = np.setdiff1d(fit, ph); evs = np.concatenate([ev, ph])
    t = text_scores(fit2, evs); ly = y[evs]; row = {"tfidf": round(auroc(t[ly == 1], t[ly == 0]), 4)}
    for l in (best, nl - 2):
        fi = torch.tensor(fit2); yf = torch.tensor(y[fit2]); v = fit_gate(bg[l], A[l][fi][yf == 1], A[l][fi][yf == 0], "lda")
        s = (A[l][torch.tensor(evs)] @ v).numpy(); row[f"L{l}"] = round(auroc(s[ly == 1], s[ly == 0]), 4)
    lk[kk] = row
out["leave_kind_out"] = lk; print("LKO", json.dumps(lk), flush=True)
# behaviour with hard-negative fit, real edit, out of fold
beh = {}
for l in sorted({best, nl - 2}):
    for a in (0.25, 0.5, 1.0):
        sc = np.zeros(len(y))
        for fit, ev in folds:
            fi = torch.tensor(fit); yf = torch.tensor(y[fit]); v = fit_gate(bg[l], A[l][fi][yf == 1], A[l][fi][yf == 0], "lda")
            dH = (Hi[l][fi][yf == 1] - H[l][fi][yf == 1]).mean(0); D = dH / dH.norm() * H[l].norm(dim=1).mean()
            with Edit(model, l, D, v, a): sc[ev] = decide(model, tok, [texts[i] for i in ev])
        r = rates(sc, y); r["by_split_fpr"] = {s: round(float((sc[[i for i, q in enumerate(pool) if q["split"] == s and q["label"] == 0]] > 0).mean()), 3) for s in ("train", "adversarial", "contrast", "cantanswer")}
        r["by_split_tpr"] = {s: round(float((sc[[i for i, q in enumerate(pool) if q["split"] == s and q["label"] == 1]] > 0).mean()), 3) for s in ("train", "adversarial", "contrast")}
        beh[f"L{l}/a{a}"] = r; print("BEH", l, a, json.dumps(r), flush=True)
out["behaviour_cv"] = beh; out["base_rates"] = rates(base, y)
save(f"E004_{NAME.split('/')[-1]}", out); print("E004_DONE")
