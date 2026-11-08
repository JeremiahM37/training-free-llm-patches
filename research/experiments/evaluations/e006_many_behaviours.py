"""E006: K conditional behaviours at once, on human-written CLINC150. Behaviour i: if the request has intent i, begin the reply with code word W_i.
The word is arbitrary, so this tests routing to something the model would not do by itself. Compares:
  prompt_all: every rule in the system prompt (K rules)
  prompt_rag: a TF-IDF router picks one rule and puts only it in the prompt (the strong retrieval baseline)
  router_only: the same router treated as the decision itself, no model involved (upper baseline for decision)
  patch_indep / patch_joint: K-rank closed-form edit, each v_i fit alone, or jointly (T01 section 3)
  probe: linear classifier on the same keys with no edit, the accuracy ceiling for a linear gate
Fit uses n per intent; alpha picked on CLINC validation; final numbers on CLINC test. Patches apply as a runtime LoRA-style add in float32."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
KS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,4,16,64,150").split(",")]
NFIT = 10; NEV = 10; SEEDS = {1: 3, 4: 3, 16: 2, 64: 1, 150: 1}; ALPHAS = [0.5, 1, 2, 4, 8]
ds = load_dataset("clinc/clinc_oos", "plus"); names = ds["train"].features["intent"].names; OOS = names.index("oos")
model, tok = load(NAME); nl = model.config.num_hidden_layers; LS = [nl - 1, nl - 4]
# Code words: capitalized single tokens from the tokenizer's own vocabulary, excluding the base model's usual first tokens
import re as _re
_vocab = sorted(t for t in tok.get_vocab() if _re.fullmatch(r"[A-Z][a-z]{4,9}", t))
WORDS = np.random.default_rng(0).permutation(_vocab).tolist()
assert len(WORDS) >= 400, len(WORDS)
wid = lambda w: tok(w, add_special_tokens=False).input_ids[0]
def sys_render(text, rules=""):
    msgs = [{"role": "system", "content": DEFAULT_SYS + rules}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
def rule(i, w): return f"If the user's request is about {names[i].replace('_', ' ')}, begin your reply with the word {w}."
@torch.no_grad()
def first_tokens(texts, bs=16, hook=None):
    h = hook() if hook else None; out = []; t0 = time.time()
    try:
        for b in batches(texts, bs):
            enc = tok(b, return_tensors="pt", padding=True).to(model.device)
            out.append(model(**enc, logits_to_keep=1).logits[:, -1].argmax(-1).cpu())
    finally:
        if h: h.remove()
    return torch.cat(out).numpy(), (time.time() - t0) / max(len(texts), 1)
_seen = first_tokens([sys_render(r["text"]) for r in ds["train"].select(range(0, len(ds["train"]), 40))])[0]
_ban = {tok.decode([int(x)]) for x in set(_seen.tolist())}; WORDS = [w for w in WORDS if w not in _ban and len(tok(w, add_special_tokens=False).input_ids) == 1][:300]; assert len(WORDS) >= 150
by = lambda split: {i: [r["text"] for r in ds[split] if r["intent"] == i] for i in range(len(names))}
TR, VA, TE = by("train"), by("validation"), by("test")
rng0 = np.random.default_rng(0); ALL = [i for i in range(len(names)) if i != OOS]
# Keys: fit examples for every intent, background from generic text
fit_txt = {i: TR[i][:NFIT] for i in ALL}
flat = [(i, t) for i in ALL for t in fit_txt[i]]
A, H, _ = collect(model, tok, [sys_render(t) for _, t in flat], LS, bs=16)
lab = np.array([i for i, _ in flat])
bgt = [sys_render(t) for t in rng0.permutation([x for i in ALL for x in TR[i][NFIT:NFIT + 2]] + TR[OOS][:100]).tolist()]
_, _, bg = collect(model, tok, bgt, LS, want_all=True, bs=16)
pr = {}; BGH = {}; BGF = {}                 # capacity inputs; background split so gate energy is measured on rows the solver never saw
for l in LS:
    idx = torch.randperm(bg[l].shape[0], generator=torch.Generator().manual_seed(0)); BGF[l] = bg[l][idx[:5000]]; BGH[l] = bg[l][idx[5000:6500]]
    X = BGF[l].double(); Xc = X - X.mean(0)
    pf = lambda Z: (lambda ev: float(ev.sum() ** 2 / (ev ** 2).sum()))(torch.linalg.eigvalsh(Z @ Z.T / Z.shape[0]).clamp(min=0))
    pr[l] = {"uncentred": pf(X), "centred": pf(Xc)}
print("PARTICIPATION_RATIO", pr, flush=True)
def solver(l, lam_rel=1e-2):
    X = BGF[l].double()
    n = X.shape[0]; lam = lam_rel * X.pow(2).sum(1).mean() / X.shape[1]
    Lc = torch.linalg.cholesky(X @ X.T / n + lam * torch.eye(n, dtype=X.dtype))
    return lambda B: (B - X.T @ torch.cholesky_solve((X @ B) / n, Lc)) / lam
SOL = {l: solver(l) for l in LS}
def fit(l, chosen, mode):
    """indep/joint use class means centred on the global mean (LDA-like, signed, leaves an offset v'mu_all). jointnull adds the raw background mean as a zero-response constraint and uses raw class means, the exact T01 form."""
    a = A[l].double(); mu_all = a.mean(0)
    if mode == "jointnull":
        M = torch.stack([a[lab == i].mean(0) for i in chosen] + [mu_all], 1); RM = SOL[l](M); G = torch.eye(M.shape[1], dtype=M.dtype)[:, :-1]
        return (RM @ torch.linalg.solve(M.T @ RM, G)).float()
    M = torch.stack([a[lab == i].mean(0) - mu_all for i in chosen], 1); RM = SOL[l](M)      # m x K
    if mode == "joint": V = RM @ torch.linalg.solve(M.T @ RM + 1e-9 * torch.eye(M.shape[1], dtype=M.dtype), torch.eye(M.shape[1], dtype=M.dtype))
    else: V = RM / (M * RM).sum(0, keepdim=True)
    return V.float()
lm = model.lm_head.weight.detach().float().cpu()
def values(l, chosen, words):
    """lm: output-embedding direction of the word (no forward pass). Scaled so a unit gate moves typical h by its typical norm fraction."""
    U = torch.stack([lm[wid(w)] / lm[wid(w)].norm() for w in words], 1)
    return U * H[l].norm(dim=1).mean() * 0.1
def patch_hook(l, U, V, a):
    U_, V_ = (a * U).cuda(), V.cuda()
    def hook():
        return down(model, l).register_forward_hook(lambda m, i, o: o + ((i[0].float() @ V_) @ U_.T).to(o.dtype))
    return hook
def ev_sets(chosen, source):
    pos = [(i, t) for i in chosen for t in source[i][:NEV]]
    r = np.random.default_rng(7)
    others = [t for i in ALL if i not in chosen for t in source[i][:2]]
    others = r.permutation(others).tolist()[:100]; oos = r.permutation(source[OOS]).tolist()[:100]
    return pos, others + oos                 # same population at every K: up to 100 unchosen intents plus 100 out-of-scope
def score(preds_pos, pos, words_of, preds_neg, wordset):
    right = np.mean([p == wid(words_of[i]) for p, (i, _) in zip(preds_pos, pos)])
    wrong_word = np.mean([p in wordset and p != wid(words_of[i]) for p, (i, _) in zip(preds_pos, pos)])
    fp = float(np.mean([p in wordset for p in preds_neg])) if len(preds_neg) else 0.0
    return {"correct": round(float(right), 4), "wrong_word": round(float(wrong_word), 4), "fpr": round(fp, 4), "J": round(float(right) - fp, 4)}
# H03 test on real activations: exact noise gain from the closed form versus the iid formula, for many K, no model runs
curve = {}
for l in LS:
    a = A[l].double(); mu = a.mean(0); M_all = torch.stack([a[lab == i].mean(0) - mu for i in ALL], 1); RM_all = SOL[l](M_all); G_all = M_all.T @ RM_all
    rows = []
    for K_ in (1, 2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 150):
        gains = []
        for s_ in range(5):
            ix = np.random.default_rng(s_ + K_).choice(len(ALL), K_, replace=False); Gk = G_all[ix][:, ix]; gains.append(float((torch.linalg.inv(Gk).diag() * Gk.diag()).mean()))
        rows.append({"K": K_, "gain_rel_to_alone": round(float(np.mean(gains)), 4)})
    curve[l] = rows; print("GAIN_CURVE", l, rows, flush=True)
res = []
for K in KS:
    for seed in range(SEEDS.get(K, 1)):
        rng = np.random.default_rng(100 * K + seed); chosen = rng.choice(ALL, K, replace=False).tolist(); words = rng.permutation(WORDS)[:K].tolist()
        words_of = dict(zip(chosen, words)); wordset = {wid(w) for w in words}
        pos, neg = ev_sets(chosen, TE); vpos, vneg = ev_sets(chosen, VA)
        ptxt, ntxt = [t for _, t in pos], neg
        row = {"K": K, "seed": seed, "n_pos": len(pos), "n_neg": len(neg)}
        # Base
        pp, tb = first_tokens([sys_render(t) for t in ptxt]); pn, _ = first_tokens([sys_render(t) for t in ntxt], bs=16)
        row["base"] = {**score(pp, pos, words_of, pn, wordset), "s_per_prompt": round(tb, 4), "added_tokens": 0}
        # All rules in the prompt
        rules = " " + " ".join(rule(i, words_of[i]) for i in chosen)
        added = len(tok(rules).input_ids)
        pp, tp = first_tokens([sys_render(t, rules) for t in ptxt], bs=4); pn, _ = first_tokens([sys_render(t, rules) for t in ntxt], bs=4)
        row["prompt_all"] = {**score(pp, pos, words_of, pn, wordset), "s_per_prompt": round(tp, 4), "added_tokens": added}
        # Router: TF-IDF over fit examples with a rejection class trained on unchosen intents and OOS. It sees labelled negatives; the patch does not, which favours the router.
        fit_x = [t for i in chosen for t in fit_txt[i]]; fit_y = [i for i in chosen for _ in fit_txt[i]]
        negfit = [t for i in ALL if i not in chosen for t in fit_txt[i][:1]] + TR[OOS][:50]
        tv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(fit_x + negfit)
        clf = LogisticRegression(C=30, max_iter=2000).fit(tv.transform(fit_x + negfit), fit_y + [-1] * len(negfit))
        def route(ts):
            P = clf.predict_proba(tv.transform(ts)); j = P.argmax(1); return [int(clf.classes_[x]) for x in j]
        rp, rn = route(ptxt), route(ntxt)
        router_correct = float(np.mean([r == i for r, (i, _) in zip(rp, pos)])); router_fp = float(np.mean([r != -1 for r in rn]))
        row["router_only"] = {"correct": round(router_correct, 4), "fpr": round(router_fp, 4), "J": round(router_correct - router_fp, 4), "note": "decision only, separate classifier"}
        rr = lambda t, r: sys_render(t, (" " + rule(r, words_of[r])) if r != -1 else "")
        pp, tr_ = first_tokens([rr(t, r) for t, r in zip(ptxt, rp)]); pn, _ = first_tokens([rr(t, r) for t, r in zip(ntxt, rn)])
        row["prompt_rag"] = {**score(pp, pos, words_of, pn, wordset), "s_per_prompt": round(tr_, 4), "added_tokens": round(len(tok(rule(chosen[0], words[0])).input_ids), 1), "fit_bytes_note": "router vocabulary is separate"}
        # Patches
        for l in LS:
            U = values(l, chosen, words)
            for mode in ("indep", "joint", "jointnull"):
                V = fit(l, chosen, mode); name = f"patch_{mode}_L{l}"
                vt = lambda a: (lambda ps, ns: score(ps, vpos, words_of, ns, wordset))(first_tokens([sys_render(t) for _, t in vpos], hook=patch_hook(l, U, V, a))[0], first_tokens([sys_render(t) for t in vneg], hook=patch_hook(l, U, V, a))[0])
                best = max(ALPHAS, key=lambda a: vt(a)["J"])
                pp, tq = first_tokens([sys_render(t) for t in ptxt], hook=patch_hook(l, U, V, best)); pn, _ = first_tokens([sys_render(t) for t in ntxt], hook=patch_hook(l, U, V, best))
                G = (A[l][np.isin(lab, chosen)].cuda() @ V.cuda()).cpu().numpy()
                row[name] = {**score(pp, pos, words_of, pn, wordset), "alpha": best, "s_per_prompt": round(tq, 4), "added_tokens": 0, "bytes_fp16": 2 * K * (U.shape[0] + V.shape[0]), "gate_energy_mean_heldout_bg": round(float((BGH[l].cuda() @ V.cuda()).pow(2).mean()), 5),
                    "mean_offset_on_bg": round(float((BGH[l].cuda() @ V.cuda()).mean()), 5), "gate_on_fit_diag": round(float(np.mean([G[lab[np.isin(lab, chosen)] == c][:, j].mean() for j, c in enumerate(chosen)])), 4)}
            # Linear probe ceiling, same keys, no edit
            with torch.no_grad():
                Xf = torch.cat([A[l][np.isin(lab, chosen)]]); yf = lab[np.isin(lab, chosen)]
                Xn = A[l][~np.isin(lab, chosen)][:: max(1, (len(lab) - len(yf)) // 200)] if K < len(ALL) else A[l][:0]
                Xall = torch.cat([Xf, Xn]).numpy(); yall = np.concatenate([yf, -np.ones(len(Xn), dtype=int)])
                ap, _, _ = collect(model, tok, [sys_render(t) for t in ptxt], [l], bs=16); an, _, _ = collect(model, tok, [sys_render(t) for t in ntxt], [l], bs=16)
                std = Xall.std(0) + 1e-6
                if len(set(yall)) > 1:
                    lr = LogisticRegression(C=1.0, max_iter=300).fit(Xall / std, yall)
                    pc = float((lr.predict(ap[l].numpy() / std) == np.array([i for i, _ in pos])).mean()); nf = float((lr.predict(an[l].numpy() / std) != -1).mean())
                    row[f"probe_L{l}"] = {"correct": round(pc, 4), "fpr": round(nf, 4), "J": round(pc - nf, 4)}
        print("ROW", json.dumps(row), flush=True); res.append(row)
        save(f"E006_{NAME.split('/')[-1]}", {"meta": provenance(NAME, exp="E006", KS=KS, nfit=NFIT, nev=NEV, alphas=ALPHAS, layers=LS), "participation_ratio": pr, "gain_curve": curve, "rows": res})
print("E006_DONE")
