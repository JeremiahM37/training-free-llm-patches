"""E007 (H08, H07): instruction to patch with no human labels and no gradient. Selection uses only the self-generated data, never dev.
A: human labels (train.jsonl), layer/alpha picked on self-generated holdout. B: same utterances, labels = the prompted model's own decisions.
C: the model writes its own utterances from the instruction; labels = prompted decisions. D: reduced-rank map from unprompted to prompted activations at the decision token."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
NAME = sys.argv[1]; SEED = 0; torch.manual_seed(SEED)
model, tok = load(NAME); nl = model.config.num_hidden_layers; LAYERS = sorted({int(nl * .7), int(nl * .8), int(nl * .9), nl - 2}); ALPHAS = [.25, .5, 1, 2]
FEW = "fewshot"
SPL = {k: read(k) for k in ("dev", "adversarial", "contrast", "style2", "cantanswer")}
T = {k: [render(tok, r["prompt"]) for r in d] for k, d in SPL.items()}; Y = {k: [r["label"] for r in d] for k, d in SPL.items()}
def evalsets():
    o = {}
    for k in SPL:
        s = decide(model, tok, T[k]); o[k] = {"fpr": round(float((s > 0).mean()), 4)} if k == "cantanswer" else rates(s, Y[k])
    return o
def brief(o): return {k: (v.get("tpr"), v.get("fpr"), v.get("auroc")) for k, v in o.items()}
@torch.no_grad()
def writes(ask, n=6):
    msg = [{"role": "user", "content": ask}]; txt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok([txt] * n, return_tensors="pt").to(model.device)
    g = model.generate(**enc, max_new_tokens=500, do_sample=True, temperature=.9, top_p=.95, pad_token_id=tok.pad_token_id)
    out = []
    for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True):
        for l in t.splitlines():
            l = l.strip().lstrip("-*0123456789.) ").strip().strip('"')
            if 12 < len(l) < 200 and not l.endswith(":") and not l.lower().startswith(("here are", "here's", "sure", "certainly", "these ")): out.append(l)
    return list(dict.fromkeys(out))
gp = writes("An assistant has this instruction: \"" + INSTRUCTION + "\"\nWrite 25 varied messages a user might send where the assistant SHOULD call recall_memory, because they depend on an earlier conversation. Vary topics and tone. One per line, no numbering, no commentary.")
gn = writes("An assistant has this instruction: \"" + INSTRUCTION + "\"\nWrite 25 varied messages a user might send where the assistant should NOT call recall_memory, including some that mention the past or personal details stated in the message itself. Vary topics and tone. One per line, no numbering, no commentary.")
print("GEN", len(gp), len(gn), gp[:4], gn[:4], flush=True)
gen = gp + gn; dec = decide(model, tok, [render_cond(tok, p, FEW) for p in gen]); intent = np.array([1] * len(gp) + [0] * len(gn))
agree = float(((dec > 0).astype(int) == intent).mean()); print("AGREE", agree, flush=True)
# Primary arm C: every generated utterance, labelled by the prompted model's decision (no agreement filter). C_filtered drops disagreements, as a comparison.
tlab = (dec > 0).astype(int); allx = np.arange(len(gen)); rng = np.random.default_rng(SEED); rng.shuffle(allx); ho = allx[: len(allx) // 4]; fi = allx[len(allx) // 4:]
fi_f = np.array([i for i in fi if tlab[i] == intent[i]])
tr = read("train"); ytr = np.array([r["label"] for r in tr]); ptr = [r["prompt"] for r in tr]
ptr_dec = (decide(model, tok, [render_cond(tok, p, FEW) for p in ptr]) > 0).astype(int)
sch_g = [render(tok, p) for p in gen]; bg_src = background_texts(model, tok, [render(tok, p) for p in ptr])
_, _, bg = collect(model, tok, bg_src, LAYERS, want_all=True)
ins = lambda ps: [render_cond(tok, p, FEW) for p in ps]
Ag, Hg, _ = collect(model, tok, sch_g, LAYERS); _, Hig, _ = collect(model, tok, ins(gen), LAYERS)
At, Ht, _ = collect(model, tok, [render(tok, p) for p in ptr], LAYERS); _, Hit, _ = collect(model, tok, ins(ptr), LAYERS)
hoT, hoY = [sch_g[i] for i in ho], tlab[ho]
def make(l, a_, h_, hi_, y):
    y = torch.tensor(y); v = fit_gate(bg[l], a_[l][y == 1], a_[l][y == 0], "lda"); d = (hi_[l][y == 1] - h_[l][y == 1]).mean(0)
    return d / d.norm() * h_[l].norm(dim=1).mean(), v
out = {"meta": provenance(NAME, exp="E007", layers=LAYERS, alphas=ALPHAS, seed=SEED, instruction=INSTRUCTION), "n_gen_pos": len(gp), "n_gen_neg": len(gn), "gen_prompted_agreement": agree, "n_fit": len(fi), "n_fit_filtered": len(fi_f), "n_holdout": len(ho), "holdout_teacher_pos": int(hoY.sum()), "gen_examples": {"pos": gp[:15], "neg": gn[:15]}}
out["fewshot_prompt"] = {}
for k in SPL:
    s = decide(model, tok, ins([r["prompt"] for r in SPL[k]])); out["fewshot_prompt"][k] = {"fpr": round(float((s > 0).mean()), 4)} if k == "cantanswer" else rates(s, Y[k])
out["base"] = evalsets(); print("PROMPT", brief(out["fewshot_prompt"]), "BASE", brief(out["base"]), flush=True)
hs = {"A_human_labels": (At, Ht, Hit, ytr), "B_pseudo_labels": (At, Ht, Hit, ptr_dec)}
for nm, ids, lb in (("C_self_generated", fi, tlab[fi]), ("C_filtered", fi_f, tlab[fi_f])):
    g_ = torch.tensor(ids); hs[nm] = ({l: Ag[l][g_] for l in LAYERS}, {l: Hg[l][g_] for l in LAYERS}, {l: Hig[l][g_] for l in LAYERS}, lb)
for name, (a_, h_, hi_, y) in hs.items():
    if len(set(y.tolist())) < 2: out[name] = {"failed": "single class labels"}; continue
    best, bj = None, -9
    for l in LAYERS:
        D, v = make(l, a_, h_, hi_, y)
        for al in ALPHAS:
            with Edit(model, l, D, v, al):
                s = decide(model, tok, hoT); j = float((s[hoY == 1] > 0).mean() - (s[hoY == 0] > 0).mean())
            if j > bj: bj, best = j, (l, al)
    l, al = best; D, v = make(l, a_, h_, hi_, y)
    with Edit(model, l, D, v, al): r = evalsets()
    out[name] = {"layer": l, "alpha": al, "holdout_J": round(bj, 4), **r}; print(name, l, al, round(bj, 3), brief(r), flush=True)
# D: reduced-rank regression at the decision token, fit on self-generated + train utterances, no labels. Applied as a runtime hook at the decision token, NOT a weight edit: this tests how much of the prompt effect a low-rank map explains.
Dres = {}
l = int(nl * .8)
for l in sorted({int(nl * .8), nl - 2}):
    X = torch.cat([Hg[l], Ht[l]]).double(); Yy = torch.cat([Hig[l], Hit[l]]).double() - X; n = len(X); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)); te, trn = idx[: n // 4], idx[n // 4:]
    mx, my = X[trn].mean(0), Yy[trn].mean(0); Xc, Yc = X[trn] - mx, Yy[trn] - my; lam = 1e-1 * Xc.pow(2).sum(1).mean()
    K = Xc @ Xc.T; Ai = torch.linalg.solve(K + lam * torch.eye(len(trn), dtype=K.dtype), Yc)    # dual ridge
    Yhat = K @ Ai; _, _, Vt = torch.linalg.svd(Yhat, full_matrices=False)
    r2 = lambda p, t: float(1 - (t - p).pow(2).sum() / (t - t.mean(0)).pow(2).sum())
    row = {"const_shift_r2_on_test": r2(my.expand_as(Yy[te]), Yy[te])}
    for rank in (1, 4, 16):
        P = Vt[:rank].T @ Vt[:rank]; B = Xc.T @ Ai @ P; pred = (X[te] - mx) @ B + my; row[f"rank{rank}_r2"] = r2(pred, Yy[te])
        U, S, Wt = torch.linalg.svd(B.float(), full_matrices=False)
        def hook(m, i, o, B=B.float().cuda(), mx=mx.float().cuda(), my=my.float().cuda()):
            x = o[0] if isinstance(o, tuple) else o; x = x.clone(); dx = (x[:, -1].float() - mx) @ B + my; x[:, -1] += dx.to(x.dtype)
            return (x, *o[1:]) if isinstance(o, tuple) else x
        h = model.model.layers[l].register_forward_hook(hook); e = evalsets(); h.remove(); row[f"rank{rank}"] = e; print("RRR", l, rank, round(row[f"rank{rank}_r2"], 3), brief(e), flush=True)
    Dres[f"L{l}"] = row; print("RRR_R2", l, {k: v for k, v in row.items() if "r2" in k}, flush=True)
out["D_reduced_rank"] = Dres
save(f"E007_{NAME.split('/')[-1]}", out); print("E007_DONE")
