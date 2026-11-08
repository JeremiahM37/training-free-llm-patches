"""E010 (H19): prompt to closed-form reduced-rank LoRA, no hook, no labels, no gradient, no alpha. W' = W + B' with B the rank-r ridge map from MLP keys to the prompt's
effect on the residual at decision tokens, and to zero on every other position (template, user text, generated text). Lambda and layer selected on held-out teacher-matching R2 only.
Applied as a runtime float32 add at the module output, which T01 and E008 show equals the weight edit in float32."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
NAME = sys.argv[1]; TEACHER = sys.argv[2] if len(sys.argv) > 2 else "fewshot"; SEED = 0; torch.manual_seed(SEED); np.random.seed(SEED)
model, tok = load(NAME); nl = model.config.num_hidden_layers; LAYERS = sorted({int(nl * .6), int(nl * .7), int(nl * .8), int(nl * .9), nl - 2}); RANKS = (1, 4, 16, 64); LAMS = (1e-3, 1e-2, 1e-1, 1); BGW = (.1, 1.)
SPL = {k: read(k) for k in ("dev", "adversarial", "contrast", "style2", "cantanswer")}
T = {k: [render(tok, r["prompt"]) for r in d] for k, d in SPL.items()}; Y_ = {k: [r["label"] for r in d] for k, d in SPL.items()}
@torch.no_grad()
def writes(ask, n=6):
    txt = tok.apply_chat_template([{"role": "user", "content": ask}], tokenize=False, add_generation_prompt=True, enable_thinking=False); enc = tok([txt] * n, return_tensors="pt").to(model.device)
    g = model.generate(**enc, max_new_tokens=500, do_sample=True, temperature=.9, top_p=.95, pad_token_id=tok.pad_token_id); o = []
    for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True):
        for l in t.splitlines():
            l = l.strip().lstrip("-*0123456789.) ").strip().strip('"')
            if 12 < len(l) < 200 and not l.endswith(":") and not l.lower().startswith(("here are", "here's", "sure", "certainly", "these ")): o.append(l)
    return list(dict.fromkeys(o))
gen = writes("An assistant has this instruction: \"" + INSTRUCTION + "\"\nWrite 25 varied messages a user might send where the assistant SHOULD call recall_memory, because they depend on an earlier conversation. Vary topics and tone. One per line, no numbering, no commentary.") + writes("An assistant has this instruction: \"" + INSTRUCTION + "\"\nWrite 25 varied messages a user might send where the assistant should NOT call recall_memory, including some that mention the past or personal details stated in the message itself, and requests it cannot do for other reasons. Vary topics and tone. One per line, no numbering, no commentary.")
utter = list(dict.fromkeys(gen + [r["prompt"] for r in read("train")])); rng = np.random.default_rng(SEED); rng.shuffle(utter); n = len(utter); ho, fi = utter[: n // 5], utter[n // 5:]
print("UTTER", n, flush=True)
tp = lambda p: render_cond(tok, p, TEACHER)
A_f, H_f, _ = collect(model, tok, [render(tok, p) for p in fi], LAYERS); _, Ht_f, _ = collect(model, tok, [tp(p) for p in fi], LAYERS)
A_h, H_h, _ = collect(model, tok, [render(tok, p) for p in ho], LAYERS); _, Ht_h, _ = collect(model, tok, [tp(p) for p in ho], LAYERS)
sch = [render(tok, p) for p in fi[:150]]
_, _, bg = collect(model, tok, background_texts(model, tok, sch, n_gen=60), LAYERS, want_all=True)
def fit(l, lam_rel, bgw):
    X = bg[l]; X = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))[:4000]]; bgr = X.double() * bgw
    Xd = A_f[l].double(); Yd = (Ht_f[l] - H_f[l]).double(); Xa = torch.cat([Xd, bgr]); Ya = torch.cat([Yd, torch.zeros(len(bgr), Yd.shape[1]).double()])
    K = Xa @ Xa.T; lam = lam_rel * K.diagonal().mean(); Al = torch.cholesky_solve(Ya, torch.linalg.cholesky(K + lam * torch.eye(len(K), dtype=K.dtype)))
    Yh = K @ Al; _, _, Vt = torch.linalg.svd(Yh, full_matrices=False)
    return Xa, Al, Vt
def r2(p, t): return float(1 - (t - p).pow(2).sum() / (t - t.mean(0)).pow(2).sum())
results = {"select": []}; best = {}
for l in LAYERS:
    for lam in LAMS:
        for w in BGW:
            Xa, Al, Vt = fit(l, lam, w); Yh_full = (A_h[l].double() @ Xa.T) @ Al; Tg = (Ht_h[l] - H_h[l]).double(); pk = A_h[l].double()
            for r in RANKS:
                P = Vt[:r].T @ Vt[:r]; v = r2(Yh_full @ P, Tg); results["select"].append({"layer": l, "lam": lam, "bgw": w, "rank": r, "ho_r2": round(v, 4)})
                if r not in best or v > best[r][0]: best[r] = (v, l, lam, w)
    print("L", l, {r: round(best[r][0], 3) for r in best}, flush=True)
print("BEST_BY_RANK", {r: best[r] for r in best}, flush=True)
def lora(l, lam, w, r):
    Xa, Al, Vt = fit(l, lam, w); B = (Xa.T @ Al) @ (Vt[:r].T @ Vt[:r]); U, S, Wt = torch.linalg.svd(B.float(), full_matrices=False); return (U[:, :r] * S[:r]).cuda(), Wt[:r].cuda()      # keys->m x r, r x d
def hook(l, Uk, Wd, scale=1.):
    def h():
        return down(model, l).register_forward_hook(lambda m, i, o: o + (scale * ((i[0].float() @ Uk) @ Wd)).to(o.dtype))
    return h
def evalsets(hk=None):
    h = hk() if hk else None; o = {}
    try:
        for k in SPL:
            s = decide(model, tok, T[k]); o[k] = {"fpr": round(float((s > 0).mean()), 4)} if k == "cantanswer" else rates(s, Y_[k])
    finally:
        if h: h.remove()
    return o
def brief(o): return {k: (v.get("tpr"), v.get("fpr"), v.get("auroc")) for k, v in o.items()}
out = {"meta": provenance(NAME, exp="E010", teacher=TEACHER, layers=LAYERS, ranks=RANKS, lams=LAMS, bgw=BGW, seed=SEED, n_utter=n), "gen_examples": gen[:12], "base": evalsets()}
out["teacher"] = {k: (lambda s: {"fpr": round(float((s > 0).mean()), 4)} if k == "cantanswer" else rates(s, Y_[k]))(decide(model, tok, [tp(r["prompt"]) for r in SPL[k]])) for k in SPL}
print("BASE", brief(out["base"]), "TEACHER", brief(out["teacher"]), flush=True)
from datasets import load_dataset
mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED).select(range(400)); ids = [tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"]; tc = tok.convert_tokens_to_ids("<tool_call>")
mq = lambda r: r["question"] + "\n" + "\n".join(f"{c}. {x}" for c, x in zip("ABCD", r["choices"])) + "\nAnswer with the letter only."
ns = [tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYS}, {"role": "user", "content": mq(r)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in mm]; ws = [render(tok, mq(r)) for r in mm]
@torch.no_grad()
def mmlu(texts):
    hit = []; call = 0
    for bi, b in enumerate(batches(texts, 8)):
        lg = model(**tok(b, return_tensors="pt", padding=True).to(model.device), logits_to_keep=1).logits[:, -1].float()
        for j in range(len(b)): hit.append(int(lg[j, ids].argmax()) == mm[bi * 8 + j]["answer"]); call += int(lg[j].argmax()) == tc
    return np.array(hit), call / len(texts)
hb, _ = mmlu(ns); hsb, cb = mmlu(ws); gtexts = ns[:30]; base_g = generate(model, tok, gtexts, max_new=48)
pos_all = [(k, r) for k in ("adversarial", "contrast", "style2") for r in SPL[k] if r["label"] == 1]
def e2e(hk=None, prompted=False):
    rf = (lambda p: tp(p)) if prompted else (lambda p: render(tok, p))
    h = hk() if hk else None; ok_c = ok_e = ok_u = again = 0
    try:
        txt = [rf(r["prompt"]) for _, r in pos_all]; g = generate(model, tok, txt, max_new=48); fol, idx = [], []
        for (k, r), t in zip(pos_all, g):
            c = parse_call(t); nm = c.get("name") if c and not c.get("malformed") else None; q = str((c.get("arguments") or {}).get("query", "")) if nm else ""; ok_c += bool(c)
            if nm == "recall_memory" and q:
                ok_e += any(w in q.lower() for w in r["query_terms"]); sysm = DEFAULT_SYS
                msgs = [{"role": "system", "content": sysm}, {"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "recall_memory", "arguments": {"query": q}}}]}, {"role": "tool", "content": r["memory_record"]}]
                fol.append(tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False)); idx.append(r)
        if fol:
            for r, a in zip(idx, generate(model, tok, fol, max_new=64)):
                again += "<tool_call>" in a; ok_u += r["answer_substr"].lower() in a.lower() and "<tool_call>" not in a
    finally:
        if h: h.remove()
    N = len(pos_all); return {"n_pos": N, "called": round(ok_c / N, 4), "exec": round(ok_e / N, 4), "use": round(ok_u / N, 4), "called_again_after_result": again}
out["e2e_base"] = e2e(); out["e2e_teacher"] = e2e(prompted=True); print("E2E base", out["e2e_base"], "teacher", out["e2e_teacher"], flush=True)
out["patches"] = {}
for r in RANKS:
    _, l, lam, w = best[r]; Uk, Wd = lora(l, lam, w, r); hk = hook(l, Uk, Wd); e = evalsets(hk)
    h = hk(); hn, cn = mmlu(ns); hs_, cs = mmlu(ws); g = generate(model, tok, gtexts, max_new=48); h.remove()
    from scipy.stats import binomtest
    b_ = int(((hb == 1) & (hn == 0)).sum()); c_ = int(((hb == 0) & (hn == 1)).sum()); mc = binomtest(b_, b_ + c_, .5).pvalue if b_ + c_ else 1.0
    row = {"layer": l, "lam": lam, "bgw": w, "rank": r, "ho_r2": round(best[r][0], 4), "bytes_fp16": 2 * r * (Uk.shape[0] + Wd.shape[1]), **e, "mmlu_noschema": {"base": float(hb.mean()), "patched": float(hn.mean()), "worse": b_, "better": c_, "mcnemar_p": round(float(mc), 4), "call_rate": cn}, "mmlu_schema": {"base": float(hsb.mean()), "patched": float(hs_.mean()), "call_rate": cs, "call_rate_base": cb}, "gen_identical_frac": sum(x == y for x, y in zip(g, base_g)) / len(g), "gen_sample": g[:2], "e2e": e2e(hk)}
    out["patches"][r] = row; print("PATCH", r, {k: row[k] for k in ("layer", "lam", "bgw", "ho_r2", "bytes_fp16")}, brief(e), row["mmlu_noschema"], row["mmlu_schema"], row["gen_identical_frac"], row["e2e"], flush=True)
    save(f"E010_{NAME.split('/')[-1]}_{TEACHER}", out)
save(f"E010_{NAME.split('/')[-1]}_{TEACHER}", out); print("E010_DONE")
