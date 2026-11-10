"""E018: probe-gated router for the in_context behaviour (E017/E017b: presence is linearly readable, ~.8 AUROC, but no patch or
prompt acts on it). A logistic probe on the last-token residual decides: fact absent -> force the call (decoding starts with
<tool_call>), fact present -> no call (the <tool_call> token is masked) and the model answers from the conversation.
Probe fit on history_fit pairs 1-20; layer and threshold chosen on pairs 21-40 (J). Held out: hard2 in_context (20 pairs).
Measured: decisions; positives end to end (used); negatives answered from context (the pair's answer_substr appears in the answer).
Baselines: no patch and the few-shot prompt, scored the same way. Per-item outcomes saved."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"; torch.manual_seed(0)
model, tok = load(NAME); NL = len(model.model.layers); LS = sorted(set(range(2, NL, 3)) | {NL - 1})
HF = read("history_fit"); FIT, SEL = [r for r in HF if r["pair"] <= 20], [r for r in HF if r["pair"] > 20]
HIC = [r for r in read("hard2") if r["kind"] == "in_context"]; ans = {r["pair"]: r["answer_substr"] for r in HIC if r["label"] == 1}
y = lambda rows: np.array([r["label"] for r in rows]); RN = lambda r: render(tok, r["prompt"], extra=r["history"])
RF = lambda r: render_cond(tok, r["prompt"], "fewshot", r["history"])
Hf, Hs, Ht = (collect(model, tok, [RN(r) for r in rows], LS)[1] for rows in (FIT, SEL, HIC))
best = None; out = {"meta": provenance(NAME, exp="E018"), "probe": {}}
for l in LS:
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(Hf[l].numpy(), y(FIT)); s = clf.decision_function(Hs[l].numpy())
    a = roc_auc_score(y(SEL), s); out["probe"][l] = round(a, 4)
    thr = max(np.unique(s), key=lambda t: (s[y(SEL) == 1] >= t).mean() - (s[y(SEL) == 0] >= t).mean())
    if best is None or a > best[1]: best = (l, a, clf, float(thr))
L, _, clf, THR = best; print("probe layer", L, "sel AUROC", round(best[1], 4), "thr", round(THR, 3), flush=True)
ps = clf.decision_function(Ht[L].numpy()); route = ps >= THR
out["router"] = {"layer": L, "threshold": THR, "hard2_in_context": {**rates(ps - THR, y(HIC)), "per_item": {r["id"]: round(float(x), 3) for r, x in zip(HIC, ps)}}}
tc = tok.convert_tokens_to_ids("<tool_call>")
def run(texts, rows, force):  # force[i]: True -> call forced, False -> call masked, None -> model decides
    gens = []
    for t, f in zip(texts, force):
        enc = tok([t + ("<tool_call>\n" if f else "")], return_tensors="pt").to(model.device)
        kw = {"bad_words_ids": [[tc]]} if f is False else {}
        g = model.generate(**enc, max_new_tokens=64, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id, **kw)
        gens.append(("<tool_call>\n" if f else "") + tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=False))
    res = {}
    for r, t in zip(rows, gens):
        c = parse_call(t); nm = c.get("name") if c and not c.get("malformed") else None; q = str((c.get("arguments") or {}).get("query", "")) if nm else ""
        if r["label"] == 1:
            ok = 0
            if nm == "recall_memory" and q:
                fol = followup(tok, r["prompt"], q, t, r["memory_record"], extra=r["history"]); a = generate(model, tok, [fol], max_new=64)[0]
                ok = int(r["answer_substr"].lower() in a.lower() and "<tool_call>" not in a)
            res[r["id"]] = {"called": int(c is not None), "used": ok}
        else:
            res[r["id"]] = {"called": int(c is not None), "answered_from_context": int(c is None and ans[r["pair"]].lower() in t.lower())}
    return res
ARMS = {"no patch": ([RN(r) for r in HIC], [None] * len(HIC)), "few-shot prompt": ([RF(r) for r in HIC], [None] * len(HIC)),
        "probe router": ([RN(r) for r in HIC], [bool(x) for x in route])}
out["arms"] = {}
for name, (T, F) in ARMS.items():
    res = run(T, HIC, F); P = [v for k, v in res.items() if k.endswith("-m")]; N = [v for k, v in res.items() if k.endswith("-n")]
    summ = {"tpr_called": round(np.mean([v["called"] for v in P]), 4), "fpr_called": round(np.mean([v["called"] for v in N]), 4),
            "used": round(np.mean([v["used"] for v in P]), 4), "used_ci": wilson(sum(v["used"] for v in P), len(P)),
            "neg_answered_from_context": round(np.mean([v["answered_from_context"] for v in N]), 4)}
    out["arms"][name] = {**summ, "per_item": res}; print("ARM", f"{name:16s}", summ, flush=True)
save("E018_router_" + NAME.split("/")[-1], out); print("E018_DONE")
