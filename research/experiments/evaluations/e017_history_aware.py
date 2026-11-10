"""E017: can a patch learn "the needed fact is already in this conversation"? (E016: every method fails hard2's in_context kind.)
Synthetic history pairs from train/dev positives: the same final message, with an earlier turn that states either this item's
memory_record (fact present -> label 0) or another item's record (fact absent -> label 1). Two framings, rotated.
  fit:       train positives (40 pairs)        select: dev positives (60 pairs), by J = TPR - FPR at 0
  held out:  hard2 in_context (20 naturally written pairs, never used for selection); side effects on hard2 (all kinds) and test.
Part A: is presence linearly readable at the decision token (logistic probe per layer)? Model and prompt baselines.
Part B: labelled gated rank-one edits (gate fires on fact-present and pushes away from the call, or fires on fact-absent and pushes
toward it), and the RRM map fitted on these pairs with targets = the few-shot teacher (label-free, as E011). Per-item outcomes saved."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"; torch.manual_seed(0)
model, tok = load(NAME); NL = len(model.model.layers)
G = sorted({round(f * NL) for f in (1 / 3, .5, .58, 2 / 3, .83)} | {NL - 2}); PROBE_L = sorted(set(range(2, NL, 3)) | {NL - 1})
HIST = "If the needed fact already appears earlier in this conversation, use it and do not call recall_memory."
COND = {"none": lambda r: render(tok, r["prompt"], extra=r.get("history", ())),
        "fewshot": lambda r: render_cond(tok, r["prompt"], "fewshot", r.get("history", ())),
        "fewshot+history rule": lambda r: render(tok, r["prompt"], DEFAULT_SYS + " " + INSTRUCTION + " " + HIST, r.get("history", ()))}
def synth(rows, off=7):
    pos = [r for r in rows if r["label"] == 1]; out = []
    for i, r in enumerate(pos):
        other = pos[(i + off) % len(pos)]["memory_record"]
        for rec, lab in ((r["memory_record"], 0), (other, 1)):
            h = ([{"role": "user", "content": "For this chat, keep this in mind: " + rec}, {"role": "assistant", "content": "Got it, I'll keep that in mind."}] if i % 2 == 0 else
                 [{"role": "user", "content": "Quick recap of what we covered so far?"}, {"role": "assistant", "content": "Earlier in this chat: " + rec}])
            out.append({**r, "history": h, "label": lab, "kind": "in_context_synth"})
    return out
STR, SDV = synth(read("train")), synth(read("dev")); H2 = read("hard2"); HIC = [r for r in H2 if r["kind"] == "in_context"]; TEST = read("test")
y = lambda rows: np.array([r["label"] for r in rows])
def J(s, yy): return float((s[yy == 1] > 0).mean() - (s[yy == 0] > 0).mean())
out = {"meta": provenance(NAME, exp="E017", grid_layers=G, probe_layers=PROBE_L, history_rule=HIST), "probe": {}, "prompts": {}, "sweeps": {}, "selected": {}}

# Part A
_, Htr, _ = collect(model, tok, [COND["none"](r) for r in STR], PROBE_L)
Hev = {"synth_dev": collect(model, tok, [COND["none"](r) for r in SDV], PROBE_L)[1], "hard2_in_context": collect(model, tok, [COND["none"](r) for r in HIC], PROBE_L)[1]}
for l in PROBE_L:
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(Htr[l].numpy(), y(STR))
    out["probe"][l] = {"synth_dev": round(roc_auc_score(y(SDV), clf.decision_function(Hev["synth_dev"][l].numpy())), 4),
                       "hard2_in_context": round(roc_auc_score(y(HIC), clf.decision_function(Hev["hard2_in_context"][l].numpy())), 4)}
    print("PROBE", l, out["probe"][l], flush=True)
for c, f in COND.items():
    out["prompts"][c] = {}
    for k, rows in (("synth_dev", SDV), ("hard2_in_context", HIC)):
        s = decide(model, tok, [f(r) for r in rows]); out["prompts"][c][k] = {**rates(s, y(rows)), "J": round(J(s, y(rows)), 4)}
    print("PROMPT", c, {k: (v["tpr"], v["fpr"], v["auroc"]) for k, v in out["prompts"][c].items()}, flush=True)

# Part B
class Hook:
    def __init__(s, mod, fn): s.mod, s.fn = mod, fn
    def __enter__(s): s.h = s.mod.register_forward_hook(s.fn)
    def __exit__(s, *a): s.h.remove()
class Nothing:
    def __enter__(s): pass
    def __exit__(s, *a): pass
ytr = y(STR); A, H, _ = collect(model, tok, [COND["none"](r) for r in STR], G)
_, _, bg = collect(model, tok, background_texts(model, tok, [COND["none"](r) for r in STR]), G, want_all=True, max_rows=6000)
Hf = collect(model, tok, [COND["fewshot"](r) for r in STR + SDV], G)[1]; Hn = collect(model, tok, [COND["none"](r) for r in STR + SDV], G)[1]
devT = [COND["none"](r) for r in SDV]; C = {}
for l in G:
    dL = H[l][ytr == 1].mean(0) - H[l][ytr == 0].mean(0); D = dL / dL.norm() * H[l].norm(dim=1).mean()
    v_on = fit_gate(bg[l], A[l][ytr == 1], A[l][ytr == 0]); v_off = fit_gate(bg[l], A[l][ytr == 0], A[l][ytr == 1])
    for a in (.25, .5, 1, 2, 4, 8):
        C[f"gate+ L{l} a{a}"] = ("gated rank-one (+, fires on fact-absent)", dict(l=l, a=a), lambda D=D, v=v_on, l=l, a=a: Edit(model, l, D, v, a))
        C[f"gate- L{l} a{a}"] = ("gated rank-one (-, fires on fact-present)", dict(l=l, a=a), lambda D=-D, v=v_off, l=l, a=a: Edit(model, l, D, v, a))
    X = Hn[l].double(); Yd = Hf[l].double() - X; mx, my = X.mean(0), Yd.mean(0); Xc, Yc = X - mx, Yd - my; Kx = Xc @ Xc.T
    for lam in (.03, .1, .3):
        Al = torch.linalg.solve(Kx + lam * Xc.pow(2).sum(1).mean() * torch.eye(len(Kx), dtype=Kx.dtype), Yc); _, _, Vt = torch.linalg.svd(Kx @ Al, full_matrices=False)
        for r in (4, 16):
            Br = (Xc.T @ Al @ Vt[:r].T @ Vt[:r]).float().to(model.device)
            def fn(m, i, o, Br=Br, mx=mx.float().to(model.device), my=my.float().to(model.device)):
                x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[:, -1] += ((x[:, -1].float() - mx) @ Br + my).to(x.dtype); return (x, *o[1:]) if isinstance(o, tuple) else x
            C[f"rrm L{l} lam{lam} r{r}"] = ("RRM hook, few-shot teacher, fit on history pairs", dict(l=l, lam=lam, r=r), lambda fn=fn, l=l: Hook(model.model.layers[l], fn))
print("candidates", len(C), flush=True)
for name, (meth, cfg, ctx) in C.items():
    with ctx(): s = decide(model, tok, devT)
    out["sweeps"][name] = {"method": meth, **cfg, "J": round(J(s, y(SDV)), 4), "tpr": round(float((s[y(SDV) == 1] > 0).mean()), 4), "fpr": round(float((s[y(SDV) == 0] > 0).mean()), 4)}
best = {}
for name, d in out["sweeps"].items():
    if d["method"] not in best or d["J"] > out["sweeps"][best[d["method"]]]["J"]: best[d["method"]] = name
FINAL = {"no patch": (COND["none"], Nothing), "few-shot prompt": (COND["fewshot"], Nothing), "few-shot + history rule (prompt)": (COND["fewshot+history rule"], Nothing),
         **{f"{m} [{n}]": (COND["none"], C[n][2]) for m, n in best.items()}}
DEVJ = {f"{m} [{n}]": out["sweeps"][n] for m, n in best.items()}
def e2e(rf, rows):
    pos = [r for r in rows if r["label"] == 1]; g = generate(model, tok, [rf(r) for r in pos], max_new=48); fol, idx = [], []; used = {r["id"]: 0 for r in pos}
    for r, t in zip(pos, g):
        c = parse_call(t); nm = c.get("name") if c and not c.get("malformed") else None; q = str((c.get("arguments") or {}).get("query", "")) if nm else ""
        if nm == "recall_memory" and q: fol.append(followup(tok, r["prompt"], q, t, r["memory_record"], extra=r.get("history", ()))); idx.append(r)
    for r, a in zip(idx, generate(model, tok, fol, max_new=64) if fol else []): used[r["id"]] = int(r["answer_substr"].lower() in a.lower() and "<tool_call>" not in a)
    return used
for name, (rf, ctx) in FINAL.items():
    res = {"per_item": {}, "dev_selection": DEVJ.get(name)}
    with ctx():
        for k, rows in (("synth_dev", SDV), ("hard2_in_context", HIC), ("hard2_all", H2), ("test", TEST)):
            s = decide(model, tok, [rf(r) for r in rows]); res[k] = rates(s, y(rows)); res["per_item"][k] = {r["id"] + ("" if k != "synth_dev" else f"#{r['label']}"): round(float(x), 3) for r, x in zip(rows, s)}
        u = e2e(rf, H2); res["per_item"]["hard2_used"] = u
        res["hard2_used"] = round(float(np.mean(list(u.values()))), 4); res["hard2_in_context_used"] = round(float(np.mean([u[r["id"]] for r in HIC if r["label"] == 1])), 4)
    out["selected"][name] = res
    print("SEL", f"{name[:70]:70s}", "| hard2 IC", res["hard2_in_context"]["tpr"], res["hard2_in_context"]["fpr"], res["hard2_in_context"]["auroc"],
          "| hard2 all", res["hard2_all"]["tpr"], res["hard2_all"]["fpr"], "| test", res["test"]["tpr"], res["test"]["fpr"], "| used all/IC", res["hard2_used"], res["hard2_in_context_used"], flush=True)
save("E017_history_" + NAME.split("/")[-1], out); print("E017_DONE")
