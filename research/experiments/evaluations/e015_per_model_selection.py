"""E015: per-model selection on SmolLM3-3B. E012 froze the Qwen3-4B recipes and every one failed (no patch FPR .97, AUROC .33).
Question: does dev-only selection fix it, or is the call decision not reachable by these patches on this model?
Part A (diagnostic): linear probe on the last-token residual per layer, fit on train labels, AUROC on dev/adversarial/contrast;
                     decisions of each prompt condition (the possible teachers).
Part B (selection): every method's layer/rank/alpha/teacher is chosen on dev only, by J = TPR - FPR at threshold 0.
                    Label-free maps fit on train+dev prompts only (labels unused). Gated and steering edits use train labels.
Evaluation of the selected configs: adversarial, contrast, style2, cantanswer (FPR) and test (decisions), end-to-end on test positives.
test.jsonl was used once for SmolLM3 in E012 with frozen recipes; nothing here is selected on it."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
NAME = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM3-3B"; torch.manual_seed(0)
model, tok = load(NAME); NL = len(model.model.layers)
G = sorted({round(f * NL) for f in (1 / 3, .5, 2 / 3, .83)} | {NL - 2})
PROBE_L = sorted(set(range(2, NL, 3)) | {NL - 1})
RULES = "Rules for recall_memory. Call it only if answering requires a personal fact from an EARLIER conversation with this user that is not in the current message (their preferences, past plans, things they told you). Never call it for general knowledge, math, code, writing, or when the message already contains the needed facts. If you call it, output only the tool call."
COND = {"none": lambda p: render(tok, p), "instruction": lambda p: render_cond(tok, p, "instruction"),
        "fewshot": lambda p: render_cond(tok, p, "fewshot"), "rules": lambda p: render(tok, p, DEFAULT_SYS + " " + RULES)}
SPL = {k: read(k) for k in ("train", "dev", "adversarial", "contrast", "style2", "cantanswer", "test")}
lab = lambda k: np.array([r["label"] for r in SPL[k]]); P = lambda k: [r["prompt"] for r in SPL[k]]
ytr, ydv = lab("train"), lab("dev")
def J(s, y): return float((s[y == 1] > 0).mean() - (s[y == 0] > 0).mean())
out = {"meta": provenance(NAME, exp="E015", grid_layers=G, probe_layers=PROBE_L, selection="dev J = TPR - FPR at 0"), "probe": {}, "prompts": {}, "sweeps": {}, "selected": {}}

# Part A: probes
A_tr, H_tr, _ = collect(model, tok, [render(tok, p) for p in P("train")], PROBE_L)
Hs = {k: collect(model, tok, [render(tok, p) for p in P(k)], PROBE_L)[1] for k in ("dev", "adversarial", "contrast")}
for l in PROBE_L:
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(H_tr[l].numpy(), ytr)
    out["probe"][l] = {k: round(roc_auc_score(lab(k), clf.decision_function(Hs[k][l].numpy())), 4) for k in Hs}
    print("PROBE", l, out["probe"][l], flush=True)
for c, f in COND.items():
    s = decide(model, tok, [f(p) for p in P("dev")]); out["prompts"][c] = {**rates(s, ydv), "J": round(J(s, ydv), 4)}
    print("PROMPT", c, out["prompts"][c]["tpr"], out["prompts"][c]["fpr"], out["prompts"][c]["auroc"], flush=True)

# Part B: candidate patches
class Add:
    def __init__(s, l, E): s.w, s.E = down(model, l).weight, E.to(model.device)
    def __enter__(s): s.old = s.w.data.clone(); s.w.data = (s.w.data.float() + s.E).to(s.old.dtype)
    def __exit__(s, *a): s.w.data = s.old
class Hook:
    def __init__(s, mod, fn): s.mod, s.fn = mod, fn
    def __enter__(s): s.h = s.mod.register_forward_hook(s.fn)
    def __exit__(s, *a): s.h.remove()
class Nothing:
    def __enter__(s): pass
    def __exit__(s, *a): pass
fitp = list(dict.fromkeys(P("train") + P("dev")))  # label-free fit pool
A, H, _ = collect(model, tok, [render(tok, p) for p in P("train")], G); _, Hi, _ = collect(model, tok, [COND["instruction"](p) for p in P("train")], G)
_, _, bg = collect(model, tok, background_texts(model, tok, [render(tok, p) for p in P("train")]), G, want_all=True, max_rows=6000)
Ap, Hp, _ = collect(model, tok, [render(tok, p) for p in fitp], G)
Ht = {c: collect(model, tok, [COND[c](p) for p in fitp], G)[1] for c in ("instruction", "fewshot", "rules")}
devT = [render(tok, p) for p in P("dev")]
C = {}  # name -> (method, cfg, ctx factory)
def sdir(l): d = (Hi[l][ytr == 1] - H[l][ytr == 1]).mean(0); return d / d.norm()
for l in G:
    W = down(model, l).weight.detach().float().cpu(); _, S_, Vh = torch.linalg.svd(W, full_matrices=False); k = W.T @ sdir(l)
    base_t = sdir(l)[:, None] @ ((torch.sign(Vh @ A[l].mean(0)) * S_)[None] @ Vh); base_s = sdir(l)[:, None] @ (k / k.norm())[None]
    for a in (-1, -.5, -.2, -.1, -.05, .05, .1, .2, .5, 1): C[f"tftv L{l} a{a}"] = ("TFTV rank-one", dict(l=l, a=a), lambda E=a * base_t, l=l: Add(l, E))
    for a in (-256, -128, -64, -32, -16, -8, 8, 16, 32, 64): C[f"s2e L{l} a{a}"] = ("Steer2Edit rank-one", dict(l=l, a=a), lambda E=a * base_s, l=l: Add(l, E))
    dL = H[l][ytr == 1].mean(0) - H[l][ytr == 0].mean(0); D = dL / dL.norm() * H[l].norm(dim=1).mean()
    v_on = fit_gate(bg[l], A[l][ytr == 1], A[l][ytr == 0]); v_off = fit_gate(bg[l], A[l][ytr == 0], A[l][ytr == 1])
    for a in (.25, .5, 1, 2, 4):
        C[f"gate+ L{l} a{a}"] = ("gated rank-one", dict(l=l, a=a, gate="fires on positives, pushes toward call"), lambda D=D, v=v_on, l=l, a=a: Edit(model, l, D, v, a))
        C[f"gate- L{l} a{a}"] = ("gated rank-one", dict(l=l, a=a, gate="fires on negatives, pushes away from call"), lambda D=-D, v=v_off, l=l, a=a: Edit(model, l, D, v, a))
    for c in ("instruction", "fewshot", "rules"):
        X = Hp[l].double(); Y = Ht[c][l].double() - X; mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; Kx = Xc @ Xc.T
        for lam in (.03, .1, .3):
            Al = torch.linalg.solve(Kx + lam * Xc.pow(2).sum(1).mean() * torch.eye(len(Kx), dtype=Kx.dtype), Yc); _, _, Vt = torch.linalg.svd(Kx @ Al, full_matrices=False)
            for r in (1, 4, 16):
                Br = (Xc.T @ Al @ Vt[:r].T @ Vt[:r]).float().to(model.device)
                def fn(m, i, o, Br=Br, mx=mx.float().to(model.device), my=my.float().to(model.device)):
                    x = o[0] if isinstance(o, tuple) else o; x = x.clone(); x[:, -1] += ((x[:, -1].float() - mx) @ Br + my).to(x.dtype); return (x, *o[1:]) if isinstance(o, tuple) else x
                C[f"rrm {c} L{l} lam{lam} r{r}"] = ("reduced-rank map (hook)", dict(l=l, teacher=c, lam=lam, r=r), lambda fn=fn, l=l: Hook(model.model.layers[l], fn))
bgp = collect(model, tok, background_texts(model, tok, [render(tok, p) for p in fitp[:150]], n_gen=60), G, want_all=True, max_rows=4000)[2]
for l in G[:3]:
    for c in ("fewshot", "rules"):
        for lam in (.1, 1):
            for r in (4, 16):
                Xb = bgp[l].double() * .1; Xa = torch.cat([Ap[l].double(), Xb])
                # target: change in this layer's residual output, attributed to its down_proj (as E010)
                Ya = torch.cat([(Ht[c][l] - Hp[l]).double(), torch.zeros(len(Xb), Hp[l].shape[1]).double()])
                K = Xa @ Xa.T; Al = torch.cholesky_solve(Ya, torch.linalg.cholesky(K + lam * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype)))
                _, _, Vt = torch.linalg.svd(K @ Al, full_matrices=False); B = ((Xa.T @ Al) @ (Vt[:r].T @ Vt[:r])).float().to(model.device)
                C[f"cflora {c} L{l} lam{lam} r{r}"] = ("closed-form LoRA (no labels)", dict(l=l, teacher=c, lam=lam, r=r), lambda B=B, l=l: Hook(down(model, l), lambda m, i, o, B=B: o + (i[0].float() @ B).to(o.dtype)))
print("candidates", len(C), flush=True)
for name, (meth, cfg, ctx) in C.items():
    with ctx(): s = decide(model, tok, devT)
    out["sweeps"][name] = {"method": meth, **cfg, "J": round(J(s, ydv), 4), "tpr": round(float((s[ydv == 1] > 0).mean()), 4), "fpr": round(float((s[ydv == 0] > 0).mean()), 4), "auroc": round(auroc(s[ydv == 1], s[ydv == 0]), 4)}
best = {}
for name, d in out["sweeps"].items():
    if d["method"] not in best or d["J"] > out["sweeps"][best[d["method"]]]["J"]: best[d["method"]] = name
bc = max(out["prompts"], key=lambda c: out["prompts"][c]["J"]); print("best prompt on dev", bc, flush=True)
FINAL = {"no patch": (COND["none"], Nothing, None), f"best prompt on dev ({bc})": (COND[bc], Nothing, None),
         **{f"{m} [{n}]": (COND["none"], C[n][2], n) for m, n in best.items()}}
pos = [r for r in SPL["test"] if r["label"] == 1]
for name, (rf, ctx, cfg) in FINAL.items():
    res = {"config": cfg, "dev": out["sweeps"][cfg] if cfg else out["prompts"][bc if "best" in name else "none"]}
    with ctx():
        for k in ("adversarial", "contrast", "style2", "test", "cantanswer"):
            s = decide(model, tok, [rf(p) for p in P(k)])
            res[k] = {"fpr": round(float((s > 0).mean()), 4), "n": len(s)} if k == "cantanswer" else rates(s, lab(k))
        g = generate(model, tok, [rf(r["prompt"]) for r in pos], max_new=48); fol, idx = [], []; called = use = 0
        for r, t in zip(pos, g):
            c = parse_call(t); nm = c.get("name") if c and not c.get("malformed") else None; q = str((c.get("arguments") or {}).get("query", "")) if nm else ""; called += bool(c)
            if nm == "recall_memory" and q: fol.append(followup(tok, r["prompt"], q, t, r["memory_record"])); idx.append(r)
        for r, a in zip(idx, generate(model, tok, fol, max_new=64) if fol else []): use += r["answer_substr"].lower() in a.lower() and "<tool_call>" not in a
    res["test_e2e"] = {"called": round(called / len(pos), 4), "used": round(use / len(pos), 4), "used_ci": wilson(use, len(pos))}
    out["selected"][name] = res
    print("SEL", f"{name:60s}", "dev J", res["dev"].get("J"), "| test", res["test"]["tpr"], res["test"]["fpr"], res["test"]["auroc"],
          "| adv fpr", res["adversarial"]["fpr"], "contrast fpr", res["contrast"]["fpr"], "cant", res["cantanswer"]["fpr"], "| used", res["test_e2e"]["used"], flush=True)
save("E015_selection_" + NAME.split("/")[-1], out); print("E015_DONE")
