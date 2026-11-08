"""E013: controls for the reduced-rank-map (RRM) hook result of E011, and wider alpha grids for the published rank-one baselines.
Raised by critique 02: is the RRM gain due to the learned map, or to intervening at that position at all?
Arms at the Qwen3-4B L34 output, last position:
  RRM r4 every-token  -- exactly as E011 ran it: the hook edits the last position of every forward call, so during generation it acts on every new token
  RRM r4 prefill-only -- acts only on the decision token (the prefill call); generation afterwards is untouched
  constant shift      -- adds the mean prompt effect only (the best constant approximation), every token
  random rank-4 map   -- same singular values as the RRM map, random singular vectors, plus the same mean; 3 seeds
Plus MMLU (400 items, schema absent and present) under no patch / RRM / constant, and TFTV / Steer2Edit re-selected on dev over a wider grid.
No test.jsonl: it was spent in E011. Fit pool and configuration are identical to E011."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
from scipy.stats import binomtest
NAME = "Qwen/Qwen3-4B-Instruct-2507"; torch.manual_seed(0)
model, tok = load(NAME)
tr = read("train"); ytr = np.array([r["label"] for r in tr]); ptr = [r["prompt"] for r in tr]
pool = list(dict.fromkeys(r["prompt"] for s in ("train", "dev", "adversarial", "contrast", "style2") for r in read(s)))
L = 34
A, H, _ = collect(model, tok, [render(tok, p) for p in ptr], [25, L]); _, Hi, _ = collect(model, tok, [render_cond(tok, p, "instruction") for p in ptr], [25, L])
_, Hp, _ = collect(model, tok, [render(tok, p) for p in pool], [L]); _, Htp, _ = collect(model, tok, [render_cond(tok, p, "fewshot") for p in pool], [L])
X = Hp[L].double(); Y = Htp[L].double() - X; mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; Kx = Xc @ Xc.T
Al = torch.linalg.solve(Kx + .1 * Xc.pow(2).sum(1).mean() * torch.eye(len(Kx), dtype=Kx.dtype), Yc); _, _, Vt = torch.linalg.svd(Kx @ Al, full_matrices=False)
Br = (Xc.T @ Al @ Vt[:4].T @ Vt[:4]); S4 = torch.linalg.svdvals(Br)[:4]
def rand_map(seed):
    g = torch.Generator().manual_seed(seed); d = Br.shape[0]
    Qi, _ = torch.linalg.qr(torch.randn(d, 4, generator=g, dtype=Br.dtype)); Qo, _ = torch.linalg.qr(torch.randn(d, 4, generator=g, dtype=Br.dtype))
    return Qi @ torch.diag(S4) @ Qo.T
dv = lambda t: t.float().to(model.device)
mxg, myg = dv(mx), dv(my)
class Hook:
    def __init__(s, B, prefill_only=False, const_only=False): s.B, s.po, s.co = (None if B is None else dv(B)), prefill_only, const_only
    def fn(s, m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        if s.po and x.shape[1] == 1: return o
        x = x.clone(); d = myg if s.co else (x[:, -1].float() - mxg) @ s.B + myg
        x[:, -1] += d.to(x.dtype); return (x, *o[1:]) if isinstance(o, tuple) else x
    def __enter__(s): s.h = model.model.layers[L].register_forward_hook(s.fn)
    def __exit__(s, *a): s.h.remove()
class Nothing:
    def __enter__(s): pass
    def __exit__(s, *a): pass
class Add:
    def __init__(s, l, E): s.w, s.E = down(model, l).weight, E.to(model.device)
    def __enter__(s): s.old = s.w.data.clone(); s.w.data = (s.w.data.float() + s.E).to(s.old.dtype)
    def __exit__(s, *a): s.w.data = s.old
SPL = {k: read(k) for k in ("dev", "adversarial", "contrast", "style2", "cantanswer")}
POS = [r for k in ("adversarial", "contrast", "style2") for r in SPL[k] if r["label"] == 1]  # the 150 non-test positives of E010
def decisions(rf):
    out = {}
    for k, rows in SPL.items():
        s = decide(model, tok, [rf(r["prompt"]) for r in rows])
        out[k] = {"fpr": round(float((s > 0).mean()), 4), "n": len(rows)} if k == "cantanswer" else rates(s, np.array([r["label"] for r in rows]))
    return out
def e2e(rf):
    g = generate(model, tok, [rf(r["prompt"]) for r in POS], max_new=48); fol, idx = [], []; called = ex = 0; used = np.zeros(len(POS), int)
    for j, (r, t) in enumerate(zip(POS, g)):
        c = parse_call(t); nm = c.get("name") if c and not c.get("malformed") else None; q = str((c.get("arguments") or {}).get("query", "")) if nm else ""; called += bool(c)
        if nm == "recall_memory" and q:
            ex += any(w in q.lower() for w in r.get("query_terms", [])) if r.get("query_terms") else 1
            msgs = [{"role": "system", "content": DEFAULT_SYS}, {"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "recall_memory", "arguments": {"query": q}}}]}, {"role": "tool", "content": r["memory_record"]}]
            fol.append(tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False)); idx.append(j)
    for j, a in zip(idx, generate(model, tok, fol, max_new=64) if fol else []): used[j] = POS[j]["answer_substr"].lower() in a.lower() and "<tool_call>" not in a
    n = len(POS); return {"called": round(called / n, 4), "executed": round(ex / n, 4), "used": round(used.mean(), 4), "use_ci": wilson(int(used.sum()), n), "n": n}, used
from datasets import load_dataset
mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0).select(range(400)); ids = [tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"]
mq = lambda r: r["question"] + "\n" + "\n".join(f"{c}. {x}" for c, x in zip("ABCD", r["choices"])) + "\nAnswer with the letter only."
ns = [tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYS}, {"role": "user", "content": mq(r)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in mm]; ws = [render(tok, mq(r)) for r in mm]
@torch.no_grad()
def mmlu(texts):
    hit = []
    for bi, b in enumerate(batches(texts, 8)):
        lg = model(**tok(b, return_tensors="pt", padding=True).to(model.device), logits_to_keep=1).logits[:, -1].float()
        hit += [int(lg[j, ids].argmax()) == mm[bi * 8 + j]["answer"] for j in range(len(b))]
    return np.array(hit, int)
def mcnemar(a, b):
    w, bt = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return {"base": round(a.mean(), 4), "patched": round(b.mean(), 4), "worse": w, "better": bt, "p": round(binomtest(w, w + bt).pvalue, 4) if w + bt else 1.0}
plain = lambda p: render(tok, p)
ARMS = {"no patch": (plain, Nothing), "few-shot prompt": (lambda p: render_cond(tok, p, "fewshot"), Nothing),
        "RRM r4 every-token (as E011)": (plain, lambda: Hook(Br)), "RRM r4 prefill-only": (plain, lambda: Hook(Br, prefill_only=True)),
        "constant shift (mean effect)": (plain, lambda: Hook(None, const_only=True)), "constant shift prefill-only": (plain, lambda: Hook(None, prefill_only=True, const_only=True))}
for sd in (1, 2, 3): ARMS[f"random rank-4 map, seed {sd}"] = (plain, (lambda B: lambda: Hook(B))(rand_map(sd)))
out = {"meta": provenance(NAME, exp="E013", layer=L, rank=4, rrm_singular_values=[round(float(x), 3) for x in S4]), "arms": {}, "mmlu": {}, "baselines_wide_grid": {}}
used_by = {}
for name, (rf, ctx) in ARMS.items():
    with ctx(): d = decisions(rf); e, used_by[name] = e2e(rf)
    out["arms"][name] = {"decision": d, "e2e": e}
    print("ARM", f"{name:34s}", {k: (v.get("tpr"), v["fpr"]) for k, v in d.items()}, "e2e", e["called"], e["executed"], e["used"], flush=True)
base_e = used_by["no patch"]
for name in used_by: out["arms"][name]["e2e_vs_no_patch_mcnemar"] = mcnemar(base_e, used_by[name])
mb_n, mb_s = mmlu(ns), mmlu(ws)
for name in ("RRM r4 every-token (as E011)", "RRM r4 prefill-only", "constant shift (mean effect)"):
    with ARMS[name][1](): out["mmlu"][name] = {"no_schema": mcnemar(mb_n, mmlu(ns)), "schema": mcnemar(mb_s, mmlu(ws))}
    print("MMLU", name, out["mmlu"][name], flush=True)
# TFTV and Steer2Edit on a wider grid, selected by J on dev only, same construction as E005/E011
def sdir(l): d_ = (Hi[l][ytr == 1] - H[l][ytr == 1]).mean(0); return d_ / d_.norm()
W34 = down(model, 34).weight.detach().float().cpu(); _, S_, Vh = torch.linalg.svd(W34, full_matrices=False)
E_t = sdir(34)[:, None] @ ((torch.sign(Vh @ A[34].mean(0)) * S_)[None] @ Vh)
W25 = down(model, 25).weight.detach().float().cpu(); k_ = W25.T @ sdir(25); E_s = sdir(25)[:, None] @ (k_ / k_.norm())[None]
dev_y = np.array([r["label"] for r in SPL["dev"]]); dev_t = [plain(r["prompt"]) for r in SPL["dev"]]
for mname, l, E, grid in (("TFTV", 34, E_t, [.02, .05, .1, .2, .5, 1.0]), ("Steer2Edit", 25, E_s, [-8, -16, -32, -64, -128, -256, 32, 64, 128])):
    rows = []
    for a in grid:
        with Add(l, a * E): r_ = rates(decide(model, tok, dev_t), dev_y)
        rows.append({"alpha": a, "tpr": r_["tpr"], "fpr": r_["fpr"], "J": round(r_["tpr"] - r_["fpr"], 4)}); print("GRID", mname, rows[-1], flush=True)
    b = max(rows, key=lambda x: x["J"])
    with Add(l, b["alpha"] * E): d = decisions(plain); e, u = e2e(plain)
    out["baselines_wide_grid"][mname] = {"layer": l, "grid": rows, "best": b, "best_at_grid_edge": b["alpha"] in (grid[0], grid[-1]) or (mname == "Steer2Edit" and b["alpha"] in (-256, 128)), "decision": d, "e2e": e, "e2e_vs_no_patch_mcnemar": mcnemar(base_e, u)}
    print("BEST", mname, b, "e2e", e, flush=True)
save("E013_hook_controls_Qwen3-4B-Instruct-2507", out); print("E013_DONE")
