"""E002: closed-form rank-one gated edit (T01) versus steering baselines. Fit on train (80 prompts), select on dev.
Conditions: uncond = h+aD at every position; lastpos = h+aD at the decision token only (an oracle for position, which a
hook can do and a weight edit cannot); cast = cosine-gated hook (CAST-style, threshold fit on train);
lcmv = rank-one edit, g = v'a with v from the constrained filter; lda = same edit with v from ridge LDA, scaled to unit
gain on positives (no zero-mean constraint), the control for whether the constraint matters."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from lib import *
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "6,10,14,16,18,20,22,24,26".split(","))]
ALPHAS = [0.25, 0.5, 1, 1.5, 2, 3, 4, 6]
LAM = 1e-2; SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)
model, tok = load(name)
layers = model.model.layers
tr, dev, adv = read("train"), read("dev"), read("adversarial")
ytr = np.array([r["label"] for r in tr])

@torch.no_grad()
def collect(texts, want_all=False, bs=16):
    """key a (pre down_proj) and h_out per layer: last position always, optionally all non-padding positions."""
    last_a, last_h, all_a = {l: [] for l in LAYERS}, {l: [] for l in LAYERS}, {l: [] for l in LAYERS}
    cur = {}
    hs = []
    for l in LAYERS:
        hs.append(layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: cur.__setitem__(("a", l), a[0].float())))
        hs.append(layers[l].register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(("h", l), (o[0] if isinstance(o, tuple) else o).float())))
    for b in batches(texts, bs):
        enc = tok(b, return_tensors="pt", padding=True).to(model.device)
        model(**enc, logits_to_keep=1)
        for l in LAYERS:
            last_a[l].append(cur["a", l][:, -1].cpu()); last_h[l].append(cur["h", l][:, -1].cpu())
            if want_all: all_a[l].append(cur["a", l][enc.attention_mask.bool()].cpu())
    for h in hs: h.remove()
    pack = lambda d: {l: torch.cat(v) for l, v in d.items() if v}
    return pack(last_a), pack(last_h), pack(all_a)

schema = [render_cond(tok, r["prompt"], "schema") for r in tr]
instr = [render_cond(tok, r["prompt"], "instruction") for r in tr]
t0 = time.time()
# background: every position of training prompts, plus generated continuations, so the gate is quiet on output tokens too
gens = generate(model, tok, schema[:40], max_new=32)
bg_texts = schema + [s + g.replace("<|im_end|>", "").replace("<|endoftext|>", "") for s, g in zip(schema[:40], gens)]
_, _, bg = collect(bg_texts, want_all=True)
a_s, h_s, _ = collect(schema); _, h_i, _ = collect(instr)
collect_s = time.time() - t0

def fit_gate(l, kind):
    # CPU double precision: cusolver is unavailable on the desktop build, and float32 is too coarse for this solve
    X = bg[l].double()                               # n x m, uncentered on purpose
    if X.shape[0] > 6000: X = X[torch.randperm(X.shape[0])[:6000]]
    n = X.shape[0]; A = a_s[l].double()
    mp, mn = A[ytr == 1].mean(0), A[ytr == 0].mean(0)
    lam = LAM * X.pow(2).sum(1).mean() / X.shape[1]
    L = torch.linalg.cholesky(X @ X.T / n + lam * torch.eye(n, dtype=X.dtype))
    def Rinv(b):                                      # Woodbury for (lam I + X'X/n)^-1 b
        return (b - X.T @ torch.cholesky_solve(((X @ b) / n)[:, None], L)[:, 0]) / lam
    if kind == "lcmv":
        M = torch.stack([mp, mn], 1); g = torch.tensor([1., 0.], dtype=X.dtype)
        RM = torch.stack([Rinv(M[:, 0]), Rinv(M[:, 1])], 1)
        v = RM @ torch.linalg.solve(M.T @ RM, g)
    elif kind == "lda":
        v = Rinv(mp - mn); v = v / (v @ mp)
    return v.float(), float((X @ v).pow(2).mean())

def run(texts, hook_fn, target):
    h = target.register_forward_hook(hook_fn)
    try: return decide(model, tok, texts)
    finally: h.remove()

def out0(o): return o[0] if isinstance(o, tuple) else o
def wrap(o, new): return (new, *o[1:]) if isinstance(o, tuple) else new

sets = {"dev": ([render_cond(tok, r["prompt"], "schema") for r in dev], [r["label"] for r in dev]),
        "adv": ([render_cond(tok, r["prompt"], "schema") for r in adv], [r["label"] for r in adv])}
base = {k: rates(decide(model, tok, t), y) for k, (t, y) in sets.items()}
print("base", base, flush=True)
rows, gates = [], {}
DA, DV = sets["dev"][0], sets["dev"][1]
for l in LAYERS:
    dH = (h_i[l][ytr == 1] - h_s[l][ytr == 1]).mean(0)                # prompted minus unprompted on positives
    dL = h_s[l][ytr == 1].mean(0) - h_s[l][ytr == 0].mean(0)          # positive minus negative, unprompted
    typical = h_s[l].norm(dim=1).mean()
    vals = {"dH": (dH / dH.norm() * typical).cuda().bfloat16(), "dL": (dL / dL.norm() * typical).cuda().bfloat16()}
    # CAST-style cosine gate on h at last position, threshold midway between class mean cosines
    cdir = (dL / dL.norm()).cuda().float()
    cos_tr = torch.nn.functional.cosine_similarity(h_s[l].cuda().float(), cdir[None], dim=1)
    th = float((cos_tr[ytr == 1].mean() + cos_tr[ytr == 0].mean()) / 2)
    vs = {}
    for kind in ("lcmv", "lda"):
        v, energy = fit_gate(l, kind); vs[kind] = v.cuda()
    # gate quality on dev, held out, before any behaviour test
    ad, _, _ = collect(DA); y = np.array(DV)
    gq = {}
    for kind in vs:
        g = (ad[l].cuda().float() @ vs[kind]).cpu().numpy()
        gq[kind] = {"auroc": auroc(g[y == 1], g[y == 0]), "mean_pos": float(g[y == 1].mean()), "mean_neg": float(g[y == 0].mean()), "std_neg": float(g[y == 0].std())}
    bgs = bg[l].cuda().float()
    gq["bg_abs_gt_half_lcmv"] = float(((bgs @ vs["lcmv"]).abs() > .5).float().mean())
    print("layer", l, "typical_norm", float(typical), "gates", gq, flush=True)
    gates[l] = gq
    def h_uncond(D, a):
        return lambda m, i, o: wrap(o, out0(o) + a * D)
    def h_last(D, a):
        def f(m, i, o):
            x = out0(o).clone()
            if x.shape[1] > 1: x[:, -1] += a * D
            return wrap(o, x)
        return f
    def h_cast(D, a):
        def f(m, i, o):
            x = out0(o); c = torch.nn.functional.cosine_similarity(x.float(), cdir[None, None], dim=-1)
            return wrap(o, x + a * (c > th)[..., None].to(x.dtype) * D)
        return f
    def h_rank1(D, a, v):
        # identical to W' = W + a D v' on down_proj; applied at the module output
        cap = {}
        pre = layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("a", i[0]))
        def f(m, i, o):
            return o + a * (cap["a"].float() @ v)[..., None].to(o.dtype) * D
        return f, pre
    for vn, D in vals.items():
        for a in ALPHAS:
            for kind in ("uncond", "lastpos", "cast", "lcmv", "lda"):
                if kind == "uncond": s = run(DA, h_uncond(D, a), layers[l])
                elif kind == "lastpos": s = run(DA, h_last(D, a), layers[l])
                elif kind == "cast": s = run(DA, h_cast(D, a), layers[l])
                else:
                    f, pre = h_rank1(D, a, vs[kind]); s = run(DA, f, layers[l].mlp.down_proj); pre.remove()
                r = rates(s, DV); r.update(layer=l, value=vn, alpha=a, kind=kind, J=round(r["tpr"] - r["fpr"], 4))
                rows.append(r)
    best = max((r for r in rows if r["layer"] == l), key=lambda r: r["J"])
    print("  best at layer", {k: best[k] for k in ("kind", "value", "alpha", "tpr", "fpr", "J", "auroc")}, flush=True)
tops = {}
for kind in ("uncond", "lastpos", "cast", "lcmv", "lda"):
    tops[kind] = max((r for r in rows if r["kind"] == kind), key=lambda r: r["J"])
    print("TOP", kind, {k: tops[kind][k] for k in ("layer", "value", "alpha", "tpr", "fpr", "J", "auroc")}, flush=True)
save(f"E002_{name.split('/')[-1]}", {"meta": provenance(name, exp="E002", layers=LAYERS, alphas=ALPHAS, lam=LAM, seed=SEED, collect_s=collect_s), "base": base, "gates": gates, "rows": rows, "tops": tops})
print("E002_DONE")
