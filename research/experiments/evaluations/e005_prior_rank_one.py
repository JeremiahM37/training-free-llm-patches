"""E005: closest published rank-one edits on our task. TFTV (2609.09054 Eq. 2-3) and Steer2Edit (2602.09870 Sec 3.5, single-component form),
reimplemented from the papers on the same steering vector, same fit data, selected on dev by J like our method. Single module (down_proj) for comparability;
both papers allow many modules, so this is not a replication of their full configurations."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
NAME = sys.argv[1]; nlf = (.5, .6, .7, .8, .9, .97)
model, tok = load(NAME); nl = model.config.num_hidden_layers; LAYERS = sorted({int(nl * f) for f in nlf})
tr = read("train"); ytr = np.array([r["label"] for r in tr]); SPL = {k: read(k) for k in ("dev", "adversarial", "contrast", "style2", "cantanswer")}
sch = [render_cond(tok, r["prompt"], "schema") for r in tr]; ins = [render_cond(tok, r["prompt"], "instruction") for r in tr]
A, H, _ = collect(model, tok, sch, LAYERS); _, Hi, _ = collect(model, tok, ins, LAYERS)
T = {k: [render(tok, r["prompt"]) for r in d] for k, d in SPL.items()}; Y = {k: [r["label"] for r in d] for k, d in SPL.items()}
class Add:
    def __init__(s, l, E): s.w, s.E = down(model, l).weight, E.cuda()
    def __enter__(s): s.old = s.w.data.clone(); s.w.data = (s.w.data.float() + s.E).to(s.old.dtype)
    def __exit__(s, *a): s.w.data = s.old
rows = []
for l in LAYERS:
    dH = (Hi[l][ytr == 1] - H[l][ytr == 1]).mean(0); s = dH / dH.norm(); W = down(model, l).weight.detach().float().cpu(); mu = A[l].mean(0)
    Us, S, Vh = torch.linalg.svd(W, full_matrices=False)
    tftv = s[:, None] @ ((torch.sign(Vh @ mu) * S)[None] @ Vh)
    k = W.T @ s; k = k / k.norm(); s2e = s[:, None] @ k[None]
    for name, E, grid in (("tftv", tftv, [.0005, .001, .002, .005, .01, .02, .05]), ("steer2edit", s2e, [.5, 1, 2, 4, 8, 16, 32])):
        for sign in (1, -1):
            for a in grid:
                with Add(l, sign * a * E): r = rates(decide(model, tok, T["dev"]), Y["dev"])
                r.update(layer=l, method=name, alpha=sign * a, J=round(r["tpr"] - r["fpr"], 4)); rows.append(r)
    print("L", l, "done", flush=True)
tops = {}
for m in ("tftv", "steer2edit"):
    b = max((r for r in rows if r["method"] == m), key=lambda r: r["J"]); l = b["layer"]; res = {"dev_select": b}
    dH = (Hi[l][ytr == 1] - H[l][ytr == 1]).mean(0); s = dH / dH.norm(); W = down(model, l).weight.detach().float().cpu(); mu = A[l].mean(0)
    if m == "tftv":
        _, S, Vh = torch.linalg.svd(W, full_matrices=False); E = s[:, None] @ ((torch.sign(Vh @ mu) * S)[None] @ Vh)
    else:
        k = W.T @ s; E = s[:, None] @ (k / k.norm())[None]
    with Add(l, b["alpha"] * E):
        for kk in SPL:
            sc = decide(model, tok, T[kk])
            res[kk] = {"fpr": round(float((sc > 0).mean()), 4)} if kk == "cantanswer" else rates(sc, Y[kk])
    tops[m] = res; print("TOP", m, json.dumps({k: (v.get("tpr"), v.get("fpr"), v.get("auroc")) for k, v in res.items()}), {x: b[x] for x in ("layer", "alpha")}, flush=True)
save(f"E005_{NAME.split('/')[-1]}", {"meta": provenance(NAME, exp="E005", layers=LAYERS), "rows": rows, "tops": tops}); print("E005_DONE")
