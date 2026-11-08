"""E008 (H12, H10): dtype effect on a merged edit; signed versus rectified gate. 1.5B model, float32, config frozen from E003."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
NAME = "Qwen/Qwen2.5-1.5B-Instruct"; l, a = 26, .25
model, tok = load(NAME, dtype=torch.float32)
tr = read("train"); y = np.array([r["label"] for r in tr]); sch = [render(tok, r["prompt"]) for r in tr]; ins = [render_cond(tok, r["prompt"], "instruction") for r in tr]
_, _, bg = collect(model, tok, background_texts(model, tok, sch), [l], want_all=True); A, H, _ = collect(model, tok, sch, [l]); _, Hi, _ = collect(model, tok, ins, [l])
v = fit_gate(bg[l], A[l][y == 1], A[l][y == 0]); d = (Hi[l][y == 1] - H[l][y == 1]).mean(0); D = d / d.norm() * H[l].norm(dim=1).mean()
dev = read("dev"); T = [render(tok, r["prompt"]) for r in dev]; Y = [r["label"] for r in dev]
adv = read("adversarial"); Ta = [render(tok, r["prompt"]) for r in adv]; Ya = [r["label"] for r in adv]
E = (a * D[:, None] @ v[None]).cuda(); W = down(model, l).weight
@torch.no_grad()
def lg(): return torch.cat([model(**tok(b, return_tensors="pt", padding=True).to(model.device), logits_to_keep=1).logits[:, -1].float().cpu() for b in batches(T[:32], 8)])
W0 = W.data.clone(); base = lg()
Dd, vd = D.cuda(), v.cuda()
h = down(model, l).register_forward_hook(lambda m, i, o: o + a * (i[0] @ vd)[..., None] * Dd); hook = lg(); s_hook = (rates(decide(model, tok, T), Y), rates(decide(model, tok, Ta), Ya)); h.remove()
def q8(x):
    m = x.abs().max(); return torch.round(x / m * 127) / 127 * m
res = {"effect_max": float((hook - base).abs().max())}
for name, f in {"float32": lambda x: x, "bfloat16": lambda x: x.bfloat16().float(), "fp16": lambda x: x.half().float(), "q8_simulated": q8, "q4_simulated": lambda x: torch.round(x / x.abs().max() * 7) / 7 * x.abs().max()}.items():
    W.data = f(W0 + E); lgs = lg(); s = rates(decide(model, tok, T), Y); sa = rates(decide(model, tok, Ta), Ya)
    W.data = f(W0); lgb = lg(); W.data = W0.clone()
    res[name] = {"max_diff_from_hook": float((lgs - hook).abs().max()), "mean_diff_from_hook": float((lgs - hook).abs().mean()), "p99_diff_from_hook": float((lgs - hook).abs().flatten().quantile(.99)), "update_kept_frac": float(((f(W0 + E) - f(W0)).abs() > 0).float().mean()), "base_quant_max_diff": float((lgb - base).abs().max()), "dev": s, "adv": sa}; print(name, res[name]["max_diff_from_hook"], s["tpr"], s["fpr"], sa["tpr"], sa["fpr"], flush=True)
res["hook_runtime"] = {"dev": s_hook[0], "adv": s_hook[1]}
h = down(model, l).register_forward_hook(lambda m, i, o: o + a * (i[0] @ vd).clamp(min=0)[..., None] * Dd); res["rectified"] = {"dev": rates(decide(model, tok, T), Y), "adv": rates(decide(model, tok, Ta), Ya)}; h.remove()
print("SIGN signed", s_hook[0]["tpr"], s_hook[0]["fpr"], s_hook[1]["tpr"], s_hook[1]["fpr"], "rect", res["rectified"]["dev"]["tpr"], res["rectified"]["dev"]["fpr"], res["rectified"]["adv"]["tpr"], res["rectified"]["adv"]["fpr"], flush=True)
save("E008", {"meta": provenance(NAME, exp="E008", layer=l, alpha=a), **res}); print("E008_DONE")
