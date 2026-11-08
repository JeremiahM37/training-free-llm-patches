"""Closed-form rank-one patch: fit from forward passes only, apply as a real weight edit on down_proj."""
import time, torch
from lib import *

def down(model, l): return model.model.layers[l].mlp.down_proj

@torch.no_grad()
def collect(model, tok, texts, layers, want_all=False, bs=16, max_rows=None):
    """Per layer: key a and h_out at the last position; optionally a at every non-padding position (randomly capped at max_rows, seed 0)."""
    la, lh, aa = ({l: [] for l in layers} for _ in range(3)); cur = {}; hs = []
    for l in layers:
        hs.append(down(model, l).register_forward_pre_hook(lambda m, a, l=l: cur.__setitem__(("a", l), a[0].float())))
        hs.append(model.model.layers[l].register_forward_hook(lambda m, i, o, l=l: cur.__setitem__(("h", l), (o[0] if isinstance(o, tuple) else o).float())))
    for b in batches(texts, bs):
        enc = tok(b, return_tensors="pt", padding=True, pad_to_multiple_of=PAD).to(model.device); model(**enc, logits_to_keep=1)
        for l in layers:
            # copy=True: on CPU, .cpu() returns the same storage, so a sliced view would keep the whole batch alive
            la[l].append(cur["a", l][:, -1].to("cpu", copy=True)); lh[l].append(cur["h", l][:, -1].to("cpu", copy=True))
            if want_all: aa[l].append(cur["a", l][enc.attention_mask.bool()].cpu())
    for h in hs: h.remove()
    pack = lambda d: {l: torch.cat(v) for l, v in d.items() if v}
    aa = pack(aa)
    if max_rows:
        aa = {l: x[torch.randperm(len(x), generator=torch.Generator().manual_seed(0))[:max_rows]] if len(x) > max_rows else x for l, x in aa.items()}
    return pack(la), pack(lh), aa

def background_texts(model, tok, schema_texts, n_gen=40):
    gens = generate(model, tok, schema_texts[:n_gen], max_new=32)
    return list(schema_texts) + [s + g.replace("<|im_end|>", "").replace("<|endoftext|>", "") for s, g in zip(schema_texts[:n_gen], gens)]

def fit_gate(X, Apos, Aneg, kind="lda", lam_rel=1e-2, cap=6000, seed=0):
    """X: background keys (n x m), uncentered. Solved in sample space by Woodbury, CPU double."""
    X = X.double(); g = torch.Generator().manual_seed(seed)
    if X.shape[0] > cap: X = X[torch.randperm(X.shape[0], generator=g)[:cap]]
    n = X.shape[0]; mp, mn = Apos.double().mean(0), Aneg.double().mean(0)
    lam = lam_rel * X.pow(2).sum(1).mean() / X.shape[1]
    L = torch.linalg.cholesky(X @ X.T / n + lam * torch.eye(n, dtype=X.dtype))
    Rinv = lambda b: (b - X.T @ torch.cholesky_solve(((X @ b) / n)[:, None], L)[:, 0]) / lam
    if kind == "lcmv":
        M = torch.stack([mp, mn], 1); RM = torch.stack([Rinv(mp), Rinv(mn)], 1)
        v = RM @ torch.linalg.solve(M.T @ RM, torch.tensor([1., 0.], dtype=X.dtype))
    else:
        v = Rinv(mp - mn); v = v / (v @ mp)
    return v.float()

class Edit:
    """Context manager: W' = W + alpha D v' on layer l's down_proj, restored exactly on exit."""
    def __init__(self, model, l, D, v, alpha): self.w, self.u = down(model, l).weight, alpha * D.to(model.device).float()[:, None] @ v.to(model.device).float()[None]
    def __enter__(self): self.old = self.w.data.clone(); self.w.data = (self.w.data.float() + self.u).to(self.old.dtype)
    def __exit__(self, *a): self.w.data = self.old
