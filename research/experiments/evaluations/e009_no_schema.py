"""E009 (H16): take away the tool schema. Can a patch alone produce a valid recall_memory call? Value fit from schema+instruction versus bare prompt."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
NAME = sys.argv[1]; model, tok = load(NAME); nl = model.config.num_hidden_layers; LAYERS = sorted({int(nl * .7), int(nl * .8), int(nl * .9), nl - 2}); ALPHAS = [.25, .5, 1, 2, 4]
def bare(p): return tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYS}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
tr = read("train"); y = torch.tensor([r["label"] for r in tr]); ptr = [r["prompt"] for r in tr]
B = [bare(p) for p in ptr]
NAMELESS = "When information from previous interactions with this user would materially improve your response, emit a tool call to look it up before answering. Do not do so when the message itself and general knowledge are enough."
def nameless(p): return tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYS + " " + NAMELESS}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ARMS = {"value_has_schema_and_name": [render_cond(tok, p, "fewshot") for p in ptr], "value_no_schema_no_name": [nameless(p) for p in ptr]}
_, _, bg = collect(model, tok, background_texts(model, tok, B), LAYERS, want_all=True); A, H, _ = collect(model, tok, B, LAYERS)
HP = {k: collect(model, tok, v, LAYERS)[1] for k, v in ARMS.items()}
dev, adv = read("dev"), read("adversarial"); Td = [bare(r["prompt"]) for r in dev]; Yd = [r["label"] for r in dev]
res = {"base_noschema": rates(decide(model, tok, Td), Yd), "arms": {}}
pos = [r for r in adv if r["label"] == 1] + read("style2")[:40]; txt = [bare(r["prompt"]) for r in pos]   # scored away from the selection set
for arm, Hp in HP.items():
    best = None; rows = []
    for l in LAYERS:
        v = fit_gate(bg[l], A[l][y == 1], A[l][y == 0]); d = (Hp[l][y == 1] - H[l][y == 1]).mean(0); D = d / d.norm() * H[l].norm(dim=1).mean()
        for a_ in ALPHAS:
            with Edit(model, l, D, v, a_): r = rates(decide(model, tok, Td), Yd)
            r.update(layer=l, alpha=a_, J=round(r["tpr"] - r["fpr"], 4)); rows.append(r)
            if best is None or r["J"] > best["J"]: best = {**r, "D": D, "v": v}
    with Edit(model, best["layer"], best["D"], best["v"], best["alpha"]):
        s_ = decide(model, tok, txt); g = generate(model, tok, txt, max_new=48)
    names = {}; ok = named = 0; samples = []; called_text = 0
    for r, x, t in zip(pos, s_, g):
        called_text += "<tool_call>" in t
        c = parse_call(t)
        if not c and x <= 0: continue
        n = (c or {}).get("name") if c and not c.get("malformed") else ("<malformed>" if c else "<none>"); names[str(n)] = names.get(str(n), 0) + 1; named += n == "recall_memory"
        q = str(((c or {}).get("arguments") or {}).get("query", "")).lower() if c and not c.get("malformed") else ""; ok += n == "recall_memory" and any(w in q for w in r["query_terms"])
        if len(samples) < 10: samples.append(t[:150])
    res["arms"][arm] = {"best": {k: best[k] for k in ("layer", "alpha", "tpr", "fpr", "J")}, "n_pos": len(pos), "first_token_call": int((s_ > 0).sum()), "text_has_call": called_text, "correct_name": int(named), "correct_name_and_query": int(ok), "name_counts": names, "samples": samples, "rows": rows}
    print("RESULT", arm, res["arms"][arm]["best"], "calls", int((s_ > 0).sum()), "of", len(pos), "name_ok", named, "full_ok", ok, sorted(names.items(), key=lambda kv: -kv[1])[:6], samples[:3], flush=True)
save(f"E009_{NAME.split('/')[-1]}", {"meta": provenance(NAME, exp="E009", layers=LAYERS, alphas=ALPHAS, nameless=NAMELESS), **res}); print("E009_DONE")
