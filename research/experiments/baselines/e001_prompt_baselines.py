"""E001: how much headroom do prompts leave? Dev split only; test stays sealed for the final comparison."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from lib import *
name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
model, tok = load(name)
out = {"meta": provenance(name, exp="E001"), "conds": {}}
for split in ["dev", "adversarial"]:
    d = read(split); y = [r["label"] for r in d]
    for cond in ["schema", "instruction", "fewshot"]:
        texts = [render_cond(tok, r["prompt"], cond) for r in d]
        s = decide(model, tok, texts)
        r_ = rates(s, y)
        # execution: does the emitted call name the tool and carry a sensible query?
        idx = [i for i, r in enumerate(d) if r["label"] == 1 and s[i] > 0]
        gen = generate(model, tok, [texts[i] for i in idx]) if idx else []
        ok = 0
        for i, g in zip(idx, gen):
            c = parse_call(g)
            q = str(((c or {}).get("arguments") or {}).get("query", "")).lower() if c and not c.get("malformed") else ""
            ok += bool(c and c.get("name") == "recall_memory" and any(t in q for t in d[i]["query_terms"]))
        r_["exec_ok_of_called"] = round(ok / max(len(idx), 1), 4)
        r_["n_tokens_prompt_overhead"] = len(tok(texts[0]).input_ids) - len(tok(render(tok, d[0]["prompt"])).input_ids)
        out["conds"][f"{split}/{cond}"] = r_
        print(split, cond, r_, flush=True)
save(f"E001_{name.split('/')[-1]}", out)
print("E001_DONE")
