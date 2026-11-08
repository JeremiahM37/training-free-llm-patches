"""E003: attempt to break the E002 winner. Config is frozen from E002 (no reselection on these sets). Test split stays sealed.
A text-only classifier control; contrast pairs; can't-answer negatives; real weight edit; execution; use; capability; sample complexity."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from patchlib import *
import random
NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
CONFIGS = eval(sys.argv[2]) if len(sys.argv) > 2 else [("lda", 26, 0.25), ("lda", 22, 0.5), ("lcmv", 22, 0.5)]
VALUE = {c[:3]: c[3] for c in CONFIGS if len(c) > 3}; CONFIGS = [c[:3] for c in CONFIGS]
SEED = 0; torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
model, tok = load(NAME)
tr = read("train"); ytr = np.array([r["label"] for r in tr])
SPLITS = {k: read(k) for k in ("dev", "adversarial", "contrast", "style2", "cantanswer")}
out = {"meta": provenance(NAME, exp="E003", configs=CONFIGS, seed=SEED)}

# Text-only control: can a bag of words or length do this? If so the dataset is confounded.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
wv, cv = TfidfVectorizer(ngram_range=(1, 2)), TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
ptr = [r["prompt"] for r in tr]; feats = lambda ps: hstack([wv.transform(ps), cv.transform(ps)])
wv.fit(ptr); cv.fit(ptr); clf = LogisticRegression(C=100, max_iter=2000).fit(feats(ptr), ytr)
ctl = {}
for k, d in SPLITS.items():
    if k == "cantanswer": continue
    y = np.array([r["label"] for r in d]); ps = [r["prompt"] for r in d]
    s = clf.decision_function(feats(ps)); L = np.array([len(p.split()) for p in ps])
    ctl[k] = {"tfidf_lr": rates(s, y), "length_auroc": round(auroc(-L[y == 1], -L[y == 0]), 4)}
out["text_control"] = ctl; print("TEXT_CONTROL", {k: (v["tfidf_lr"]["auroc"], v["tfidf_lr"]["tpr"], v["tfidf_lr"]["fpr"], v["length_auroc"]) for k, v in ctl.items()}, flush=True)

# Fit
LAYERS = sorted({c[1] for c in CONFIGS})
schema = [render_cond(tok, r["prompt"], "schema") for r in tr]; instr = [render_cond(tok, r["prompt"], "instruction") for r in tr]
t0 = time.time()
_, _, bg = collect(model, tok, background_texts(model, tok, schema), LAYERS, want_all=True)
A, H, _ = collect(model, tok, schema, LAYERS); _, Hi, _ = collect(model, tok, instr, LAYERS)
t_fp = time.time() - t0
patches = {}
for kind, l, a in CONFIGS:
    t1 = time.time()
    v = fit_gate(bg[l], A[l][ytr == 1], A[l][ytr == 0], kind)
    dH = (Hi[l][ytr == 1] - H[l][ytr == 1]).mean(0); dL = H[l][ytr == 1].mean(0) - H[l][ytr == 0].mean(0)
    dv = dL if VALUE.get((kind, l, a)) == "dL" else dH; D = dv / dv.norm() * H[l].norm(dim=1).mean()
    patches[(kind, l, a)] = (D, v); key = f"{kind}/L{l}/a{a}"
    out.setdefault("fit", {})[key] = {"forward_pass_s": round(t_fp, 2), "solve_s": round(time.time() - t1, 2), "bytes_fp16": 2 * (D.numel() + v.numel())}
print("FIT", out["fit"], flush=True)

# Hook and weight edit must agree: T01 section 1 is an identity
kind, l, a = CONFIGS[0]; D, v = patches[CONFIGS[0]]
enc = tok(render_cond(tok, SPLITS["dev"][0]["prompt"], "schema"), return_tensors="pt").to(model.device)
with torch.no_grad():
    cap = {}
    h1 = down(model, l).register_forward_hook(lambda m, i, o: o + a * (i[0].float() @ v.cuda())[..., None].to(o.dtype) * D.cuda().to(o.dtype))
    lg_hook = model(**enc).logits.float(); h1.remove()
    with Edit(model, l, D, v, a): lg_edit = model(**enc).logits.float()
    lg_base = model(**enc).logits.float()
out["identity"] = {"max_abs_logit_diff_hook_vs_edit": float((lg_hook - lg_edit).abs().max()), "max_abs_effect": float((lg_hook - lg_base).abs().max())}
print("IDENTITY", out["identity"], flush=True)

# Prompt baselines, stronger set, picked on dev
PROMPTS = {"instruction": INSTRUCTION,
 "rules": "Rules for recall_memory. Call it only if answering requires a personal fact from an EARLIER conversation with this user that is not in the current message (their preferences, past plans, things they told you). Never call it for general knowledge, math, code, writing, or when the message already contains the needed facts. If you call it, output only the tool call.",
 "decide": "Before replying, ask yourself: does this message depend on something this user said in an earlier conversation? If yes, your entire reply must be a recall_memory tool call. If no, answer normally and do not use any tool.",
 "shots8": INSTRUCTION + "\nExamples.\n'Which hotel did I say I preferred?' -> tool call\n'What is the capital of Portugal?' -> answer\n'Book my usual seat.' -> tool call\n'My seat preference is aisle; write a note saying so.' -> answer\n'What was that recipe from last month?' -> tool call\n'What is a memory leak?' -> answer\n'Send it to my sister's new address.' -> tool call\n'Last week I quit my job; draft a letter about it.' -> answer"}
def prompted(p, name): return render(tok, p, DEFAULT_SYS + " " + PROMPTS[name])
dv = SPLITS["dev"]; ydv = [r["label"] for r in dv]; pr = {}
for n in PROMPTS:
    pr[n] = rates(decide(model, tok, [prompted(r["prompt"], n) for r in dv]), ydv); print("PROMPT", n, pr[n]["tpr"], pr[n]["fpr"], pr[n]["auroc"], flush=True)
best_prompt = max(pr, key=lambda n: pr[n]["tpr"] - pr[n]["fpr"]); out["prompts_dev"] = pr; out["best_prompt"] = best_prompt
# One-number baseline: a constant bias on the <tool_call> logit, threshold picked on dev. Decision only.
s0 = decide(model, tok, [render(tok, r["prompt"]) for r in dv]); ydn = np.array(ydv)
tau = max(sorted(s0), key=lambda t: (s0[ydn == 1] > t).mean() - (s0[ydn == 0] > t).mean())
lb = {"tau": float(tau)}
for k, d in SPLITS.items():
    s_ = decide(model, tok, [render(tok, r["prompt"]) for r in d]) - tau; yy = np.array([r["label"] for r in d])
    lb[k] = {"fpr": round(float((s_ > 0).mean()), 4)} if k == "cantanswer" else rates(s_, yy)
out["logit_bias"] = lb; print("LOGIT_BIAS", json.dumps(lb), flush=True)
COND = {"base": (lambda p: render(tok, p), None), f"prompt:{best_prompt}": (lambda p: prompted(p, best_prompt), None)}
for c in CONFIGS: COND[f"{c[0]}/L{c[1]}/a{c[2]}"] = ((lambda p: render(tok, p)), c)

class Ctx:
    def __init__(self, c): self.c = c
    def __enter__(self):
        if self.c: self.e = Edit(model, self.c[1], *patches[self.c], self.c[2]); self.e.__enter__()
    def __exit__(self, *a):
        if self.c: self.e.__exit__()

def record(prompt, q): return [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "recall_memory", "arguments": {"query": q}}}]}]
TAIL = lambda t: t.lower()
res = {}
for cname, (rf, c) in COND.items():
    res[cname] = {}
    with Ctx(c):
        for k, d in SPLITS.items():
            texts = [rf(r["prompt"]) for r in d]; s = decide(model, tok, texts); y = np.array([r["label"] for r in d])
            if k == "cantanswer":
                res[cname][k] = {"fpr": round(float((s > 0).mean()), 4), "by_kind": {kk: round(float(np.mean([s[i] > 0 for i, r in enumerate(d) if r["kind"] == kk])), 3) for kk in sorted({r["kind"] for r in d})}}
            else:
                res[cname][k] = rates(s, y)
                if k == "contrast":                      # both sides of a pair right
                    byp = {}
                    for i, r in enumerate(d): byp.setdefault(r["pair"], {})[r["label"]] = s[i] > 0
                    res[cname][k]["pair_both_right"] = round(float(np.mean([v.get(1, False) and not v.get(0, True) for v in byp.values()])), 4)
                # Execution and use over all positives, end to end. Failing to call counts as failing.
                pos = [i for i, r in enumerate(d) if r["label"] == 1]; called = [i for i in pos if s[i] > 0]
                ok_exec = ok_use = recall_again = 0
                gens = generate(model, tok, [texts[i] for i in called], max_new=48) if called else []
                follow, idxs = [], []
                for i, g in zip(called, gens):
                    pc = parse_call(g); r = d[i]
                    q = str(((pc or {}).get("arguments") or {}).get("query", "")) if pc and not pc.get("malformed") else ""
                    if pc and pc.get("name") == "recall_memory" and q:
                        ok_exec += any(t in q.lower() for t in r["query_terms"])
                        sysmsg = DEFAULT_SYS if not cname.startswith("prompt") else DEFAULT_SYS + " " + PROMPTS[best_prompt]
                        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": r["prompt"]}, *record(r["prompt"], q), {"role": "tool", "content": r["memory_record"]}]
                        follow.append(tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False)); idxs.append(i)
                if follow:
                    ans = generate(model, tok, follow, max_new=64)
                    for i, a_ in zip(idxs, ans):
                        recall_again += "<tool_call>" in a_
                        ok_use += d[i]["answer_substr"].lower() in a_.lower() and "<tool_call>" not in a_
                res[cname][k].update(exec_of_pos=round(ok_exec / max(len(pos), 1), 4), e2e_use_of_pos=round(ok_use / max(len(pos), 1), 4), use_given_call=round(ok_use / max(len(follow), 1), 4), recalled_after_result=recall_again)
    print("COND", cname, json.dumps(res[cname]), flush=True)
out["splits"] = res

# Capability preservation: MMLU, with and without the tool schema
from datasets import load_dataset
mm = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=SEED).select(range(400))
def mq(r): return r["question"] + "\n" + "\n".join(f"{c}. {x}" for c, x in zip("ABCD", r["choices"])) + "\nAnswer with the letter only."
ids = [tok(c, add_special_tokens=False).input_ids[0] for c in "ABCD"]; tc = tok.convert_tokens_to_ids("<tool_call>")
@torch.no_grad()
def mmlu(texts):
    acc, call, lps = 0, 0, []
    for bi, b in enumerate(batches(texts, 16)):
        enc = tok(b, return_tensors="pt", padding=True).to(model.device); lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for j in range(len(b)):
            acc += int(lg[j, ids].argmax()) == mm[bi * 16 + j]["answer"]; call += int(lg[j].argmax()) == tc
        lps.append(lg.log_softmax(-1).half().cpu())
    return acc / len(texts), call / len(texts), torch.cat(lps)
noschema = [tok.apply_chat_template([{"role": "system", "content": DEFAULT_SYS}, {"role": "user", "content": mq(r)}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in mm]
withschema = [render(tok, mq(r)) for r in mm]
cap = {}; gen_texts = noschema[:40]
base_nos = mmlu(noschema); base_sch = mmlu(withschema); base_gen = generate(model, tok, gen_texts, max_new=48)
for cname, (_, c) in COND.items():
    if c is None and cname != "base":
        a_, cl, _ = mmlu([render(tok, mq(r), DEFAULT_SYS + " " + PROMPTS[best_prompt]) for r in mm]); cap[cname] = {"acc_schema": a_, "call_schema": cl}; continue
    with Ctx(c):
        an, cn, ln = mmlu(noschema); asc, cs, _ = mmlu(withschema); g = generate(model, tok, gen_texts, max_new=48)
    kl = float((base_nos[2].exp() * (base_nos[2] - ln)).sum(-1).mean())
    cap[cname] = {"acc_noschema": an, "call_noschema": cn, "acc_schema": asc, "call_schema": cs, "first_token_KL_noschema": round(kl, 5), "gen_identical_frac": sum(x == y for x, y in zip(g, base_gen)) / len(g)}
    print("CAP", cname, cap[cname], flush=True)
out["capability"] = cap

# Sample complexity: fit on n per class, repeated, at the frozen best config
kind, l, a = CONFIGS[0]; curve = {}
di = SPLITS["dev"]; advd = SPLITS["adversarial"]; can = SPLITS["cantanswer"]; dt = [render(tok, r["prompt"]) for r in di]; at = [render(tok, r["prompt"]) for r in advd]; ct = [render(tok, r["prompt"]) for r in can]
ydi, yad = [r["label"] for r in di], [r["label"] for r in advd]
for n in (2, 5, 10, 20, 40):
    runs = []
    for seed in range(1 if n == 40 else 8):
        rng = np.random.default_rng(seed); pi = rng.choice(np.where(ytr == 1)[0], n, replace=False); ni = rng.choice(np.where(ytr == 0)[0], n, replace=False)
        v_ = fit_gate(bg[l], A[l][pi], A[l][ni], kind); dH_ = (Hi[l][pi] - H[l][pi]).mean(0); D_ = dH_ / dH_.norm() * H[l].norm(dim=1).mean()
        with Edit(model, l, D_, v_, a):
            r1 = rates(decide(model, tok, dt), ydi); r2 = rates(decide(model, tok, at), yad); f3 = float((decide(model, tok, ct) > 0).mean())
        runs.append([r1["tpr"], r1["fpr"], r2["tpr"], r2["fpr"], f3])
    curve[n] = {"mean": np.mean(runs, 0).round(3).tolist(), "std": np.std(runs, 0).round(3).tolist(), "cols": ["dev_tpr", "dev_fpr", "adv_tpr", "adv_fpr", "cant_fpr"]}
    print("CURVE", n, curve[n]["mean"], curve[n]["std"], flush=True)
out["curve"] = curve
save(f"E003_{NAME.split('/')[-1]}", out); print("E003_DONE")
