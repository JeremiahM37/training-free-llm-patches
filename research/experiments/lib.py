"""Shared harness. Decision, execution and use are measured separately, as research.md section 8 requires."""
import json, os, platform, subprocess, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE.parent / "results" / "metrics"
# Pad batches to a multiple of PAD tokens: on CPU every distinct sequence length allocates new kernel buffers, and this bounds them. Left padding keeps the decision token last.
PAD = int(os.environ.get("PAD_MULTIPLE", 32))

TOOL = {"type": "function", "function": {"name": "recall_memory",
        "description": "Search notes saved from previous conversations with this user.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "What to look up"}}, "required": ["query"]}}}
DEFAULT_SYS = "You are a helpful assistant."
INSTRUCTION = ("When information from previous interactions with this user would materially improve your response, "
               "you do not have it in this conversation: call recall_memory with a short query before answering. "
               "Do not call it when the message itself and general knowledge are enough.")
SHOTS = [("Which hotel did I say I preferred in Lisbon?", True), ("What is the capital of Portugal?", False),
         ("Book it with my usual seat preference.", True), ("My seat preference is aisle. Write a note to the airline saying so.", False)]

def load(name="Qwen/Qwen2.5-1.5B-Instruct", dtype=None):
    """GPU in bf16 if one is present, otherwise CPU in float32 (bf16 matmuls are slow or missing on many CPUs)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # bf16 where supported; fp16 on older GPUs such as the Colab T4; float32 on CPU
    dtype = dtype or ((torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dev == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    kw = {}
    if os.environ.get("LOAD_8BIT") and dev == "cuda":
        # Phase D on a 16 GB GPU: LLM.int8 everywhere except the down_proj of the layers a recipe edits (KEEP_BF16_LAYERS), so weight edits stay exact
        from transformers import BitsAndBytesConfig
        keep = [f"model.layers.{l}.mlp.down_proj" for l in os.environ.get("KEEP_BF16_LAYERS", "").split(",") if l]
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"] + keep)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map={"": dev}, **kw).eval()
    model.requires_grad_(False)
    return model, tok

def read(split): return [json.loads(l) for l in (DATA / f"{split}.jsonl").open()]

def render(tok, prompt, system=DEFAULT_SYS, extra=()):
    msgs = [{"role": "system", "content": system}, *extra, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False)

def render_cond(tok, prompt, cond, extra=()):
    if cond == "schema": return render(tok, prompt, extra=extra)
    if cond == "instruction": return render(tok, prompt, DEFAULT_SYS + " " + INSTRUCTION, extra)
    if cond == "fewshot":
        sys = DEFAULT_SYS + " " + INSTRUCTION + "\n\nExamples:\n" + "\n".join(
            f"User: {q}\nAction: {'call recall_memory' if m else 'answer directly'}" for q, m in SHOTS)
        return render(tok, prompt, sys, extra)
    raise ValueError(cond)

def batches(xs, n):
    for i in range(0, len(xs), n): yield xs[i:i + n]

@torch.no_grad()
def decide(model, tok, texts, bs=int(os.environ.get("DECIDE_BS", 32))):
    """Returns first-token log-odds of <tool_call> versus the best other token, per prompt."""
    tc = tok.convert_tokens_to_ids("<tool_call>")
    out = []
    for b in batches(texts, bs):
        enc = tok(b, return_tensors="pt", padding=True, pad_to_multiple_of=PAD).to(model.device)
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        t = lg[:, tc].clone(); lg[:, tc] = -1e9
        out.append((t - lg.max(-1).values).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def generate(model, tok, texts, max_new=48, bs=int(os.environ.get("GEN_BS", 16))):
    res = []
    for b in batches(texts, bs):
        enc = tok(b, return_tensors="pt", padding=True, pad_to_multiple_of=PAD).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        res += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=False)
    return res

def parse_call(text):
    if "<tool_call>" not in text: return None
    body = text.split("<tool_call>", 1)[1].split("</tool_call>")[0].strip()
    try:
        d = json.loads(body)
        return d if isinstance(d, dict) else None
    except Exception:
        return {"malformed": True}

def followup(tok, prompt, q, call_text, record, system=DEFAULT_SYS, extra=()):
    """Render the turn after a tool call. Uses the template's tool_calls support; if the template silently drops tool_calls
    (SmolLM3 does), the assistant turn carries the model's own emitted call text and the result is wrapped in <tool_response>."""
    msgs = [{"role": "system", "content": system}, *extra, {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "recall_memory", "arguments": {"query": q}}}]},
            {"role": "tool", "content": record}]
    s = tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if q in s: return s
    i = len(msgs) - 2
    msgs[i] = {"role": "assistant", "content": call_text.split("</tool_call>")[0].strip() + "\n</tool_call>"}
    msgs[i + 1] = {"role": "tool", "content": f"<tool_response>\n{record}\n</tool_response>"}
    return tok.apply_chat_template(msgs, tools=[TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False)

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** .5
    return (round((c - r) / d, 4), round((c + r) / d, 4))

def auroc(pos, neg):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def rates(scores, labels):
    s, y = np.asarray(scores), np.asarray(labels)
    p, n = s[y == 1], s[y == 0]
    tp, fp = int((p > 0).sum()), int((n > 0).sum())
    return {"tpr": round(tp / max(len(p), 1), 4), "tpr_ci": wilson(tp, len(p)), "fpr": round(fp / max(len(n), 1), 4),
            "fpr_ci": wilson(fp, len(n)), "auroc": round(auroc(p, n), 4) if len(p) and len(n) else None, "n_pos": len(p), "n_neg": len(n)}

def provenance(model_name, **cfg):
    def sh(c):
        try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception: return ""
    from huggingface_hub import model_info
    try: rev = model_info(model_name).sha
    except Exception: rev = "unknown"
    return {"time": time.strftime("%FT%T"), "model": model_name, "revision": rev, "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
            "data_sha": sh(f"cat {DATA}/*.jsonl | sha256sum | cut -c1-16"), "config": cfg}

def save(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
