"""Tier 0: rebuild the paper's main tables from the shipped result files. Needs only the Python standard library; runs in a second."""
import json
from pathlib import Path
M = Path(__file__).resolve().parents[1] / "research" / "results" / "metrics"
J = lambda f: json.loads((M / f).read_text())
def tf(r): return f"{r['tpr']:.2f}/{r['fpr']:.2f}" if "tpr" in r else f"-/{r['fpr']:.2f}"
q = "Qwen3-4B-Instruct-2507"; e3 = J(f"E003_{q}.json"); e5 = J(f"E005_{q}.json"); e7 = J(f"E007_{q}.json"); e10 = J(f"E010_{q}_fewshot.json")
S = ("dev", "adversarial", "contrast", "style2", "cantanswer")
print(f"Table 1. Single behaviour on {q}, TPR/FPR of calling recall_memory (dev is the tuning set)\n")
print(f"{'method':44s}" + "".join(f"{s:>13s}" for s in S))
rows = [("no patch", e3["splits"]["base"]), (f"best prompt ({e3['best_prompt']})", e3["splits"][f"prompt:{e3['best_prompt']}"]), ("few-shot prompt (teacher)", e10["teacher"]), ("logit bias, 1 number", e3["logit_bias"]),
        ("TFTV rank-one (2609.09054), our reimpl.", e5["tops"]["tftv"]), ("Steer2Edit (2602.09870), our reimpl.", e5["tops"]["steer2edit"])]
rows += [(f"rank-one gated patch {k}", v) for k, v in e3["splits"].items() if k.startswith(("lda", "lcmv"))]
rows += [(f"closed-form LoRA, no labels, rank {r}", e10["patches"][r]) for r in ("4", "16")]
rows += [("reduced-rank map, hook, rank 4, L34", e7["D_reduced_rank"]["L34"]["rank4"])]
for n, r in rows: print(f"{n:44s}" + "".join(f"{tf(r[s]):>13s}" for s in S))
print("\nTable 2. State dependence of the prompt effect at the decision token (held-out R^2), E007")
for l, r in e7["D_reduced_rank"].items(): print(f"  {l}: constant shift {r['const_shift_r2_on_test']:.2f}, rank1 {r['rank1_r2']:.2f}, rank4 {r['rank4_r2']:.2f}, rank16 {r['rank16_r2']:.2f}")
print("\nTable 3. End to end on 150 held-out positives: called / executed / used the answer, E010")
for n, r in [("no patch", e10["e2e_base"]), ("few-shot prompt", e10["e2e_teacher"])] + [(f"closed-form LoRA rank {k}", e10["patches"][k]["e2e"]) for k in ("1", "4", "16", "64")]:
    print(f"  {n:28s} {r['called']:.2f} / {r['exec']:.2f} / {r['use']:.2f}")
print("  MMLU (n=400, no schema) under the rank-16 LoRA:", e10["patches"]["16"]["mmlu_noschema"])
e6 = J(f"E006_{q}.json"); print("\nTable 4. K behaviours at once on CLINC150 (J = correct - false positive rate; tokens added; seconds per prompt), E006")
for r in e6["rows"]:
    if r["seed"]: continue
    best = max((k for k in r if k.startswith("patch_")), key=lambda k: r[k]["J"])
    print(f"  K={r['K']:3d}  prompt_all J {r['prompt_all']['J']:+.2f} ({r['prompt_all']['added_tokens']} tok, {r['prompt_all']['s_per_prompt']:.3f}s)  router+1 rule J {r['prompt_rag']['J']:+.2f}  best patch J {r[best]['J']:+.2f} ({best})  linear probe J {r[[k for k in r if k.startswith('probe')][0]]['J']:+.2f}")
e9 = J(f"E009_{q}.json"); print("\nTable 5. Without the tool schema, valid recall_memory calls (E009):", {k: f"{v['correct_name']}/{v['n_pos']}" for k, v in e9["arms"].items()})
e4 = J(f"E004_{q}.json"); print("Table 6. Words vs activations, 5-fold AUROC (E004):", {"tfidf": e4["cv"]["tfidf"], "best gate": e4["cv"]["layers"][f"L{e4['cv']['best_layer']}/lda"]["all"], "diff CI": e4["cv"]["auroc_diff_best_minus_tfidf"]["ci95"]})

e11 = J("E011_sealed_test.json"); print("\nTable 0 (headline). Sealed test, single use, Qwen3-4B: TPR/FPR, AUROC, called/executed/used of 120 positives (E011)")
for n, r in e11["methods"].items():
    print(f"  {n:58s} {r['tpr']:.2f}/{r['fpr']:.2f}  {r['auroc']:.3f}" + (f"   {r['called']:.2f}/{r['executed']:.2f}/{r['used']:.2f}" if "used" in r else ""))
