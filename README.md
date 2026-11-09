# Training-free behaviour patches for LLMs: what works, what doesn't

> **Status: ongoing research.** A revised first draft exists (critique 02 applied; controls, a second family and an 8B check added 2026-11-08). A harder held-out split, per-model selection for other families and a second reading wave are open (`research/next_actions.md`). Negative results are reported as measured.

Can you add a behaviour to a pretrained LLM without gradient training, as a small patch instead of a prompt? The test behaviour is **deciding to call a memory tool** when an answer depends on an earlier conversation. Decision, execution and use of the result are scored separately.

The short answer, measured on Qwen2.5-1.5B and Qwen3-4B (details in `research/reports/final/draft.md`):

- **On a capable model, a short prompt beats every zero-token patch for a single behaviour.** A near-perfect result on the 1.5B model did not survive a text-classifier control, hard negatives and a stronger model.
- **Sealed test (Qwen3-4B):** a rank-4 map fitted from unlabelled prompt pairs matches the 119-token few-shot prompt it imitates at zero prompt tokens (87% vs 83% correct end-to-end, n=120, within noise). It still needs a runtime hook at the decision token.
- **The constant steering vector explains almost none of a prompt's effect; a low-rank function of the state explains much of it.** At the decision token, R² is about 0 for a constant shift against 0.40 at rank 4 (held out, no labels).
- **The gain is the learned map's, not the intervention's.** A constant shift or random rank-4 maps with the same spectrum recover a quarter of the prompt's gain; the learned map recovers all of it, with 0 of 400 MMLU answers changed (E013).
- **The recipes are model-specific.** Frozen, they transfer within the family to Qwen3-8B (8-bit), but fail on SmolLM3-3B, which over-calls the tool; only a prompt transferred there (E012, E014).
- **Selection per model works when the method fits the failure.** SmolLM3 encodes the decision (linear probe AUROC 1.00) but its output ignores it (AUROC .29). Prompt-imitating patches cannot beat a bad teacher; a labelled, gated rank-one edit chosen on dev cuts its false calls from 97% to 18% at unchanged recall (E015).
- **A closed-form LoRA from prompt pairs** (no labels, no gradient, no hook) recovers a third to a half of the prompt's end-to-end gain at zero prompt tokens (33% on the sealed test, 58% on other held-out sets), with no measurable MMLU change.
- **With many behaviours at once, prompts collapse** (64 rules: 1,154 tokens, 39× slower, worse than nothing). But beyond about 16 behaviours, a retrieval router or a linear probe beats the patches.
- **A patch cannot stand in for a tool schema:** 0 of 90 valid calls without one.
- Closed-form conditional edits and prompt-to-weights compilation are already published (SteerEdit 2604.12359, ICC Findings of EMNLP 2026). They are not claimed as new here, and the same primitive can implant backdoors.

## Run it

Three tiers, from instant to full. Nothing needs an API key or paid service.

| Tier | What | Hardware | Time |
|---|---|---|---|
| 0 | Rebuild every table from the shipped result files | any Python 3.10+, no packages | 1 s |
| 1 | Reproduce the core mechanisms on Qwen2.5-0.5B | laptop CPU, 8 GB RAM | ~15 min (measured: 852 s on 8 cores, 5.6 GB peak, under a hard 6 GB cap) |
| 2 | Full experiments on Qwen3-4B | one 16 GB GPU (e.g. a free Colab T4) | hours |

```bash
python repro/tables.py                                    # tier 0

python -m venv .venv && . .venv/bin/activate              # tier 1
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r repro/requirements.txt
python repro/quick.py                                     # ~15 min on 8 cores; --threads N; --skip-clinc avoids the dataset download
```

Tier 1 reproduces the mechanisms, not the exact headline numbers: those came from larger models.
- **Identity:** the weight edit equals the hook, to about 1e-5.
- **State dependence:** a constant shift explains about none of the prompt's effect, while a low-rank map explains much of it.
- **Words vs activations:** a text classifier is a strong control.
- **Decision table:** patch against prompt on a 0.5B model.
- **Capacity:** the noise-gain curve for many gates.

Tier 2 scripts are in `research/experiments/` (`E001`–`E010`, each documented at the top). They auto-detect the GPU.

## Layout

- `research.md`: the original assignment plus hardware corrections.
- `research/reports/final/draft.md`: the write-up.
- `research/results/interpretations/log.md`: every result as it was measured, including negatives.
- `research/decisions.md`: why each path was taken or dropped.
- `research/hypotheses/`: 19 hypotheses and an adversarial critique.
- `research/theory/`: derivations.
- `research/literature/`: `papers_index.jsonl.gz` (30,453 papers, metadata only), `triage.jsonl`, `reading_queue_index.jsonl.gz`, structured reading notes for 125 papers, and the citation graph. Abstracts and paper full texts are not redistributed; `research/tools/` re-fetches them from the original sources.
- `research/experiments/data/`: benchmark splits, written for this project. `test.jsonl` was sealed until E011 used it once with frozen methods; it is now spent, so future work needs a new held-out split.

## Licence

Code: MIT. Writing and the benchmark data: CC BY 4.0. Model weights and CLINC150 are downloaded from their original sources under their own licences.
