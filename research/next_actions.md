# Next actions (updated 2026-11-09)

In order of value per cost:
1. **History:** E018 probe router works on Qwen3-4B (n=20). Replicated on Qwen3-8B (done). Next: a second natural in_context test set (n is small); check it does not hurt no-history items (route only when history exists, which is trivially true now).
2. **Done (E015).** Per-model selection on SmolLM3 (dev-only sweep of layer, rank, λ, α): E012 shows frozen recipes fail there; the open question is whether selection fixes it or the model's decision is simply not linearly accessible at the first token.
3. Second reading wave: ~150 papers from `literature/reading_queue_index.jsonl.gz` and the citation-graph forward list (no hardware).
4. RRM hook latency measurement; CF-LoRA vs RRM gap (hook-free distillation is the obvious next method).
5. Fix the E005 grid-edge alphas — done in E013 (same α, interior); drop.

Running experiments remotely: `research/experiments/run_resume.sh` skips finished work; set `GPU_HOST`. 8-bit Phase D: `LOAD_8BIT=1 KEEP_BF16_LAYERS=21,25,34 DECIDE_BS=8 GEN_BS=8` on a 16 GB GPU.
Literature tools: the shipped index has no abstracts; rerun `tools/harvest.py`, `dedup.py`, then `triage.py` (needs a local Ollama with `nomic-embed-text`, `OLLAMA_URL`).
