# Next actions (updated 2026-11-08)

In order of value per cost:
1. Second critic pass on `reports/final/draft.md` (no hardware).
2. Second reading wave: ~150 papers from `literature/reading_queue_index.jsonl.gz` and the citation-graph forward list (no hardware).
3. Second model family: rerun the frozen E011 methods on a non-Qwen model around 3B (one 16 GB GPU, a few hours). Note the sealed test is spent; use a freshly written held-out split.
4. Phase D: one 7B model in 4- or 8-bit on a 16 GB GPU.
5. Fix the E005 grid-edge alphas (critique section 8).

Running experiments remotely: `research/experiments/run_resume.sh` skips finished work; set `GPU_HOST` to the GPU machine's ssh name.
Literature tools: the shipped index has no abstracts; rerun `tools/harvest.py`, `dedup.py`, then `triage.py` (needs a local Ollama with `nomic-embed-text`, `OLLAMA_URL`) to rebuild them.
