# Research state (2026-11-09)

**Status:** ongoing. The first draft exists (`reports/final/draft.md`); the scaling phase and a second reading wave are still open. Nothing runs between sessions; no scheduler is configured.

**Done:**
- 30,453-paper harvest (arXiv, OpenAlex, Hugging Face Papers, Crossref); 17,363 triaged; 125 papers read with structured, quote-checked notes; partial citation graph.
- 19 hypotheses (H01-H19) with statuses; adversarial critique 01 applied; derivation T01.
- Critique 02 (15 findings) applied. E012 frozen transfer to SmolLM3-3B (fails), E013 hook controls (learned map causes the gain), E014 Qwen3-8B 8-bit (transfers, near ceiling).
- Experiments E001-E011 on Qwen2.5-0.5B/1.5B/3B and Qwen3-4B, all recorded in `results/interpretations/log.md`. E011 is the single use of the sealed test split.
- Reviewer reproduction: `repro/tables.py` (no packages, 1 s) and `repro/quick.py` (laptop CPU, ~14 min, 5.6 GB peak, measured from a fresh pinned install).

- E015: per-model dev selection on SmolLM3: probe AUROC 1.00 but model output .29; a labelled suppressive gated edit cuts test FPR .97 -> .18; label-free imitation is bounded by its teacher.
- E016 on hard2 (fresh, 160 items): RRM matches or beats its few-shot teacher at 4B (.45 vs .39 used) and 8B (.53 vs .45); no method handles the fact already being in the conversation history.

- E017: history-aware fitting on synthetic pairs: learned perfectly (probe 1.00, RRM at teacher) but does not transfer to natural in-context pairs (AUROC ~.5); E017b with natural fit pairs: probe .8 on hard2, yet no patch or prompt acts on it. Negative, reported as a limitation.

**Not done:** probe-gated router / history-position interventions for in_context; anything above 8B; second reading wave (~150 papers).

**Current understanding (measured, still small models):**
- On a capable model a short prompt beats every tested zero-token patch for a single behaviour (sealed test: rules prompt 93% end-to-end).
- A reduced-rank map fitted on unlabelled prompt pairs matches the few-shot prompt it imitates at zero prompt tokens (87% vs 83%, n=120, within noise) but needs a runtime hook.
- A prompt's effect at the decision token is state-dependent: constant shift R² ≈ 0, rank 4 ≈ 0.4.
- The conditional rank-one edit decides well but breaks the call it then writes (11% used).
- Many behaviours: prompts collapse as K grows; patches win at K ≤ 4 but a router or probe wins beyond about 16.
- E013: constant shift and random rank-4 maps of the same spectrum recover ~25% of the prompt's gain, the learned map ~100%; MMLU 0/400 changed.
- Recipes are model-specific: transfer to Qwen3-8B frozen, fail on SmolLM3-3B. Dev-only selection fixes SmolLM3's over-calling only with a labelled suppressive gate; the decision is linearly present (probe 1.00) but the model's output ignores it.
- Patches cannot replace the tool schema (0/90 valid calls without it).
- Conditional closed-form rank-one edits (SteerEdit) and prompt-to-weights ridge compilation (ICC) are prior art.
