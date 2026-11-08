# Critique 02 — adversarial review of the first draft (2026-11-08)

Reviewer: a Haiku 5.5 subagent, read-only, checking every number against `results/metrics/` and the log. The PI verified each finding against the files before acting.

| # | Finding | Verified | Disposition |
|---|---|---|---|
| 1 | Table 2 0.5B row did not match the final CPU run; its JSON lived in gitignored `repro/out/` | yes (row predates the memory-fix rerun) | row corrected; reference output shipped as `metrics/T1_cpu_quick_Qwen2.5-0.5B-Instruct.json` |
| 2 | 0.5B TF-IDF ".98 dev" is pooled 5-fold CV, not dev | yes | relabelled |
| 3 | bf16 "0.75 logits" vs E008 .014 not reconciled | yes, two different measurements (whole bf16 forward vs weight rounding only) | both stated with sources |
| 4 | Fit times and patch sizes in §6 wrong | yes (E003: 1.44–1.48 s, 3.8 s, 20,992 B) | corrected |
| 5 | "No MMLU change" is absence of evidence | yes | "no drop detected; a few points not resolvable" |
| 6 | RRM "matches on contrast" is a 12-point gap; CIs missing | yes (.75 vs .87 TPR) | corrected; CIs added to Table 1 |
| 7 | "Compiled without labels" hides that the teacher prompt is hand-written; hook cost omitted | yes | reworded; hook cost listed as unmeasured |
| 8 | TF-IDF is a baseline, not a control; "every split" false for 1.5B; data provenance contradictory | yes | reworded |
| 9 | 0/90 without schema generalises from one construction | yes | scoped |
| 10 | Many-behaviour ranking overstated; K=150 row omitted; "39×" not derivable | yes (table gives 19× at K=64) | scoped, row added, ratio recomputed |
| 11 | "Tie" between activations and words has no margin and unpenalised layer selection | yes | "no detected difference" |
| 12 | "Used" is positives-only and its grading undefined | yes | grading defined under Table 1 |
| 13 | State-dependence → mechanism is inferred; constant shift's R² ≈ 0 by construction; "two families" wrong | yes | reworded; second family added (E012) |
| 14 | TFTV/Steer2Edit at grid-edge α | yes | E013 re-selects on a wider dev grid |
| 15 | No control attributing the RRM gain to the learned map | yes | E013: constant-shift, random rank-4 (3 seeds), prefill-only, MMLU under the hook |

Found by the PI while writing E013 (not raised by the reviewer): the E011 RRM hook edits the last position of **every forward call**, so during generation it also acts on every generated token. The draft described it as a decision-token intervention. Corrected in the text; E013 measures the prefill-only variant.
