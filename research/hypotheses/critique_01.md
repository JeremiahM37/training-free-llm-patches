# Critique 01: adversarial review of H01, H03, H05, H08, H16 and the experiment code

Scope: `hypotheses/hypothesis_registry.md`, `theory/mathematical_derivations/T01_rank_one_gate.md`, `results/interpretations/log.md`, `experiments/evaluations/e003_try_to_break.py`, `experiments/evaluations/e006_many_behaviours.py`. Also read, because the claims depend on them: `experiments/patchlib.py`, `experiments/lib.py`, `experiments/prototypes/e002_rank_one_gate.py`, `experiments/evaluations/e005_prior_rank_one.py`, `e007_self_compile.py`, `e009_no_schema.py`, `experiments/data/*.jsonl`, `results/metrics/*.json`.

Checks run: numerical toy checks (Woodbury form, zero-forcing noise gain, centering), a lexical-cue and duplicate scan of the data files, and audits of the saved JSON against the log. No experiment was run and no file other than this one was changed.

Evidence labels: **[measured]** = I ran it or read it from a saved file; **[code]** = read from source; **[speculative]** = plausible from the code, not checked.

---

## Ranking, most damaging first

1. **H01's headline result is a dev-selected maximum on a set a text classifier solves, and the frozen config loses to that classifier on the one set it was not selected on.** (E002, E003)
2. **H05 cannot separate tokenless patches from a linear probe with a fixed logit bias, and E006 has a negative-set bug and an unbalanced router baseline.** (E006, H05, H11, H06)
3. **T01's noise-gain formula is wrong (offset off by two, divergence point wrong), and the code breaks the "exactly zero crosstalk" identity it claims to test.** (T01 §3, E006 line 61)
4. **H16's falsifier is built on a value vector that already contains the tool schema.** (E009)
5. **H08's selection is circular, its "compile" of the reduced-rank map is a forward hook rather than a weight edit, and its support number has no artifact.** (E007, H08)
6. **H03 has no measurement, and the code cannot falsify its prediction.** (E006, H03)
7. **Prior art covers most of H01's mechanism, H07 is essentially In-Context Compilation, and H08 is partly covered by Doc-to-LoRA.**
8. **Experiment artefacts disagree with the scripts: stale JSON, a bf16-quantised edit, an underpowered capability check.**
9. **Speculative items.**

---

## 1. H01: the headline is selection plus a cue confound

**[measured] Selection breadth.** `results/metrics/E002_Qwen2.5-1.5B-Instruct.json` has **720** rows, 144 per condition (`uncond`, `lastpos`, `cast`, `lcmv`, `lda`). The log says "360 configs" (`log.md` line 17), which is wrong. Of the 144 `lda` rows, **one** has J = 1.00 and three have J ≥ 0.90. The best row is chosen by `max(...J)` on dev (`e002_rank_one_gate.py` lines 131 and 135). A J of 1.00 on 60 + 60 examples, taken as the maximum of about 720 configurations, is what winner's-curse selection produces. The registry still lists it as H01 support ("M E002, E003").

**[code] Dev is not held out.** `e002_rank_one_gate.py` line 92 labels the dev gate AUROC "held out", but dev is the selection set. E003 then reports dev as a confirmation set (`log.md` lines 36–37).

**[measured] Dev and contrast are text-separable.** The E003 JSON gives TF-IDF AUROC .999 on dev and .987 on contrast, while length alone gives .61 and .54. I also scanned the data for first-person or past-reference cues (`my|I|I'd|I'm|me|told you|said|we`):

| split | positives with cue | negatives with cue |
|---|---|---|
| train | 0.82 | 0.38 |
| dev | 0.62 | 0.33 |
| adversarial | 0.72 | 0.27 |
| contrast | 0.88 | **0.05** |
| style2 | 0.90 | 0.80 |

The contrast negatives are almost cue-free, so contrast is close to a cue detector. Style2 is the only cue-balanced set, which is why it matters. It is written by the PI rather than Haiku, and per §8 it has **never been scored** on any patch.

**[measured] The frozen config does not beat the text baseline on adversarial, the one set E002 did not select on.** E002 reads adversarial (`e002_rank_one_gate.py` lines 15 and 74) but only for the base rate. From the E003 JSON:

| adversarial | AUROC | J (TPR − FPR) |
|---|---|---|
| TF-IDF LR | **.802** | .43 |
| patch L26 (frozen) | .746 | .43 |
| patch L22 | .858 | .46 |
| best prompt | .677 | .31 |

The layer choice flips the comparison with the text baseline. The log's "What survives" (`log.md` lines 50–53) says the patch beats the prompts, which holds, but it says nothing about TF-IDF on adversarial. L22's J of .46 versus .43 is within noise at n = 50 and 70.

**[measured] The gate is two-sided, so the FPR drop is not only a trigger gate.** E002's JSON gives `mean_neg = -1.279` for the L26 LDA gate. Patchlib normalises the gain to 1 on positives (`patchlib.py` line 39), so negatives receive about −1.28·α·D. The edit pushes negatives away from `<tool_call>`. T01 §2 says a good gate responds 0 on background, and this gate does not. Some of the FPR reduction therefore comes from a signed push on negatives, which is a bidirectional steering vector. H10 is meant to test this, but E008 is not in the tree.

**[measured] Prior art already covers the mechanism** (arXiv ids checked in `literature/papers.jsonl` and `reading_notes/_all.json`):
- 2609.09054 (TFTV): closed-form rank-one weight edit from a steering direction and the SVD of W. E005 line 20 implements the same right factor, `sign(Vᵀμ)·σ`.
- 2602.09870 (Steer2Edit): rank-one component edit with right factor proportional to Wᵀs.
- 2409.05907 (CAST): condition-vector gate with a threshold.
- 2605.03907 (PSR): a ReLU probe on the hidden state scales a steering vector, trained by gradient. The registry concedes this form.
- 2406.11717 (Arditi et al.): rank-one orthogonalisation of a refusal direction, which is the weight form of a steering vector.

The novelty claim is narrow. It is the decoupling of trigger contrast from steering direction. E005 is the test for that and has not been run.

**Verdict:** the claim that DKE beats prompts and steering is supported on one 1.5B model and one dev-selected config. The claim that it beats a text classifier is not supported on adversarial.

---

## 2. H05: the patch reduces to a linear readout, and E006 has design and code problems

**[code] The patch is a gated logit bias, so there is no per-word logit-bias baseline.** E006 uses the output-embedding rows of each code word as the value directions (`e006_many_behaviours.py` lines 66–69, `U = lm_head row / norm`, scaled by a fixed 0.1·typical norm). Adding `g_k(a)·u_w` to the residual at layer l raises that word's logit through the logit-lens path. At the top layer, `LS = [nl-1, nl-4]` (line 18), this is a linear readout of `a` with a fixed logit map, which is H11. E006 includes a linear probe (lines 124–134) but **no per-word gated logit bias**, which is the natural H06 baseline. So a win over `prompt_all` cannot be attributed to the gate rather than a probe plus a bias.

**[code] Negative-set bug.** Line 77 builds `others` from non-chosen intents in index order, two each. Line 78 is `neg = (others + source[OOS])[:max(100, len(pos))]`. For K < 150, `others` has at least 100 items, so the first 100 are taken and **OOS is never included**. For K = 150, the negatives are OOS only. So each K is scored on a different negative population, J is not comparable across K, the hardest negative class (out-of-scope) is absent from every K < 150 row, and the negatives are the first ~50 intents by index rather than a random sample.

**[code] The router baseline is not what its comment says.** Line 101 says "with an abstain threshold chosen on validation". Lines 105–108 have no threshold: abstain is simply argmax over a class −1 that is trained on OOS examples (line 103). So the router gets 50 labelled OOS negatives, and the patch never sees any, since its negatives are background (line 45). That asymmetry favours the router, and the comment misdescribes the baseline.

**[code] Background contains trigger-class data.** Line 45 draws `TR[i][10:12]` for every intent in `ALL`, including the chosen ones. The gate's R, and therefore the "collateral" energy, is estimated partly on the trigger distribution.

**[measured] No result exists.** There is no E006 file in `results/metrics/`, and the log has no E005–E009 entry. H05 is marked `active` with no measurement. Its prediction is untested.

**[speculative] Router strength.** On CLINC150, TF-IDF over 10 examples per intent may be strong because the intents are lexically distinct. If so, `prompt_rag` or `router_only` wins and H05 reduces to retrieval, as its own failure condition says. This cannot be checked without running E006.

---

## 3. T01: the noise-gain formula is wrong, and the code does not implement the identity

**What is right.** The Woodbury form in T01 §2 is correct. I checked it numerically against `patchlib.py` lines 33–34 (max error 1e-14) and the LCMV solve (constraint residual 4e-16). The reduction `vᵢᵀRvᵢ = [(MᵀR⁻¹M)⁻¹]ᵢᵢ` in §3 is also correct.

**What is wrong: the expectation.** For iid isotropic whitened means, `G` is r × K and `GᵀG` is Wishart(r, I_K). The inverse-Wishart mean gives E[(GᵀG)⁻¹] = I / (r − K − 1). So the noise gain is **r/(r − K − 1)**, not r/(r − K + 1) as written in T01 §3 (line 38). The divergence point is K = r − 1, not K = r + 1, and "diverges as K → r" in P1 (line 40) is also off.

**[measured] numpy check**, r = 60, 400 draws, normalised by ‖μ̃‖²:

| K | measured mean | r/(r−K−1) (correct) | r/(r−K+1) (T01) |
|---|---|---|---|
| 20 | 1.539 | 1.538 | 1.463 |
| 40 | 3.151 | 3.158 | 2.857 |
| 55 | 15.31 | 15.00 | 10.00 |

At K = 59 the median is 58.9, against T01's 30. Heavy tails make the mean noisy near the divergence, which is the regime P1 cares about.

**[code + measured] Centering breaks the "exactly zero" claim.** T01 §3 says μⱼᵀvᵢ = δᵢⱼ with uncentered trigger means, so mean crosstalk is zero by construction. `e006_many_behaviours.py` line 61 instead builds `M` from **centred** class means (`a[lab==i].mean(0) - mu_all`), so the constraint holds on centred means. The uncentred response has a common offset vᵢᵀμ_all on every input. A toy check with a shared offset gave uncentred diagonals of 0.87–1.07 and off-diagonals of about ±0.13, not zero. A gate with no bias turns that offset into always-on background firing. How large it is on real activations is **[speculative]**, because it was not measured.

**[code] P1 cannot be falsified by E006 as written.** The predicted curve is never computed. `gate_energy_mean` is recorded (line 123) but not compared with r/(r−K−1) or any normalisation by ‖μ̃‖². The participation ratio is computed from the first 6000 rows of an uncentred R (line 49), which is not a random subsample. An uncentred R from a shared-template corpus has a dominant mean eigenvalue, so this PR is not the r of the iid formula. **[speculative]** on magnitude; the mechanism is clear from the code.

**Minor notation.** T01 §1 writes a(h) = σ(W_g h) ⊙ (W_u h) with h the block input. Qwen2's MLP uses SiLU on an RMSNorm of the post-attention residual. The identity survives because only `a` as computed matters, but the formula as written is not the model's.

---

## 4. H16: the falsifier is confounded by the schema

**[code] The value vector contains the schema.** `e009_no_schema.py` line 7 defines `P = render_cond(tok, p, "fewshot")`. `render` in `lib.py` lines 29–31 passes `tools=[TOOL]`, so the prompted activations include the `recall_memory` definition. Line 12 then fits `D = mean(Hp − H)` against bare-prompt activations `H`. So D carries the instruction, the examples and the schema, and the tool name is stored in D by construction.

The experiment cannot discriminate. "Patch-only yields correct calls" is the success condition the registry calls "a larger positive result", and this design produces it whether or not H16 is true. The fix is to fit D from instruction-only prompts with no tool definition, or to hold the instruction context fixed while varying only the schema.

**[code] In-sample reporting.** Line 16 picks (layer, alpha) by dev J, and line 18 scores names and queries on `dev + adv`, which includes dev.

**[code] Lenient criterion.** Line 25 counts a query as correct if any `query_terms` substring appears in it. Terms like "vet", "dog" and "gift" are short (see §8 and the data).

---

## 5. H08: selection is circular, the "compile" is a hook, and the support figure is unsupported

**[code] Agreement filter.** `e007_self_compile.py` line 34 keeps only utterances whose prompted decision equals the generator's intent. Line 42 draws the holdout from that same set. The layer/alpha selection (line 61) therefore measures agreement with the teacher on items the teacher already gets right. The filter also removes the boundary cases that E003 and E004 show matter, such as can't-answer negatives, before fitting. The "zero labels" claim depends on the teacher acting as an oracle for which utterances to keep.

**[code] Label noise from the generator.** Lines 24–26 keep any line of 12–200 characters from the generator's output, including preambles such as "Here are 25 messages".

**[code] Mechanism mismatch with H07.** H07 says the reduced-rank map is applied "in weights" (registry). Lines 79–81 apply it as a forward hook on the residual stream with a full d × d matrix B. It is not a rank-one edit of `down_proj`, and it is not a LoRA.

**Credit.** Selection in E007 uses only the self-generated holdout (lines 42–65), and dev is reported separately. That part is clean. The holdout itself is circular, as above.

**[measured] Unsupported support figure.** The registry says "M Qwen3-4B few-shot prompt is accurate (dev AUROC .993)". The log says the Qwen3 E001 rerun is pending (`log.md` line 68), and `results/metrics/` has no Qwen3 file. In this tree, .993 appears only as a 1.5B TF-IDF value in E004 (writing kind). I cannot rule out a file on the desktop, but nothing here supports the figure.

**Prior art.**
- **doi:10.18653/2026.findings-emnlp.983 (In-Context Compilation):** encodes a context-induced behavioural shift as a sparse closed-form edit to attention output-projection weights, calibrated from no-context and with-context activation differences, with no gradient. This is H07 with a different low-rank fit, so H07's novelty should be `known`.
- **2602.15902 (Doc-to-LoRA):** a hypernetwork that internalises a context into LoRA weights instantly.
- **2506.06105 (Text-to-LoRA):** generates adapters from a task description.
- **2410.00487 (SELF-PARAM):** distils a context into weights with KL.
H08's "no labels, no gradient" is the only new part. The label-free part is weakened by the agreement filter.

---

## 6. H03: no measurement, and the prediction is weak as stated

- No result exists (see §2 and §3). The registry's support is "D only".
- The prediction "gate_energy flat for K ≪ r" is nearly built into the shape `1 + K/r` for small K, so it is weak. The informative part is the divergence near K ≈ r − 1. The code prints the participation ratios (line 51) but never reports r for the layers used, and K goes up to 150, so the divergence regime may not be reached at all.
- The gate energy is measured on `bg[l][:3000]` (line 123). The solver fits a random 6000-row subset of the same background (line 54), so the energy is partly in-sample. Energy on fitted rows is biased low. P1 needs held-out background.
- H03 cites 2505.11611, 2610.03199 and 2608.09490 as the nearest work. These are about SAE features and merges. The closer prior art for closed-form zero-forcing or null-space constraints on edits is 2604.12359 (compiling steering into weights under null-space constraints), 2410.02355 (AlphaEdit, null-space-constrained editing) and 2306.03819 (LEACE, closed-form whitened concept erasure). The registry does not cite any of these, so "open" is not established.

---

## 7. Prior art summary (arXiv or DOI ids checked in the literature files)

| id | what it is | bearing |
|---|---|---|
| 2406.11717 | refusal direction, rank-one weight orthogonalisation | H01 form |
| 2409.05907 | CAST, conditional steering | H01 gate |
| 2605.03907 | PSR, learned ReLU probe times steering vector | H01 gate, H07 |
| 2609.09054 | TFTV, closed-form rank-one weight edit | H01 baseline, E005 |
| 2602.09870 | Steer2Edit | H01 baseline, E005 |
| 2604.12359 | compiling steering into weights under null-space constraints | H01, H03 |
| 2410.02355 | AlphaEdit, null-space constrained editing | H03 |
| 2306.03819 | LEACE, closed-form whitened erasure | H03, T01 |
| 2202.05262 | ROME | T01 (cited there) |
| doi:10.18653/2026.findings-emnlp.983 | In-Context Compilation, sparse closed-form edit | H07, H08 |
| 2602.15902 | Doc-to-LoRA | H07, H08 |
| 2506.06105 | Text-to-LoRA | H08 |
| 2410.00487 | SELF-PARAM | H08 |
| 2602.07276 | Steer2Adapt, composing steering vectors | H05, H13 |
| 2610.04283 | First-Order Steering, composing steering via weight updates | H05, H13 |
| 2607.22837 | logit bias as black-box adaptation | H06, H05 |
| 2601.04600 | limits of rank-one editing for multi-hop | H01 limits |
| 2312.06681 | CAA steering | background |

---

## 8. Experiment artefacts and reproducibility

- **[measured] E003 JSON predates the current script.** `results/metrics/E003_Qwen2.5-1.5B-Instruct.json` (17:48) has no `style2` and no `logit_bias` key. The script (modified 18:00) includes style2 in `SPLITS` (line 11) and computes the logit bias (lines 67–74). So the H06 logit-bias baseline and the style2 test have **never been run**, while the log describes them as added (D007). The reported E003 table cannot be reproduced from the current script.
- **[code] Quantised weight edit.** `patchlib.py` lines 44–45 add αDvᵀ to the bf16 weight and round back to bf16. Any update smaller than half a bf16 ulp of the weight is erased, so the "real weight edit" is a quantised update, not Δvᵀ. The log's "reported numbers use the real weight edit" (`log.md` line 55) therefore reports the quantised version. This is a result-changing artefact. It is the likeliest source of the 0.75-logit gap, but that is **[speculative]**, because the identity check (E003 lines 47–54) compares the **maximum** absolute logit difference over the full vocabulary, an extreme-value statistic, on one example. The check that would settle it is the same comparison in float32, reporting the mean and 99th percentile.
- **[code] In-sample reporting in E003.** `e003_try_to_break.py` line 66 picks the best prompt on dev. Line 69 picks the logit-bias threshold on dev, and line 73 then reports dev rates from that same sample.
- **[code] E002 mislabel.** Line 92 says "held out" for dev, which is the selection set (lines 131 and 135).
- **[measured] Config count.** 720 rows, not 360 (see §1).
- **[measured] Capability claim.** E003 MMLU accuracy is .605 with the patch and .6075 without (n = 400). An unpaired 95% half-width at accuracy 0.6 with n = 400 is about ±0.05, so the test cannot see drops of several points. "Preserved" should read "no drop detected". A paired test would be tighter, but none was run.
- **[measured] No leakage found.** There are 0 exact duplicate prompts across the splits. The sealed `test.jsonl` is not read by any script, so there is no final held-out result at all. The registry's "held-out" claims are dev or CV only. The one test-answer string that also occurs in train is trivial (1 of 240).
- **[code] E005 is not scale-matched.** The TFTV alpha grid (`e005_prior_rank_one.py` line 22) is not normalised to DKE's typical-norm scale, so the best alpha may sit at a grid edge. The docstring admits the single-module restriction. **[speculative]** until E005 runs.

---

## 9. Speculative or minor

- CLINC150 and MMLU are public, so Qwen may have seen them. Base first-token accuracy on MMLU is only a preservation check, so this matters less there. **[speculative]**
- The H10 sign mechanism (§1) and E008 are not in the tree.
- The log's "360 configs" and the E002 "held out" wording are documentation errors, not result errors.

---

## What would change my mind

- **H01:** a held-out result on style2 (cue-balanced, never seen by any patch) with the same frozen config beating TF-IDF, plus a clamped-negative ablation (H10) showing the gain is not from the signed push.
- **H05:** E006 with a per-word gated logit-bias baseline, a fixed negative set that includes OOS at every K, and the router threshold as the comment describes. The patch must beat the probe and the bias, not only the prompts.
- **T01:** a numerical check of r/(r−K−1) on real background activations, and the centred versus uncentred gate responses reported side by side.
- **H16:** E009 with the value fitted from instruction-only prompts.
- **H08:** holdout selection on examples the teacher gets wrong as well as right, and a weight-space version of the reduced-rank map.
