# Results log (newest last)

## E001, 2026-11-03: prompt baselines, Qwen2.5-1.5B, dev (60/60) and adversarial (50/70)
Measured. Decision = first token is `<tool_call>`.
| split / condition | TPR | FPR | AUROC | prompt overhead |
|---|---|---|---|---|
| dev / schema only | 0.22 | 0.47 | 0.38 | 0 |
| dev / + instruction | 0.75 | 0.82 | 0.45 | 47 tok |
| dev / + few-shot | 0.57 | 0.52 | 0.55 | 119 tok |
| adv / schema only | 0.32 | 0.39 | 0.57 | 0 |
| adv / + instruction | 0.86 | 0.57 | 0.59 | 47 tok |
| adv / + few-shot | 0.68 | 0.34 | 0.74 | 119 tok |

Reading: on this model the prompts move the call *rate* and barely discriminate. Limits: one small model and only three prompts, so this is not yet evidence about prompts in general. Stronger prompts and larger models must be tried before claiming a patch beats prompting. The execution metric depends on `query_terms` matching and may be strict.

## E002, 2026-11-03: closed-form gated edits, Qwen2.5-1.5B, fit on train (80), best of grid on dev (60/60)
Measured, best config per method. **Selected on the same dev set it is scored on (720 configs, 144 per method; corrected from 360), so these are optimistic.**
| method | layer, value, alpha | TPR | FPR | J |
|---|---|---|---|---|
| no patch | | 0.22 | 0.47 | -0.25 |
| instruction prompt (E001) | | 0.75 | 0.82 | -0.07 |
| unconditional steer | 18, dH, 1 | 0.52 | 0.13 | 0.38 |
| decision-token-only steer | 22, dH, 1.5 | 0.75 | 0.42 | 0.33 |
| CAST-style cosine gate | 14, dH, 0.5 | 0.98 | 0.37 | 0.62 |
| rank-one, LCMV gate (T01) | 22, dH, 0.5 | 0.92 | 0.23 | 0.68 |
| rank-one, ridge LDA gate | 26, dH, 0.25 | 1.00 | 0.00 | 1.00 |

Findings:
- **My T01 zero-mean constraint lost to the simpler LDA gate.** LDA's response on negatives is strongly *negative* (mean -1.28 at layer 26), so it actively pushes a trigger-happy model away from calling, which a zero-mean gate cannot do. Negative result for the LCMV constraint on this task.
- Suspicions not yet ruled out: (a) gate at layer 26 may read "the model is about to say it can't answer", not "memory is needed"; (b) dataset may be separable by trivial features; (c) selection bias; (d) capability damage unmeasured; (e) decision only, not execution or use.

## E003, 2026-11-03: attempt to break E002 (config frozen, Qwen2.5-1.5B, real weight edit, 80 fit examples)
Measured. Patch = rank-one LDA, layer 26, alpha 0.25, 20,992 bytes at fp16; solve 1.5 s plus 3.8 s of forward passes.
| set | metric | no patch | best of 4 prompts | patch L26 | patch L22 | TF-IDF text classifier |
|---|---|---|---|---|---|---|
| dev | TPR / FPR | .22 / .47 | .92 / .80 | .98 / .05 | .93 / .12 | .98 / .03 |
| adversarial | TPR / FPR | .32 / .39 | .94 / .63 | .72 / .29 | .60 / .14 | .76 / .33 |
| adversarial | AUROC | .57 | .68 | .75 | .86 | .80 |
| contrast pairs | both sides right | .23 | .13 | .80 | .80 | not computed (AUROC .99) |
| can't-answer negatives | FPR | **.19** | .61 | **.81** | .61 | n/a |
| dev | calls, then uses answer (of all pos) | .07 | .70 | .75 | .72 | n/a |
| adversarial | same | .12 | .54 | .32 | .36 | n/a |
| MMLU, n=400, no schema | accuracy | .6075 | n/a | .605 | .59 | n/a |

**What this overturns:**
1. **The dev and contrast sets are nearly text-separable** (TF-IDF AUROC .999 and .987; length alone only .61 and .54). The E002 1.00/0.00 is mostly an easy dataset. Most likely cue: first-person possessives with no supplied fact.
2. **The gate does not mean "needs memory".** It fires on 81% of unanswerable-for-other-reasons requests against a base of 19%, including 95% of realtime questions. It is closer to "cannot answer from what I have". Prompts also inflate this (61%), so the confusion is not unique to the patch, but the patch is worse here.
3. **End-to-end on adversarial, the prompt wins** (.54 versus .32) by calling far more often, at a false-positive cost.

**What survives:**
- Better discrimination than any of the 4 prompts tried on dev, adversarial and contrast at a fraction of the false positives, on this one small model.
- No capability drop detected on MMLU at L26 (.605 versus .6075, first-token KL .002); with n=400 and an unpaired half-width of about ±.05, this cannot see a drop of a few points. Greedy text changed on 35% of 40 no-schema MMLU prompts, so "preserved" is not "identical".
- Saturates fast: 10 examples per class match 40 (dev TPR .975 vs .983; adv TPR .67 vs .72, within about one std). Improvement then stops, so what limits it is bias rather than sample count.

**Weight-edit precision:** hook and bf16 weight edit differ by up to 0.75 logits against a 5.9 effect. The identity in T01 is exact in real arithmetic, so the likely cause is bf16 rounding of a small update to large weights (untested). Reported numbers use the real bf16-rounded weight edit, not the exact update. Consequence if confirmed: merged edits on low-precision weights lose fidelity and a patch should stay a separate LoRA at runtime.

## E004, 2026-11-03: activations versus words, Qwen2.5-1.5B, pooled hard data (150 pos, 270 neg), 5-fold CV with pairs grouped
Measured.
- TF-IDF logistic regression AUROC **.980** (hard-only .983). Best activation gate (L21 LDA) **.930** (hard-only .950). Difference -.050, 95% CI [-.076, -.028] by group bootstrap, even though the layer was chosen on the same data, which favours the gate.
- LDA again beats the LCMV constraint at most layers. The T01 constraint is now rejected for this use.
- Leave-one-negative-kind-out: gate is better than TF-IDF on some unseen kinds (past_public .93-.98 vs .91; self_contained_past .93-.94 vs .89; unknowable .97-.99 vs .94) and much worse on others (capability .55-.64 vs .90; missing_input .35-.68 vs .73).
- Behaviour with hard negatives in the fit, out of fold, real edit: L21 alpha .25 gives TPR .67, FPR .13, can't-answer FPR **.12** (down from .81 in E003). So the can't-answer confusion is largely fixable by including such negatives. This cost TPR, and train-split TPR is only .35, so the patch is not uniformly good.

Caveat: all data came from one author (Haiku), so the words may carry author-style cues. TF-IDF's win may not transfer. Needs a different-author set.

## E001 on larger models, 2026-11-03
- Qwen2.5-3B: schema dev AUROC .36, adversarial .21 (worse than chance). Few-shot .83 and .63. Bigger does not automatically mean better.
- Qwen3-4B schema only, dev: **TPR .50, FPR .00, AUROC .971**, exec 1.00 of called. Remaining conditions hit VRAM OOM (fixed by keeping only last-position logits); rerun pending.
- Implication: on a stronger model, the discrimination is already present and it is mainly under-calling. A one-number logit bias is now a required baseline.

## Qwen3-4B, 2026-11-03 (E001-E005, E007-E009). Measured unless marked.
Numbers are TPR / FPR. All splits except dev are outside selection, but dev itself is the selection set for E002, E003, E005.

| method (tokens added) | dev | adversarial | contrast | style2 | can't-answer FPR |
|---|---|---|---|---|---|
| schema only (0) | .50/.00 | .04/.17 | .13/.02 | .50/.03 | .00 |
| logit bias, 1 number, tau on dev (0) | .97/.17 | .80/.33 | .80/.22 | 1.0/.55 | .28 |
| few-shot prompt (119) | .92/.02 | .68/.09 | .87/.03 | .93/.10 | .02 |
| rules prompt | .93/.08 | .86/.17 | .80/.15 | .98/.30 | .08 |
| TFTV, single module, L34 (0) | .88/.00 | .50/.23 | .47/.03 | .85/.15 | .12 |
| Steer2Edit, single component, L25 (0) | .97/.05 | .82/.26 | .87/.33 | .95/.28 | .64 |
| DKE rank-one, L26, a.5, dH (0) | .58/.00 | .24/.13 | .47/.02 | .80/.00 | .00 |
| Reduced-rank map, rank 4, L34, hook at decision token, no labels (0) | .92/.00 | .48/.13 | .75/.02 | .98/.08 | .03 |
| Text classifier (TF-IDF), decision only | .98/.03 | .76/.33 | 1.0/.18 | .85/.13 | n/a |

Caveats: TFTV alpha .05 and Steer2Edit alpha -32 sit at grid edges (the critic predicted this), so these baselines may be understated. E003's L34 row used dH rather than the dL selected in E002, a bug in the config hand-off: fixed for the rerun. E003's capability stage hit an OOM on Qwen3-4B: rerun pending. Reimplementations are single-module, not the papers' full configurations.

**Findings**
1. **On a capable model, a prompt wins for one behaviour.** Few-shot J: .90, .59, .83, .83. No patch beats that across sets. H05 single-behaviour regime closed.
2. **Gating adds little here.** In E002, unconditional steer J .88, decision-token .90, cosine gate .95, LCMV .90, LDA .90, all at dev selection. The model already discriminates, so a value vector amplifies a decision it has already formed. H01's prediction "beats TFTV and Steer2Edit" is not supported overall; DKE has lower FPR but much lower TPR, and a higher alpha was not selected.
3. **A logit bias is not enough** (H06 rejected): it inflates FPR on style2 (.55) and adversarial (.33) versus few-shot (.10, .09).
4. **The prompt effect is state-dependent.** At the decision token, a constant shift explains R² -.02 of prompted-minus-unprompted activations on held-out utterances. Rank 1, 4, 16 linear maps explain .19-.21, .37-.40, .50-.51. No labels and no gradient. This is the clearest new measurement so far and may explain why steering falls short of prompts.
5. **Reduced-rank map matches few-shot on several sets at 0 tokens** (dev, style2, contrast within noise), loses on adversarial (.48/.13 vs .68/.09). It is a position-aware hook, not a weight edit.
6. **Activations catch up with words on a stronger model** (E004): gate L21 .973 vs TF-IDF .980, difference -.007, CI [-.024, .010]. So H17 is not rejected on Qwen3-4B. Unseen kinds: gate wins capability, missing_input, refusal, unknowable; loses general, math, code, writing.
7. **Self-compile as implemented fails** (H08, E007): holdout selection tied at J .889 and picked the smallest alpha, leaving behaviour nearly unchanged (dev .53/.00, adversarial .04/.17). Model-written utterances are too easy: the unpatched model already separates them, so they cannot calibrate strength. Discrimination still rose (style2 AUROC .906 vs .878).
8. **Agreement between generator intent and prompted decision was 98.5%,** which again says the generated data is easy.
9. **A patch cannot replace the schema** (H16 supported, E009): with the value containing the schema it reaches first-token calls but degenerates into repeating `<tool_call>`, 0 of 90 valid names. With no schema and no name in the value, 0 calls. One value construction, one model.
10. **Precision** (H12, E008, 1.5B): in float32, hook and edit agree to 2e-5, confirming the identity. bf16 rounding of the weight alone gives .014, fp16 .002, simulated Q8 .52, simulated Q4 10.2 (but that also destroys the unpatched model). So the .75 gap in E003 came mostly from bf16 inference arithmetic, not weight rounding, contrary to my earlier guess. Simulation only.
11. **Signed edge** (H10, E008, 1.5B): signed gate dev .93/.00, adv .54/.21; clamped .97/.47, .62/.39. Most of the FPR gain on that trigger-happy model comes from pushing negatives away, not from a quiet trigger.

Process note: a shell bug in my wait loop sat idle from about 18:50 to 22:05. No data was affected; three hours were lost.

## E006, E010, E011 (Qwen3-4B; results produced 2026-11-03 late, read 2026-11-07)

**E011 sealed test, single use** (120 pos / 120 neg, test domains). Every method frozen beforehand. The test domains turned out easier than adversarial or style2: no patch already reaches AUROC .97.
| method (prompt tokens added) | TPR/FPR | AUROC | called / executed / used, of 120 |
|---|---|---|---|
| no patch (0) | .61/.00 | .970 | .62/.62/.56 |
| few-shot prompt (119) | .89/.00 | .989 | .89/.88/.83 |
| rules prompt (~70) | .99/.10 | .996 | .99/.99/.93 |
| TFTV (0) | .88/.05 | .974 | .89/.89/.83 |
| Steer2Edit (0) | .95/.08 | .988 | .95/.95/.88 |
| gated rank-one, L34 (0) | .88/.00 | .991 | .98/.48/**.11** |
| closed-form LoRA, no labels, r16 (0) | .71/.00 | .984 | .71/.71/.65 |
| reduced-rank map at decision token, hook, no labels, r4 (0) | .94/.00 | .988 | .94/.94/.87 |
| TF-IDF classifier (decision only) | .89/.05 | | n/a |

Findings: the zero-token reduced-rank hook matches or beats the 119-token few-shot prompt it imitates (.87 vs .83 used; FPR .00 both). With n=120 the difference is within noise; 'matches' is the defensible claim. The gated rank-one edit gets the decision right but breaks execution: it stays active while the call is generated, and only 11% end with a correct answer. Unconditional published edits (TFTV, Steer2Edit) execute fine. The closed-form LoRA without a hook gains less here (+.09 used).

**E010 closed-form LoRA (no hook, no labels), Qwen3-4B, non-test sets:** rank 16 at L21: dev .78/.00, adversarial .24/.13, contrast .65/.02, style2 .83/.00, can't-answer .00; end-to-end used .49 vs no patch .18 and few-shot prompt .71, on 150 held-out positives. MMLU 400: .7175 → .71 (4 worse, 1 better, McNemar p .375), and greedy generations on 30 MMLU prompts stay identical (rank 16: 30/30). Held-out R² from MLP keys: rank 1 .03, rank 4 .52, rank 16 .67, rank 64 .74.

**E006 K behaviours at once, CLINC150, Qwen3-4B, test utterances.** J = correct − false-positive rate. Best patch per K is chosen post hoc among 6 variants (optimistic).
| K | all rules in prompt (tokens, s/prompt) | router + 1 rule | best patch | linear probe |
|---|---|---|---|---|
| 1 | .30-.59 (18, .007) | .00-.80 | .80-.99 | .60-1.0 |
| 4 | .39-.81 (75, .012) | .32-.47 | .57-.81 | .41-.77 |
| 16 | .14-.35 (291, .030) | .35-.52 | .43-.48 | .56-.68 |
| 64 | -.05 (1154, .132) | .46 | .31 | .56-.62 |
| 150 | -.17 (2719, .392) | .19 | .18 | -.29 (no rejection class) |

Findings: prompts degrade fast with K (false positives from unrelated rules) and cost 39-115x per prompt at K=64-150. Patches are best at K≤4, then fall behind the linear probe on the same keys. The probe's gate is fine, so the loss is in the value path (H13 supported). The router is limited by 10 examples per intent.
**H03 rejected as stated:** the participation ratio of the background keys (3.5 at L35, 30-34 at L32) is far below K=140, yet the noise gain at K=140 is only 1.6-2.2. Real class means are not iid in an r-dimensional space, so the formula's r is not the participation ratio. K=150 is singular by construction (150 centred means are linearly dependent), an artefact, not a capacity limit.

## 2026-11-08

**Critique 02** (`hypotheses/critique_02.md`): 15 findings, all verified, draft revised. PI-found correction: the E011 RRM hook acted on every generated token, not only the decision token.

**E012 frozen-recipe transfer, SmolLM3-3B (HF, not Qwen-derived), test split, decision only (end-to-end "used" was mis-scored and is being rerun).** Every method keeps its Qwen3-4B configuration; layers map by relative depth (36 → 36, so identical indices). TPR / FPR / AUROC:
| method | TPR | FPR | AUROC |
|---|---|---|---|
| no patch | .74 | .97 | .33 |
| few-shot prompt | .34 | .52 | .34 |
| rules prompt | .88 | 1.00 | .38 |
| TFTV (L34, α .05) | .00 | .00 | .19 |
| Steer2Edit (L25, α −32) | .28 | .10 | .55 |
| gated rank-one (L34, α 1) | .11 | .03 | .59 |
| CF-LoRA r16 L21 | .49 | .75 | .40 |
| RRM r4 L34 hook | .25 | .66 | .23 |
| TF-IDF (same split) | .89 | .05 | |
SmolLM3 calls the tool on nearly every negative and its first-token score ranks the classes backwards; neither prompt fixes it. No frozen recipe transfers: they either shut calling off or stay near or below chance. The text is separable (TF-IDF), so the failure is the model's decision representation, not the data. Reading: these recipes need per-model selection; "training-free" does not mean "model-free". Harness bug found: SmolLM3's template drops assistant `tool_calls` and renders tool results as user turns, so the follow-up never showed the model its own call; `lib.followup` fixes it (Qwen rendering unchanged), E012 rerun queued.
