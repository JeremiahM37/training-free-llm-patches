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

**E013 hook controls, Qwen3-4B, non-test sets (150 positives of adversarial + contrast + style2; decisions on dev, adversarial, contrast, style2, can't-answer).** Same fit pool and configuration as E011 (L34, rank 4).
| arm | used [95% CI] | adversarial TPR/FPR | contrast TPR/FPR | can't-answer FPR |
|---|---|---|---|---|
| no patch | .18 [.13, .25] | .04/.17 | .13/.02 | .00 |
| few-shot prompt (119 tok) | .71 [.64, .78] | .66/.09 | .88/.03 | .02 |
| RRM r4, every token (as E011) | .73 [.65, .79] | .72/.11 | .88/.02 | .05 |
| RRM r4, prefill only | .74 [.66, .80] | .72/.11 | .88/.02 | .05 |
| constant shift (mean effect) | .31 [.24, .39] | .16/.21 | .32/.02 | .01 |
| random rank-4, same singular values, 3 seeds | .31-.32 | .16/.21 | .28-.32/.02 | .01 |
| TFTV, wide grid (α .05, interior) | .49 [.41, .57] | .50/.24 | .52/.03 | .13 |
| Steer2Edit, wide grid (α −32, interior) | .78 [.71, .84] | .86/.26 | .90/.35 | .71 |
Findings: (1) the gain is the learned map's: a constant shift or random rank-4 maps with identical singular values recover .31, the learned map .74, matching the prompt; this is the end-to-end form of the state-dependence result (H-state supported). (2) Acting on every generated token vs only the decision token makes no difference, so the E011 hook-placement error does not change any result. (3) MMLU under the RRM hook: 0 of 400 answers changed, schema absent or present; constant shift 1 of 400. (4) The published baselines were not understated: on wider grids both choose the same α, now interior. Steer2Edit's higher "used" comes with indiscriminate calling (can't-answer FPR .71, contrast FPR .35). RRM is the only zero-token method matching the prompt on both positives and negatives.

**E012 rerun, SmolLM3-3B, with the follow-up fixed (`lib.followup`).** Decision numbers are identical to the first run. End-to-end on the 120 test positives, called / used [95% CI]: no patch .77 / .60 [.51, .68]; few-shot .68 / .37 [.29, .46]; rules .92 / .69 [.60, .77]; TFTV .00 / .00; Steer2Edit .70 / .35; gated rank-one .18 / .08; CF-LoRA .52 / .41; RRM .78 / .08 [.05, .15]. TF-IDF AUROC .98.
Reading: on a model whose problem is over-calling (FPR .97 unpatched), no frozen Qwen recipe helps: every patch lowers "used" below the unpatched .60, and only the rules prompt raises it (.69, still FPR 1.00). The RRM hook keeps the call rate (.78) but wrecks use (.08): with per-model fitting but Qwen's layer, rank and λ, it acts on every generated token and the answer no longer survives. The first-token decision metric also under-reads RRM here (TPR .25 vs called .78), because the hook makes the model defer the call past the first token. Conclusion for the paper: the recipes are model-specific; transfer without per-model selection fails, and a prompt is the only method that transferred at all.

**E014 = E012 protocol on Qwen3-8B (Phase D, minimal).** LLM.int8 weights except the down_proj of L21/25/34 and lm_head (bf16), so the weight edits stay exact; batch 8 (a first run at batch 32 ran out of memory after the no-patch arm, which read .93/.10 there against .93/.08 here: padding-dependent wobble of one negative). Frozen Qwen3-4B recipes, identical layer indices (36 layers), test split. TPR / FPR / used:
no patch .93/.08/.87; few-shot .93/.02/.84; rules .96/.06/.88; TFTV .03/.00/.17; Steer2Edit .93/.18/.86; gated rank-one .93/.01/.51; CF-LoRA r16 .95/.00/.86; RRM r4 .92/.01/.87; TF-IDF .89/.05.
Reading: at 8B the behaviour is nearly saturated on this (easy) test split, so the only room left is false positives. Within the family the label-free recipes transfer to 2x scale without re-selection: CF-LoRA and RRM cut FPR from .08 to .00-.01 with use unchanged, matching the few-shot prompt at zero tokens. The two known failure modes reproduce: the gated rank-one edit breaks execution (used .51), and TFTV's frozen α shuts calling off. A harder split is needed to separate methods at this scale.

**E015 = per-model, dev-only selection on SmolLM3-3B (2026-11-09).** `evaluations/e015_per_model_selection.py`. Probe (logistic, last-token residual, fit on train labels): dev AUROC .98 at L2 rising to 1.00 from L17; adversarial .76-.88, contrast .95-.99. Model's own first-token decision on dev: AUROC .29; prompts on dev TPR/FPR: none .85/1.00, instruction .00/.18, few-shot .22/.50, rules .95/1.00. 309 candidates selected by dev J. Selected, test TPR/FPR/AUROC, used: no patch .75/.97/.33, .59; rules prompt .84/1.00/.38, .68; TFTV L30 α−.1 1.00/.78/.66, .78; Steer2Edit L12 α64 .80/.63/.60, .59; gated rank-one with a suppressive gate (fit on negatives, value −dL) L12 α4 .78/.18/.87, .59 (adv FPR .40, contrast .25, can't-answer .28 vs .18 unpatched); RRM (rules teacher, L18, r16) .98/1.00/.49, .81; CF-LoRA (rules teacher, L18, r16) .98/1.00/.49, .76.
Reading: the decision is linearly present in SmolLM3 but its output ignores it. Label-free imitation cannot beat its teacher and no prompt is a good teacher here, so RRM/CF-LoRA select "always call". A labelled suppressive gated edit is the only method that fixes the over-calling (FPR .97 -> .18, recall and use unchanged), but it remains far from TF-IDF (.05) and its α is narrow (α2 dev J .40 vs α4 .73). Selection over 309 configs on 120 dev items; test confirms the winner. Training-free transfers as a procedure, not as a configuration.

**E016 = frozen E011 recipes on the fresh hard2 split (2026-11-09).** `evaluations/e016_hard_split.py`, `data/hard2.jsonl` (80 matched pairs, 4 kinds x 20, 15 new domains, written by a Haiku agent to a fixed spec, 12 pairs spot-checked by hand; decoy negatives are formulaic). History turns rendered before the final message. 8B run OOMed at batch 8 in fit-pool collection, rerun at batch 4 with expandable_segments. TPR/FPR/used:
Qwen3-4B: no patch .43/.23/.21; few-shot .66/.19/.39; rules .80/.58/.45; TFTV .56/.48/.29; Steer2Edit .81/.68/.49; gated .55/.20/.03; CF-LoRA .60/.38/.31; RRM .75/.30/.45; TF-IDF .70/.46.
Qwen3-8B 8-bit: no patch .78/.55/.44; few-shot .79/.34/.45; rules .78/.48/.40; TFTV .16/.01/.13; Steer2Edit .84/.63/.49; gated .25/.04/.00; CF-LoRA .74/.29/.39; RRM .83/.38/.53.
Reading: hard2 separates methods and defeats the word classifier. RRM >= its few-shot teacher at both scales (CIs overlap). in_context is unsolved by every method (TPR ~ FPR), since all patches act on the final message's decision token and were fitted without history. Multilingual is easy for most methods. Gated edit and TFTV failure modes reproduce. Per-item outcomes not saved, so no paired tests.

**E017 = history-aware fitting on Qwen3-4B (2026-11-09).** `evaluations/e017_history_aware.py`. Synthetic pairs: final message = a train/dev positive; history states this item's record (present, 0) or the record 7 items on (absent, 1); two framings. Fit on train pairs (80 items), select on dev pairs (120) by J, test on hard2 in_context (40 natural items). First run OOMed at batch 32 on the longer prompts; rerun at batch 8. Per-item scores and used flags saved.
Probe on residual (fit synthetic train): synthetic dev 1.00 from L8; natural .41 (L2), .64-.77 (L5-L35).
Prompts, synthetic dev / natural TPR,FPR,AUROC: none .07,.02,.81 / .05,.05,.46; few-shot .60,.02,.95 / .35,.25,.56; few-shot + history rule .10,.00,.93 / .05,.15,.45.
Selected: gate+ L18 a1 (dev J .10) natural .05/.10/.50; gate- L18 a1 (dev J .10) .10/.00/.50; RRM L34 lam .03 r4 (dev J .62) .40/.30/.56, used IC .40, hard2 all used .34, test FPR .00.
Reading: negative. Template presence is trivially learnable and does not transfer to natural histories; Qwen3-4B mostly stops calling when any history is present; RRM again reaches its teacher and no further. n=20 natural pairs, wide CIs.

**E017b = E017 with naturally written history pairs (2026-11-09).** `FIT=natural`, `data/history_fit.jsonl` (40 pairs by a Haiku agent to a fixed spec, validated: fact in the label-0 history only, no 8-word overlap with memory_record, no prompt overlap). Fit pairs 1-20, select 21-40. Probe (fit natural): selection pairs .79-.99, hard2 in_context .76-.81 at all layers >= 2 (synthetic fit gave .41-.77). Prompts on selection pairs: none .05/.15, few-shot .25/.30, history rule .00/.10 (all AUROC <= .57). Selected: gate+ L21 a4 (dev J .05) hard2 IC .05/.00 AUROC .47; gate- L21 a1 (J .05) .05/.05/.41; RRM L30 r4 (J .00) .20/.25/.51.
Reading: confirmed negative. The presence signal is linearly readable and transfers between natural sets (~.8), but no tested patch or prompt converts it into calling behaviour; the gates and maps cannot even fit the selection pairs. Candidate next methods: a probe-gated router, or interventions at the history positions.

**E018 = probe-gated router for in_context (2026-11-09).** `evaluations/e018_probe_router.py`. Logistic probe (last-token residual) fit on history_fit pairs 1-20; layer L20 (selection AUROC .99) and threshold .252 chosen by J on pairs 21-40. Policy: probe says absent -> decode starting with <tool_call>; present -> <tool_call> masked (bad_words_ids). hard2 in_context (20 pairs), called-when-absent / called-when-present / used / present-items answered from context: no patch .05/.05/.05/.45; few-shot .35/.25/.35/.40; router .75/.25/.75 [.53,.89]/.40. Exact McNemar on used: router vs no patch 15-1 p=.0005; vs few-shot 10-2 p=.039.
Reading: the readable presence signal becomes behaviour once it is read out and acted on directly; the weight/activation patches could not. Same lesson as E006 (router beats patches). Small n; probe needs labelled history pairs; answer quality on suppressed calls unchanged.
E018 on Qwen3-8B (8-bit; first run OOMed in collect at batch 16, COLLECT_BS env added to patchlib, rerun at 4). Probe L11, selection AUROC .99, thr .368. Called-absent / called-present / used / present answered from context: no patch .70/.80/.70/.10; few-shot .85/.85/.85/.10; router .95/.30/.95 [.76,.99]/.55. Paired vs few-shot: false calls 12-1 p=.003, context answers 9-0 p=.004, used 3-1 p=.63. vs no patch: false calls 12-2 p=.013, context answers 9-0 p=.004, used 6-1 p=.13.
Reading: replicates with the opposite base failure (8B over-calls with history, 4B under-calls); the router fixes both directions, and on 8B suppressing the call also makes the model answer from the conversation.

## 2026-11-09: second reading wave (C1-C4, 63 papers) and hard3_ic

- C1-C3: 48 papers hand-picked from the unread, full-text part of the reading queue. Embedding triage had filled it with off-topic theory (neural ODEs, NTK, spiking networks). C4: 15 papers from a title search for tool-decision internals, which triage had missed (2602.04935 ASA returned 404 on arXiv). Every evidence quote was checked mechanically against the full text, 0 failures.
- The wave changes how the in-context results must be positioned. Probe&Prefill (2605.09252, NeurIPS 2026) is the E018 method: a linear probe on the pre-generation state gates a prefill. 2605.14038 reports the E017 knowing-doing gap: necessity and action are both decodable, with near-orthogonal directions late in the network. 2608.25198 steers the call rate from 0 to 90%+ with one training-free direction. SAKIKO 2609.36138 finds decodability does not imply steerability, and logit shifting matches steering on Qwen3-8B. The draft's related work and §5.3f now credit these. Our remaining contribution is the case where the fact is already in the conversation, which none of them test.
- hard3_ic.jsonl: 40 new natural in_context pairs in 20 new domains, with the in-history answer stored explicitly (`context_answer`). In hard2, 2 of 20 pairs state a different value in history than in the memory record, so E018's "answered from context" could not fire on those two (conservative, the same for every arm). The E018 runs on hard3 were cut off when the GPU host went offline; not yet run.
