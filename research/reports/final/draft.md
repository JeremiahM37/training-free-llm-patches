# Prompts, Steering Vectors and Closed-Form Weight Edits for a Tool-Use Decision: What a Training-Free Patch Can and Cannot Do

*First draft, 2026-11-07; revised 2026-11-08 after critique 02. Every number traces to a file in `research/results/metrics/` (the CPU-tier reference output is `T1_cpu_quick_Qwen2.5-0.5B-Instruct.json`). `python repro/tables.py` prints the tables; `python repro/quick.py` reproduces the core mechanisms on a laptop CPU.*

## Abstract

We ask whether a behaviour can be added to a pretrained language model without gradient training, as a compact patch rather than a prompt. The behaviour is *deciding to query an external memory when the answer depends on an earlier conversation*. We score the decision, the execution of the call, and the use of the retrieved answer separately. We compare prompts, a one-number logit bias, two published rank-one weight edits (TFTV, Steer2Edit), a conditional rank-one edit fitted in closed form, a closed-form low-rank adapter fitted from prompt pairs with no labels, and a low-rank activation map at the decision token.

Five findings.
1. **An apparently perfect small-model result was an artefact.** On Qwen2.5-1.5B a 21 KB closed-form patch reached TPR 1.00 and FPR 0.00 on the tuning split. A TF-IDF classifier matched it, and it fired on 81% of requests that were unanswerable for other reasons.
2. **On a capable model a prompt is hard to beat for one behaviour.** On a sealed test with Qwen3-4B, a ~70-token rules prompt yields 93% correct end-to-end answers.
3. **A prompt's effect is largely a low-rank function of the model state, not a constant shift.** At the decision token a constant shift explains R² ≈ 0 of the prompt-induced activation change across held-out inputs; a rank-4 linear map explains 0.37–0.52, depending on layer and representation. This holds on a 0.5B model too, and is consistent with constant steering falling short.
4. **A prompt can be distilled into a zero-token map without labels or gradients.** A rank-4 map fitted only from paired forward passes (the hand-written few-shot prompt is the teacher, so its information enters through the prompt) is statistically indistinguishable from that 119-token prompt at zero prompt tokens: 87% vs 83% end-to-end on 120 test positives (95% CIs [.79, .92] and [.75, .88]). Controls show the gain is the learned map's: a constant shift or random rank-4 maps with the same singular values reach 31%, the learned map 74%, on 150 non-test positives (E013), and no MMLU answer changed. It needs a runtime hook. A hook-free closed-form LoRA recovers less of the gain, and no MMLU drop was detected (400 items; a few-point drop would not be resolvable).
5. **Patches have hard limits.** A conditional rank-one edit that decides correctly can wreck the call that follows (11% end-to-end). With the value construction tested, no patch produced a valid call without the tool schema (0/90). The recipes transferred to Qwen3-8B without re-selection, but not to a second model family (SmolLM3-3B). With many behaviours at once, patches beat an all-rules prompt at small K but fall behind a linear probe on the same features by K = 16 (ranges overlap at K = 4 and against the router at K = 16).

Conditional closed-form weight edits (SteerEdit) and prompt-to-weights ridge compilation (ICC) are prior art. We do not claim them; our contribution is the controlled comparison, the state-dependence measurement, and the limits. Probe-gated control of the call decision (Probe&Prefill 2605.09252) and the readable-but-not-acted-on gap (2605.14038) are also prior art. We add the in-conversation case.

## 1. Question and scope

Fine-tuning, reinforcement learning and adapter training are the usual ways to change a model's behaviour. Pretrained models often already represent the behaviour they should exhibit. A model that understands memory tools may simply not use them consistently. Can such a behaviour be switched on by a small computed modification, with no gradient updates?

We study one behaviour in depth, because it decomposes cleanly into three capabilities that a single "tool call rate" metric would conflate:
- **decision:** emit `<tool_call>` as the first token when, and only when, memory is needed;
- **execution:** a well-formed `recall_memory` call with a sensible query;
- **use:** given the retrieved record, answer with the stored fact.

Methods in scope need no gradient training. All fitting is forward passes plus closed-form linear algebra, and costs seconds.

## 2. Related work (closest)

- **Constant steering:** CAA 2312.06681, ActAdd 2308.10248, ITI 2306.03341. Its reliability is known to vary per input: 2407.12404, 2505.22637.
- **Gated steering:** CAST 2409.05907 gates a steering vector on cosine similarity with a condition vector. GAPS 2609.01878 and ORBIT 2606.22357 refine the gate.
- **Steering compiled into weights:** Arditi et al. 2406.11717, TFTV 2609.09054 and Steer2Edit 2602.09870 are unconditional. SteerEdit 2604.12359 is trigger-conditional, with null-space constraints, and is used for backdoors.
- **Prompts compiled into weights:** In-Context Compilation (Findings of EMNLP 2026) uses closed-form ridge on last-token activations into attention output rows. SELF-PARAM 2410.00487 and Doc-to-LoRA 2602.15902 use gradients or meta-trained hypernetworks. PSR 2605.03907 trains a gated rank-1 mimic of a prompt.
- **Tool use:** PRISMS 2608.00218 detects tool misuse from a few neurons and steers with a separate probe. Heading-specific steering 2607.05790 suppresses tool calls.
- **Reading and steering the call decision (found in the second reading wave, after E018 was run):**
  - Probe&Prefill 2605.09252 is the closest prior method to our E018 router. A linear probe on the pre-generation hidden state reads tool necessity (AUROC .89–.96 across six models) and gates a prefilled steering sentence. It cuts calls by 38% at 1.7% accuracy loss.
  - 2605.14038 reports the knowing-doing gap we see in E017/E017b. Necessity and the call action are both linearly decodable, but their probe directions are nearly orthogonal in late layers at the last token, and most mismatches arise between cognition and action.
  - 2608.25198 controls the call rate from near 0% to over 90% with a single training-free residual direction, in both directions. 2605.07990 reads and steers which tool is chosen from tool-vector centroids. 2610.09624 traces the call decision to a vector shaped by suppression of a scaffold-set prior.
  - SAKIKO 2609.36138 audits router-conditioned steering of K-way pre-execution tool decisions. It finds linear decodability does not imply steerability, and on Qwen3-8B an output-score (logit-shift) baseline matches activation steering on net gain.
  - None of these tests whether the needed fact is already in the conversation. That case is ours (E016–E018).

`reports/progress/landscape.md` gives the map built from 30k harvested records and 128 papers read.

## 3. Setup

**Models.** Qwen2.5-0.5B, 1.5B and 3B-Instruct, and Qwen3-4B-Instruct-2507, all using their native tool-call templates. The memory tool is `recall_memory(query)`.

**Data.** All splits except CLINC150 were written for this study (by Haiku, except style2, written by the PI), with domains disjoint across splits:

| split | positives / negatives | purpose |
|---|---|---|
| train | 40 / 40 | fitting |
| dev | 60 / 60 | tuning |
| test | 120 / 120 | sealed until E011 |
| adversarial | 50 / 70 | implicit references; memory as a topic; the past supplied in the message |
| contrast | 60 minimal pairs | near-identical wording, opposite labels |
| can't-answer | 100 negatives | requests that cannot be answered for reasons memory would not fix: realtime, refusal, physical action, missing attachment |
| style2 | 40 / 40 | a second author, first-person in both classes |
| CLINC150 | | human-written intents, for the many-behaviour study |

Author style is therefore a known confound. A TF-IDF classifier is reported as a strong text baseline (dev, adversarial and contrast on 1.5B; every split on Qwen3-4B); it can exploit author style, so beating it is necessary but not sufficient.

**Controls.** The controls are a TF-IDF classifier, a one-number logit bias, the best of four prompts, and MMLU (400 items, McNemar test).

## 4. Methods

**Identity.** For a block `h_out = h + W a`, the edit `W' = W + αΔvᵀ` gives exactly `h' = h + α(vᵀa)Δ`. A gated steering intervention is therefore a rank-one LoRA, with the gate computed from the block's own MLP features. We confirmed the identity numerically in float32: 1.6e-5 logits on 0.5B (CPU tier) and 2e-5 on 1.5B (E008). In bf16, rounding the merged weight alone moves logits by at most .014 (E008), but running the whole bf16 forward pass with hook versus merged edit differs by up to 0.75 logits against a 5.9 effect (E003); all GPU results use bf16 with the merged edit.

**Conditional rank-one edit (DKE).** `v = R⁻¹(μ₊ − μ₋)` with `R` the uncentred second moment of background keys, solved by Woodbury in sample space. `Δ` is the prompted-minus-unprompted (or positive-minus-negative) residual mean. We tested a minimum-variance version with zero mean response on negatives (LCMV), and it was worse (E002, E004).

**Reduced-rank map at the decision token (RRM).** This is a dual ridge regression from unprompted to prompted residuals at the last token, truncated to rank r by SVD. Its inputs are paired forward passes over unlabelled utterances; no labels are used, though the teacher prompt is hand-written. It runs as a forward hook on the last position. **Correction (2026-11-08):** in E011 that hook edits the last position of every forward call, so during generation it also acts on every generated token, not only the decision token. E013 shows this makes no difference (end-to-end .73 vs .74 prefill-only).

**Closed-form LoRA (CF-LoRA).** This is the same regression from MLP keys, with zero targets at every other position (template, user text, generated text), so the position logic lives in the weights. It runs as a plain LoRA, and its strength comes from the regression, with no α.

**Baselines.** TFTV and Steer2Edit are reimplemented from the papers' equations, on a single module for comparability. Their best α sat at the edge of the E005 grid; re-selected on a wider dev grid (E013) both choose the same α, now interior.

## 5. Results

### 5.1 The small-model trap (Qwen2.5-1.5B; E002–E004, E008)

The best closed-form patch on dev reached TPR 1.00 and FPR 0.00, as the maximum of 720 configurations. On frozen settings, three things undid it.

1. **The data is nearly text-separable.** TF-IDF gets AUROC .999 on dev and .987 on contrast pairs.
2. **The gate fires on 81% of can't-answer requests,** against 19% unpatched. It encodes something closer to "I cannot answer from what I have" than "this needs memory". Adding such negatives to the fit cut that rate to 12%, at a cost in recall (E004).
3. **The false-positive gain comes from a signed push.** On this trigger-happy model, clamping the gate's negative side raises dev FPR from .00 to .47. The patch works by pushing negatives away, not by staying silent on them (E008).

Pooled 5-fold CV over all hard data gives TF-IDF .980 against the best gate .930. The difference is −.050, with 95% CI [−.076, −.028]. The CPU tier shows the same pattern on Qwen2.5-0.5B (patch and prompt on dev; TF-IDF is pooled 5-fold CV, so not the same quantity):

| method (0.5B, tier 1) | AUROC | can't-answer FPR |
|---|---|---|
| few-shot prompt | .46 | .55 |
| rank-one patch | .98 | .50 |
| TF-IDF (5-fold CV, all data) | .98 | n/a |

### 5.2 On a capable model, the ranking reverses (Qwen3-4B)

Qwen3-4B separates the classes unaided (AUROC .971 with only the schema) but under-calls (TPR .50). Prompts now work well, a logit bias overshoots (style2 FPR .55), and gating adds little. In E002, at dev selection, an unconditional steer reaches J .88 against .90–.95 for gated variants. Activations are now close to words: CV AUROC of the best of about 12 gate layers .973 vs TF-IDF .980, difference CI [−.024, .010]. With layer selection unpenalised and no equivalence margin, this is "no detected difference", not a tie.

**Table 1. Sealed test, single use, Qwen3-4B (E011).** 120 positives and 120 negatives from unseen domains, with every method frozen beforehand. These test domains are easier than the adversarial and style2 sets: the unpatched model already reaches AUROC .97.

| method | prompt tokens | TPR | FPR | used the answer (of 120), 95% CI |
|---|---|---|---|---|
| no patch | 0 | .61 | .00 | .56 [.47, .64] |
| few-shot prompt | 119 | .89 | .00 | .83 [.75, .88] |
| rules prompt | ~70 | .99 | .10 | **.93** [.86, .96] |
| TFTV (rank-one)† | 0 | .88 | .05 | .83 [.75, .88] |
| Steer2Edit (rank-one)† | 0 | .95 | .08 | .88 [.81, .93] |
| gated rank-one (DKE) | 0 | .88 | .00 | **.11** [.06, .18] |
| CF-LoRA r16, no labels | 0 | .71 | .00 | .65 [.56, .73] |
| RRM r4, no labels, hook | 0 | .94 | .00 | **.87** [.79, .92] |
| TF-IDF (decision only) | n/a | .89 | .05 | n/a |

"Used" is graded on the 120 positives only: the model called `recall_memory`, and its answer after the stored record contains the record's answer string. False calls on negatives show only in FPR (rules prompt .10, CI [.06, .17]). Single seed. † α at the edge of the E005 grid; a wider grid (E013) picks the same α.

Three readings of Table 1:
- **RRM is indistinguishable from the prompt it imitates** at zero prompt tokens: .87 vs .83 end-to-end, overlapping CIs at n = 120. On harder non-test sets (TPR/FPR, E007) it trails the prompt on adversarial (.48/.13 vs .68/.09) and contrast (.75/.02 vs .87/.03), and matches it on style2 (.98/.08 vs .93/.10). The learned map, not the intervention itself, causes the gain (§5.3, E013).
- **The conditional rank-one edit decides well but breaks generation.** It remains active while the call is written, and execution falls from .98 called to .48 executed. Unconditional published edits do not have this failure.
- **CF-LoRA: no side effect detected.** MMLU .7175 → .71 on 400 items (4 worse, 1 better, p = .375; a drop of a few points would not be resolvable), and greedy generations identical on 30 prompts (E010). It recovers less of the prompt's end-to-end gain over no patch: 58% on 150 non-test positives ((.49−.18)/(.71−.18)) and 33% on test ((.65−.56)/(.83−.56)).

### 5.3 The prompt's effect is state-dependent and low-rank (E007, E010, tier 1)

**Table 2.** Held-out R² for predicting the prompt-induced change at the decision token, from unlabelled utterance pairs.

| model, input | constant shift | rank 1 | rank 4 | rank 16 |
|---|---|---|---|---|
| Qwen3-4B, residual, L28 / L34 | −.02 / −.03 | .19 / .21 | .37 / .40 | .51 / .50 |
| Qwen3-4B, MLP keys, L21 (E010) | | .03 | .52 | .67 |
| Qwen2.5-0.5B, residual, L14 / L19 / L21 (CPU tier) | −.02 / −.02 / −.02 | .14 / .16 / .15 | .30 / .36 / .38 | .44 / .52 / .50 |

The constant shift is the training-set mean, so its held-out R² is near zero by construction: the point is that the effect varies across inputs, and a low-rank linear function of the state captures a third to two thirds of that variation. This agrees with ICC's observation that fixed task vectors recover only part of in-context gains, and suggests (not shows) why steering underperforms prompting: steering applies the average effect, the prompt an input-specific one. E013 tests this end to end at the same hook, on 150 non-test positives:

**Table 2b. End-to-end controls at the RRM hook (Qwen3-4B, E013).**

| arm | used [95% CI] | contrast FPR | can't-answer FPR |
|---|---|---|---|
| no patch | .18 [.13, .25] | .02 | .00 |
| few-shot prompt (119 tokens) | .71 [.64, .78] | .03 | .02 |
| RRM rank 4, learned | .74 [.66, .80] | .02 | .05 |
| constant shift (mean effect) | .31 [.24, .39] | .02 | .01 |
| random rank-4 map, same singular values (3 seeds) | .31–.32 | .02 | .01 |
| Steer2Edit, wide grid | .78 [.71, .84] | .35 | .71 |

The constant shift and random maps of identical spectrum recover about a quarter of the prompt's gain; the learned map recovers all of it. So the input-specific part of the map is what does the work. Steer2Edit's higher score comes from calling indiscriminately. Under the RRM hook, 0 of 400 MMLU answers changed (schema absent or present).

### 5.3b The recipes do not transfer to another model family (E012, SmolLM3-3B)

Every method keeps its Qwen3-4B configuration (same layer indices, both models have 36 layers; same α, rank and λ) and is refitted only from SmolLM3's own forward passes. Nothing is tuned on SmolLM3. Test split, 120 + 120:

| method | TPR | FPR | AUROC | called | used [95% CI] |
|---|---|---|---|---|---|
| no patch | .74 | .97 | .33 | .77 | .60 [.51, .68] |
| few-shot prompt | .34 | .52 | .34 | .68 | .37 [.29, .46] |
| rules prompt | .88 | 1.00 | .38 | .92 | .69 [.60, .77] |
| TFTV | .00 | .00 | .19 | .00 | .00 |
| Steer2Edit | .28 | .10 | .55 | .70 | .35 [.27, .44] |
| gated rank-one | .11 | .03 | .59 | .18 | .08 [.04, .14] |
| CF-LoRA r16 | .49 | .75 | .40 | .52 | .41 [.32, .50] |
| RRM r4 hook | .25 | .66 | .23 | .78 | .08 [.05, .15] |
| TF-IDF (decision only) | .89 | .05 | .98 | | |

SmolLM3 has the opposite problem to Qwen: it calls the tool on almost every negative, and its first-token score ranks the classes backwards. No frozen recipe helps: every patch lowers end-to-end use below the unpatched model, and only the rules prompt raises it (while still calling on every negative). The text is separable, so this is the model's decision representation, not the data. The RRM hook keeps the call rate but destroys use of the answer; the first-token metric also under-reads it here (TPR .25 vs called .78), since the hook makes the model defer the call. **Training-free does not mean model-free:** these recipes need per-model selection, and a prompt was the only method that transferred at all.

### 5.3c Scale within the family (E014, Qwen3-8B, 8-bit)

Same frozen recipes and layer indices, on Qwen3-8B with LLM.int8 weights except the edited down_proj layers (kept in bf16 so the edits stay exact). Test split:

| method | TPR | FPR | used |
|---|---|---|---|
| no patch | .93 | .08 | .87 |
| few-shot prompt | .93 | .02 | .84 |
| rules prompt | .96 | .06 | .88 |
| TFTV | .03 | .00 | .17 |
| Steer2Edit | .93 | .18 | .86 |
| gated rank-one | .93 | .01 | .51 |
| CF-LoRA r16 | .95 | .00 | .86 |
| RRM r4 hook | .92 | .01 | .87 |

At 8B the behaviour is nearly saturated on this split; only false positives are left to fix. Within the family the label-free recipes transfer to twice the scale without re-selection: CF-LoRA and RRM bring FPR from .08 to about zero with use unchanged, matching the few-shot prompt at zero tokens. The gated edit again breaks the call it writes, and TFTV's frozen α again shuts calling off. A harder split is needed to separate methods at this scale.

### 5.3d Per-model selection on SmolLM3 (E015)

Is SmolLM3's failure in E012 the recipes' fault or the model's? Two measurements, all selection on dev only (J = TPR − FPR at 0, 309 candidate configurations over layer, α, rank, λ and teacher prompt):

**The information is there; the readout is wrong.** A logistic probe on the last-token residual, fitted on the 80 train labels, separates dev perfectly from layer 17 on (AUROC 1.00; adversarial .88, contrast .99). The model's own first-token decision on the same dev prompts has AUROC .29. Prompts do not repair it: the instruction suppresses calling entirely (TPR 0), the rules prompt calls on everything (FPR 1.00), few-shot sits between (.22 / .50).

Selected configurations, test split (120 + 120), with held-out FPR on the other splits:

| method (dev-selected config) | dev J | test TPR | test FPR [95% CI] | AUROC | adv. FPR | contrast FPR | can't-answer FPR | used [95% CI] |
|---|---|---|---|---|---|---|---|---|
| no patch | −.15 | .75 | .97 [.92, .99] | .33 | .91 | .75 | .18 | .59 [.50, .68] |
| best prompt (rules) | −.05 | .84 | 1.00 | .38 | 1.00 | .97 | .42 | .68 [.59, .75] |
| TFTV (L30, α −.1) | .22 | 1.00 | .78 | .66 | .91 | 1.00 | .98 | .78 |
| Steer2Edit (L12, α 64) | .27 | .80 | .63 | .60 | .46 | .53 | .55 | .59 |
| **gated rank-one, suppressive gate (L12, α 4)** | **.73** | .78 | **.18 [.12, .25]** | **.87** | .40 | .25 | .28 | .59 [.50, .68] |
| RRM hook (rules teacher, L18, r16) | .00 | .98 | 1.00 | .49 | 1.00 | .97 | .69 | .81 |
| CF-LoRA (rules teacher, L18, r16) | .00 | .98 | 1.00 | .49 | 1.00 | .98 | .71 | .76 |
| TF-IDF (E012, decision only) | | .89 | .05 | .98 | | | | |

Three readings:
- **The label-free methods are bounded by their teacher.** RRM and CF-LoRA imitate a prompt; on SmolLM3 no prompt is a good teacher (best dev J −.05), so the best they can select is "call on everything". Their higher "used" is that, not a better decision.
- **A labelled, gated edit does fix it.** With the gate fitted to fire on negatives and push away from the call, one rank-one weight edit cuts false calls on test from 97% to 18% at unchanged recall and unchanged end-to-end use, and it holds on adversarial and contrast pairs. It does not break the call it writes here (used .59 = no patch), unlike the call-promoting gate on Qwen.
- It is still well short of a text classifier (FPR .05), the selected setting is narrow (α 2 at the same layer gives dev J .40), and the can't-answer FPR rises from .18 to .28.

So "training-free" transfers as a procedure (a fixed recipe plus a dev-only sweep, minutes on one GPU), not as a configuration. Which method family wins depends on the model's failure mode: imitation where a prompt already works (Qwen), a labelled suppressive gate where no prompt does (SmolLM3).

### 5.3e A harder held-out split (E016, hard2)

The original test split is spent and near ceiling at 8B. `hard2` is a fresh split never used for any selection: 80 matched pairs (160 items) in 15 domains absent from all other splits, of four kinds (20 pairs each):
- **in_context:** the needed fact is, or is not, already in this conversation's earlier turns;
- **multilingual:** Spanish, French, German, Portuguese;
- **buried:** one reference to an earlier conversation inside a 60–120-word message;
- **decoy:** memory-sounding words with no memory need, against implicit references such as "same as last time".

The decoy negatives are formulaic ("I remember …, right?"). Frozen E011 configurations, history rendered before the final message for every method. TPR / FPR / used, with 95% CIs on used (80 positives):

| method | Qwen3-4B TPR / FPR | used [95% CI] | Qwen3-8B (8-bit) TPR / FPR | used [95% CI] |
|---|---|---|---|---|
| no patch | .43 / .23 | .21 [.14, .31] | .78 / .55 | .44 [.33, .55] |
| few-shot prompt (119 tokens) | .66 / .19 | .39 [.29, .50] | .79 / .34 | .45 [.35, .56] |
| rules prompt | .80 / .58 | .45 [.35, .56] | .78 / .48 | .40 [.30, .51] |
| TFTV | .56 / .48 | .29 | .16 / .01 | .13 |
| Steer2Edit | .81 / .68 | .49 | .84 / .63 | .49 |
| gated rank-one | .55 / .20 | .03 [.01, .09] | .25 / .04 | .00 [.00, .05] |
| CF-LoRA r16 | .60 / .38 | .31 [.22, .42] | .74 / .29 | .39 [.29, .50] |
| **RRM r4 hook (0 tokens)** | .75 / .30 | **.45 [.35, .56]** | .83 / .38 | **.53 [.42, .63]** |
| TF-IDF (final message only) | .70 / .46 | | .70 / .46 | |

The split works: the text classifier drops from .89/.05 to .70/.46, and both models lose most of their end-to-end success (4B .21, 8B .44 used). Four readings:
- **The zero-token map again matches or exceeds the few-shot prompt it imitates**, at both scales (.45 vs .39 at 4B, .53 vs .45 at 8B; the intervals overlap, so "matches" is the supported claim). It is the best or tied-best method on "used" at both sizes, with no re-selection.
- **No method handles in_context.** Every method, prompts included, fires about as often when the fact is already in the conversation as when it is not (e.g. 8B no patch .65 vs .85, RRM .80 vs .75). All patches here act on the final message's decision token and none was fitted with history, so this is a behaviour the current methods do not reach, not one they get wrong in degree.
- **Multilingual is the easy kind:** fitted on English only, no patch, few-shot, CF-LoRA and RRM keep TPR ≥ .90 at FPR ≤ .05 on both models. The gated edit (TPR .75 / .70) and TFTV at 8B (.65) lose recall; rules and Steer2Edit on 4B raise FPR (.40 / .50).
- **The known failure modes reproduce on new data:** the gated rank-one edit decides but does not complete the call (used .03 / .00), and TFTV's frozen α suppresses calling at 8B.

The results are summary rates only; per-item outcomes were not saved, so paired tests between methods are not available for E016.

### 5.3f Fitting on history pairs does not reach natural in-context cases (E017, Qwen3-4B)

Can the in_context failure be patched? Synthetic pairs were built from the train and dev positives: the same final message, with an earlier turn that states either this item's memory record (fact present, label 0) or another item's record (fact absent, label 1). Patches were fitted on the train pairs, selected on the dev pairs (108 configurations) and tested on hard2's 20 naturally written in_context pairs.

- **Presence is perfectly readable in the template and weakly readable in natural text.** A probe fitted on synthetic train pairs reaches AUROC 1.00 on synthetic dev from layer 8 on, but only .66–.77 on the natural pairs.
- **With any history, Qwen3-4B almost stops calling.** No patch: synthetic dev TPR .07, natural .05. The few-shot prompt restores calling on synthetic pairs (.60 / .02) but not discrimination on natural ones (.35 / .25, AUROC .56).
- **Patches inherit the same ceiling.** Natural in_context TPR / FPR / AUROC:
  - RRM fitted on history pairs: selected dev J .62, at its teacher's .58; natural .40 / .30 / .56, used .40. Elsewhere it is unchanged (hard2 all-kinds used .34 vs .39 for few-shot; test FPR .00).
  - Gated edits in either direction: barely moved synthetic dev (best J .10), and on natural pairs .05–.10 / .00–.10, AUROC .50.
  - An explicit history rule added to the prompt: suppresses calling on natural in_context (.05 / .15) and raises hard2 FPR to .59.

With 20 natural pairs the intervals are wide (e.g. RRM TPR .40 [.22, .61]). The supported conclusion is negative: a template-built history signal is learned perfectly and does not transfer to natural conversations, and no tested patch or prompt makes Qwen3-4B tell whether a fact is already in its context. **E017b, fitting on natural pairs.** The same protocol, with 40 naturally written history pairs (`history_fit.jsonl`, 10 earlier-used domains, disjoint from hard2): fit on pairs 1–20, select on 21–40, test on hard2. The probe improves to AUROC .78–.81 on hard2 at every layer from 2 on, so the signal is present and partly shared across natural conversations. Nothing turns it into behaviour:
- no selected patch beats chance even on the selection pairs (best dev J: gates .05, RRM .00);
- on hard2 in_context, AUROC is .41–.51 for every patch and .45–.56 for every prompt.

So the limit is not the template: the presence of a fact in the conversation is linearly readable at the decision token, but none of these patches, nor a prompt, makes Qwen3-4B act on it. Two directions follow: **a probe-gated router** (tested next, E018) and **an intervention on the history positions** rather than on the final token (not tried).

**E018, a probe-gated router does act on it.** The policy:
1. A logistic probe on the last-token residual decides. It is fitted on the 20 natural fit pairs, with the layer (L20) and threshold chosen on the other 20.
2. "Fact absent" means the call is forced: decoding starts with `<tool_call>`.
3. "Fact present" means the `<tool_call>` token is masked, and the model answers from the conversation.

Results on hard2's 20 in_context pairs, never used for any choice:

| arm | calls when the fact is absent | calls when it is present | used [95% CI] | fact-present items answered from context |
|---|---|---|---|---|
| no patch | .05 | .05 | .05 [.01, .24] | .45 |
| few-shot prompt | .35 | .25 | .35 [.18, .57] | .40 |
| **probe router** | .75 | .25 | **.75 [.53, .89]** | .40 |

The router beats no patch (15 items better, 1 worse, exact McNemar p = .0005) and the few-shot prompt (10 better, 2 worse, p = .04) on end-to-end use, with the same false-call rate as few-shot. It adds no prompt tokens: one 2,560-wide logistic probe read at the decision token, plus a constrained first token.

It is a router, not a weight patch, which repeats §5.4: when a signal is linearly present but the patches cannot steer the model with it, reading it out and acting on it directly works. The method is not new. Probe&Prefill (2605.09252) gates a prefill on a linear probe in the same way, and 2605.14038 describes the same knowing-doing gap. What E017/E018 add is the in-conversation case, a router that acts in both directions (forcing and masking the call), and end-to-end scoring. The caveats:
- n is small (20 pairs).
- The probe needs labelled history pairs (40 here).
- Answering from context when the call is suppressed is no better than without the router (.40 vs .45), so on 4B the router fixes the decision, not the answer.

**Replication on Qwen3-8B (8-bit), same protocol.** The probe is at L11 (selection AUROC .99). The 8B model fails the other way: with history it calls even when the fact is already there.

| arm | called (absent) | called (present) | used [95% CI] | present answered from context |
|---|---|---|---|---|
| no patch | .70 | .80 | .70 [.48, .85] | .10 |
| few-shot prompt | .85 | .85 | .85 [.64, .95] | .10 |
| **probe router** | .95 | **.30** | **.95 [.76, .99]** | **.55** |

Against the few-shot prompt, the router:
- cuts false calls on fact-present items (12 better, 1 worse, p = .003);
- raises answers drawn from the conversation from .10 to .55 (9–0, p = .004);
- keeps use on fact-absent items (.95 vs .85, 3–1, n.s.).

So the same recipe works across a model that under-calls (4B) and one that over-calls (8B), with the probe fitted afresh on each model from the same 40 pairs.

### 5.4 Many behaviours at once (E006, CLINC150)

Each of K intents triggers a distinct code word as the reply's first token. We compare all K rules in the prompt, a router that inserts one rule, K-rank patches, and a linear probe on the same keys.

| K | all rules: J (tokens, s/prompt) | router + 1 rule | best patch | linear probe |
|---|---|---|---|---|
| 1 | .30–.59 (18, .007) | .00–.80 | .80–.99 | .60–1.0 |
| 4 | .39–.81 (75, .012) | .32–.47 | .57–.81 | .41–.77 |
| 16 | .14–.35 (291, .030) | .35–.52 | .43–.48 | .56–.68 |
| 64 | −.05 (1154, .132) | .46 | .31 | .56–.62 |
| 150 | −.17 (2719, .392) | .19 | .18 | −.29 (no rejection class) |

J is correct minus false positives. Ranges span variants or layers; there are no CIs. The "best patch" is selected after the fact from 6 variants, which flatters it. At K = 150 the patch's class means are linearly dependent by construction, so that row is an artefact for patches.

- **Prompts:** degrade quickly, and per-prompt latency at K = 64 is 19× that at K = 1 (.132 vs .007 s), 56× at K = 150.
- **Patches:** win at K ≤ 4, then fall behind the probe built on the same features. The gate holds up but the value path interferes, consistent with H13.
- **Capacity:** our zero-forcing capacity prediction was wrong as stated. The participation ratio (3–34) is far below K = 140, yet the measured noise gain is only 1.6–2.2.

### 5.5 Limits (E009)

Without the tool schema in context, no patch produced a valid `recall_memory` call: 0 of 90 in both arms (one value construction, one model). With a value vector that contained the schema, the model degenerated into repeated `<tool_call>` tokens. Without it, the model never called at all. A patch can select among behaviours the context makes available. In these tests it did not supply the missing interface.

### 5.6 Negative results kept

- **LCMV zero-mean gate:** worse than LDA (E002, E004).
- **Label-free compile from model-written utterances:** that data was too easy to calibrate strength, so selection tied and chose a near-no-op (E007).
- **Logit bias:** over-triggers (E003).
- **Merged-weight edits:** bf16 inference arithmetic, not weight rounding, causes most of the hook-to-edit gap; simulated Q4 destroys the base model before any patch is applied (E008).

## 6. Cost

- **Fit time:** seconds of closed-form solving after a few hundred forward passes (E003, 1.5B: 1.44–1.48 s solve, 3.8 s of forward passes).
- **Patch size:** 21 KB for a rank-one patch (E003) and 0.4 MB for CF-LoRA r16 (fp16, E010).
- **Inference cost:** no added prompt tokens. CF-LoRA adds rank-16 multiplies at one layer. RRM adds a rank-4 map plus a forward hook at one layer; its latency was not measured.
- **Prompt cost:** 70–119 tokens per request for one behaviour, and 1,154 tokens for 64 rules.

## 7. Limitations

- **Models:** Qwen2.5 and Qwen3 (one vendor) at 0.5–4B, plus frozen-recipe transfers to SmolLM3-3B (E012, failed) and Qwen3-8B in 8-bit (E014, held, but near ceiling on an easy split). Nothing above 8B.
- **Behaviour:** one behaviour in depth, plus one synthetic many-behaviour task.
- **Data:** hand-written, with author style a known confound. The test split proved easier than the adversarial sets.
- **Seeds and samples:** single seeds for most runs. End-to-end differences at n = 120 below about 0.08 are not resolvable.
- **Baselines:** TFTV and Steer2Edit are single-module reimplementations.
- **Literature:** 128 of 30k harvested papers read, and the citation graph covers 23% of records.
- **Hook dependence:** the best zero-token method (RRM) still needs a runtime hook at the decision position. CF-LoRA removes the hook but loses accuracy, and closing that gap is the obvious next step.

## 8. Dual use

A conditional, weight-resident patch is structurally a backdoor (SteerEdit 2604.12359). This study includes no attack evaluation. We recommend that behaviour patches ship as separate, signed adapters rather than merged weights, so they can be audited and removed.

## 9. Reproducibility

| tier | what it does | requirements | time |
|---|---|---|---|
| 0 | `repro/tables.py` rebuilds every table above from shipped JSON | standard library only | 1 s |
| 1 | `repro/quick.py` reproduces the identity, the state dependence, the words-versus-activations control, the 0.5B decision table and the capacity curve | laptop CPU, 8 GB RAM | ~15 min (852 s on 8 cores, 5.6 GB peak) |
| 2 | `research/experiments/*` runs the full experiments | one ≥16 GB GPU | |

Every result JSON records model revision, data hash, configuration, seeds and hardware. Literature tooling is in `research/tools/`.
