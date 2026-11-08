# Prompts, Steering Vectors and Closed-Form Weight Edits for a Tool-Use Decision: What a Training-Free Patch Can and Cannot Do

*First draft, 2026-11-07. Every number traces to a file in `research/results/metrics/`. `python repro/tables.py` prints the tables; `python repro/quick.py` reproduces the core mechanisms on a laptop CPU.*

## Abstract

We ask whether a behaviour can be added to a pretrained language model without gradient training, as a compact patch rather than a prompt. The behaviour is *deciding to query an external memory when the answer depends on an earlier conversation*. We score the decision, the execution of the call, and the use of the retrieved answer separately. We compare prompts, a one-number logit bias, two published rank-one weight edits (TFTV, Steer2Edit), a conditional rank-one edit fitted in closed form, a closed-form low-rank adapter fitted from prompt pairs with no labels, and a low-rank activation map at the decision token.

Five findings.
1. **An apparently perfect small-model result was an artefact.** On Qwen2.5-1.5B a 21 KB closed-form patch reached TPR 1.00 and FPR 0.00 on the tuning split. A TF-IDF classifier matched it, and it fired on 81% of requests that were unanswerable for other reasons.
2. **On a capable model a prompt is hard to beat for one behaviour.** On a sealed test with Qwen3-4B, a ~70-token rules prompt yields 93% correct end-to-end answers.
3. **A prompt's effect is a low-rank function of the model state, not a constant shift.** At the decision token a constant shift explains R² ≈ 0 of the prompt-induced activation change (held out). A rank-4 map explains 0.37–0.52, depending on layer and representation. This holds on a 0.5B model too, and it accounts for why constant steering falls short.
4. **That structure can be compiled without labels or gradients.** A rank-4 map fitted only from paired forward passes matches the 119-token few-shot prompt it imitates at zero prompt tokens: 87% vs 83% end-to-end on 120 test positives, a difference within noise. A hook-free closed-form LoRA recovers less of the gain, with no measurable MMLU change.
5. **Patches have hard limits.** A conditional rank-one edit that decides correctly can wreck the call that follows (11% end-to-end). No patch produces a valid call without the tool schema (0/90). With many behaviours at once, patches beat an all-rules prompt but lose to a retrieval router or a linear probe beyond about 16 behaviours.

Conditional closed-form weight edits (SteerEdit) and prompt-to-weights ridge compilation (ICC) are prior art. We do not claim them; our contribution is the controlled comparison, the state-dependence measurement, and the limits.

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

`reports/progress/landscape.md` gives the map built from 30k harvested records and 128 papers read.

## 3. Setup

**Models.** Qwen2.5-0.5B, 1.5B and 3B-Instruct, and Qwen3-4B-Instruct-2507, all using their native tool-call templates. The memory tool is `recall_memory(query)`.

**Data.** All data was hand-written for this study, with domains disjoint across splits:

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

Haiku wrote the main splits and the PI wrote style2, so author style is a known confound. We control for it with a text classifier on every split.

**Controls.** The controls are a TF-IDF classifier, a one-number logit bias, the best of four prompts, and MMLU (400 items, McNemar test).

## 4. Methods

**Identity.** For a block `h_out = h + W a`, the edit `W' = W + αΔvᵀ` gives exactly `h' = h + α(vᵀa)Δ`. A gated steering intervention is therefore a rank-one LoRA, with the gate computed from the block's own MLP features. We confirmed the identity numerically: a 1.5e-5 logit difference on a 0.5B model and 2e-5 on 1.5B, in float32. In bf16 inference, arithmetic rounding raises the gap to about 0.75 logits.

**Conditional rank-one edit (DKE).** `v = R⁻¹(μ₊ − μ₋)` with `R` the uncentred second moment of background keys, solved by Woodbury in sample space. `Δ` is the prompted-minus-unprompted (or positive-minus-negative) residual mean. We tested a minimum-variance version with zero mean response on negatives (LCMV), and it was worse (E002, E004).

**Reduced-rank map at the decision token (RRM).** This is a dual ridge regression from unprompted to prompted residuals at the last token, truncated to rank r by SVD. Its inputs are paired forward passes over unlabelled utterances, and no labels are used. It is applied at the decision position, which requires a runtime hook.

**Closed-form LoRA (CF-LoRA).** This is the same regression from MLP keys, with zero targets at every other position (template, user text, generated text), so the position logic lives in the weights. It runs as a plain LoRA, and its strength comes from the regression, with no α.

**Baselines.** TFTV and Steer2Edit are reimplemented from the papers' equations, on a single module for comparability. Their best α sat at the edge of the grid, so these baselines may be understated.

## 5. Results

### 5.1 The small-model trap (Qwen2.5-1.5B; E002–E004, E008)

The best closed-form patch on dev reached TPR 1.00 and FPR 0.00, as the maximum of 720 configurations. On frozen settings, three things undid it.

1. **The data is nearly text-separable.** TF-IDF gets AUROC .999 on dev and .987 on contrast pairs.
2. **The gate fires on 81% of can't-answer requests,** against 19% unpatched. It encodes something closer to "I cannot answer from what I have" than "this needs memory". Adding such negatives to the fit cut that rate to 12%, at a cost in recall (E004).
3. **The false-positive gain comes from a signed push.** On this trigger-happy model, clamping the gate's negative side raises dev FPR from .00 to .47. The patch works by pushing negatives away, not by staying silent on them (E008).

Pooled 5-fold CV over all hard data gives TF-IDF .980 against the best gate .930. The difference is −.050, with 95% CI [−.076, −.028]. The CPU tier reproduces the same pattern on Qwen2.5-0.5B:

| method (0.5B, tier 1) | dev AUROC | can't-answer FPR |
|---|---|---|
| few-shot prompt | .46 | .55 |
| rank-one patch | .98 | .50 |
| TF-IDF | .98 | n/a |

### 5.2 On a capable model, the ranking reverses (Qwen3-4B)

Qwen3-4B separates the classes unaided (AUROC .971 with only the schema) but under-calls (TPR .50). Prompts now work well, a logit bias overshoots (style2 FPR .55), and gating adds little. In E002, at dev selection, an unconditional steer reaches J .88 against .90–.95 for gated variants. Activations now tie with words: CV AUROC gate .973 vs TF-IDF .980, difference CI [−.024, .010].

**Table 1. Sealed test, single use, Qwen3-4B (E011).** 120 positives and 120 negatives from unseen domains, with every method frozen beforehand. These test domains are easier than the adversarial and style2 sets: the unpatched model already reaches AUROC .97.

| method | prompt tokens | TPR | FPR | used the answer (of 120) |
|---|---|---|---|---|
| no patch | 0 | .61 | .00 | .56 |
| few-shot prompt | 119 | .89 | .00 | .83 |
| rules prompt | ~70 | .99 | .10 | **.93** |
| TFTV (rank-one) | 0 | .88 | .05 | .83 |
| Steer2Edit (rank-one) | 0 | .95 | .08 | .88 |
| gated rank-one (DKE) | 0 | .88 | .00 | **.11** |
| CF-LoRA r16, no labels | 0 | .71 | .00 | .65 |
| RRM r4, no labels, hook | 0 | .94 | .00 | **.87** |
| TF-IDF (decision only) | n/a | .89 | .05 | n/a |

Three readings of Table 1:
- **RRM matches the prompt it imitates** at zero prompt tokens: .87 vs .83 end-to-end, inside noise at n = 120. On harder non-test sets it trails the prompt on adversarial (.48/.13 vs .68/.09) and matches it on style2 and contrast (E007).
- **The conditional rank-one edit decides well but breaks generation.** It remains active while the call is written, and execution falls from .98 called to .48 executed. Unconditional published edits do not have this failure.
- **CF-LoRA keeps everything else intact**, with no MMLU change (E010: .7175 → .71, p = .375, and identical greedy generations on 30 prompts). It recovers less of the prompt's end-to-end gain over no patch: 58% on 150 non-test positives ((.49−.18)/(.71−.18)) and 33% on test ((.65−.56)/(.83−.56)).

### 5.3 The prompt's effect is state-dependent and low-rank (E007, E010, tier 1)

**Table 2.** Held-out R² for predicting the prompt-induced change at the decision token, from unlabelled utterance pairs.

| model, input | constant shift | rank 1 | rank 4 | rank 16 |
|---|---|---|---|---|
| Qwen3-4B, residual, L28 / L34 | −.02 / −.03 | .19 / .21 | .37 / .40 | .51 / .50 |
| Qwen3-4B, MLP keys, L21 (E010) | | .03 | .52 | .67 |
| Qwen2.5-0.5B, residual, L14 / L19 / L21 (CPU tier) | −.02 / −.03 / −.03 | .15 / .14 / .15 | .32 / .36 / .38 | .48 / .54 / .53 |

A constant steering vector is the best constant approximation to this change, and it explains none of its variance across inputs. A low-rank linear function of the state explains a third to two thirds. This agrees with ICC's observation that fixed task vectors recover only part of in-context gains. It also gives a mechanistic reason why steering underperforms prompting: steering applies the average effect, while the prompt applies an input-specific one.

### 5.4 Many behaviours at once (E006, CLINC150)

Each of K intents triggers a distinct code word as the reply's first token. We compare all K rules in the prompt, a router that inserts one rule, K-rank patches, and a linear probe on the same keys.

| K | all rules: J (tokens, s/prompt) | router + 1 rule | best patch | linear probe |
|---|---|---|---|---|
| 1 | .30–.59 (18, .007) | .00–.80 | .80–.99 | .60–1.0 |
| 4 | .39–.81 (75, .012) | .32–.47 | .57–.81 | .41–.77 |
| 16 | .14–.35 (291, .030) | .35–.52 | .43–.48 | .56–.68 |
| 64 | −.05 (1154, .132) | .46 | .31 | .56–.62 |

J is correct minus false positives. The "best patch" is selected after the fact from 6 variants, which flatters it.

- **Prompts:** degrade quickly and cost 39× more per prompt at K = 64.
- **Patches:** win at K ≤ 4, then fall behind the probe built on the same features. The gate holds up but the value path interferes, consistent with H13.
- **Capacity:** our zero-forcing capacity prediction was wrong as stated. The participation ratio (3–34) is far below K = 140, yet the measured noise gain is only 1.6–2.2.

### 5.5 Limits (E009)

Without the tool schema in context, no patch produced a valid `recall_memory` call: 0 of 90 in both arms. With a value vector that contained the schema, the model degenerated into repeated `<tool_call>` tokens. Without it, the model never called at all. A patch can select among behaviours the context makes available. In these tests it did not supply the missing interface.

### 5.6 Negative results kept

- **LCMV zero-mean gate:** worse than LDA (E002, E004).
- **Label-free compile from model-written utterances:** that data was too easy to calibrate strength, so selection tied and chose a near-no-op (E007).
- **Logit bias:** over-triggers (E003).
- **Merged-weight edits:** bf16 inference arithmetic, not weight rounding, causes most of the hook-to-edit gap; simulated Q4 destroys the base model before any patch is applied (E008).

## 6. Cost

- **Fit time:** seconds of closed-form solving after a few hundred forward passes (E003: 1.5–2.2 s solve; 3.8–8.8 s of forward passes).
- **Patch size:** 21–25 KB for a rank-one patch and 0.4 MB for CF-LoRA r16 (fp16).
- **Inference cost:** no added prompt tokens. CF-LoRA adds rank-16 multiplies at one layer.
- **Prompt cost:** 70–119 tokens per request for one behaviour, and 1,154 tokens for 64 rules.

## 7. Limitations

- **Models:** two model families at small scale. No 7B+ result.
- **Behaviour:** one behaviour in depth, plus one synthetic many-behaviour task.
- **Data:** hand-written, with author style a known confound. The test split proved easier than the adversarial sets.
- **Seeds and samples:** single seeds for most runs. End-to-end differences at n = 120 below about 0.08 are not resolvable.
- **Baselines:** TFTV and Steer2Edit are single-module reimplementations with grid-edge α.
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
