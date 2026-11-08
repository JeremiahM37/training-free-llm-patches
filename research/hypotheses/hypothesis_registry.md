# Hypothesis registry

Status values: `active`, `supported` (by named experiments), `rejected`, `parked`. Evidence labels: **M** measured here, **L** reported in a paper we read, **D** deduction, **S** speculation.
Novelty values describe the result of searching so far, never a final claim: `known`, `combination`, `open` (not found yet, search incomplete).

Each entry gives: mechanism; motivation; support; contradiction; what differs from prior work; prediction; failure condition; difficulty; cheapest falsifier.

## H01. Discriminant-keyed rank-one edit (DKE). `active`, novelty `known` in form (2604.12359 SteerEdit, see D010)
- **Mechanism:** `W' = W + αΔvᵀ` on `down_proj`, `v = R⁻¹(μ₊−μ₋)`, normalized to unit gain; `Δ` from a behaviour contrast fitted separately. Math: T01.
- **Motivation:** a gate from the model's own MLP features, no hook, runs as a LoRA.
- **Support:** M E002, E003. Rank-one LDA has the best J of the methods tried on the 1.5B model. Saturates at about 10 examples per class.
- **Contradiction:** M E003 fires on can't-answer inputs; M E004 loses to TF-IDF. L 2406.11717, 2609.09054, 2602.09870 already make rank-one edits from steering vectors.
- **Difference:** published right factors are `sign(μ'v_i)σ_i v_i` (TFTV), `W'v` (Steer2Edit), both built around the steering direction or expected input. Here the right factor comes from a labelled trigger contrast, decoupling when from what. CAST has that split as a hook with threshold; PSR has the form but trains it.
- **Prediction:** beats TFTV and Steer2Edit on FPR at equal TPR on the same task.
- **Failure:** E005 shows prior edits equal it, or the logit-bias baseline does.
- **Difficulty:** done. **Falsifier:** E005.

## H02. LCMV zero-mean gate. `rejected`
The T01 section 2 constraint `vᵀμ₋=0`. **M E002, E004:** worse than plain LDA at most layers. Likely reason (S): a signed gate pushes negatives away from calling, which a zero-mean gate gives up. Kept as a recorded negative.

## H03. Zero-forcing capacity law. `rejected as stated` by E006 (participation ratio is not the effective r; gain at K=140 only 1.6-2.2), novelty `open` (closest found: 2604.12359, 2410.02355, 2306.03819, to read)
- **Mechanism:** joint solve `V=R⁻¹M(MᵀR⁻¹M)⁻¹`. Gate noise scales about `r/(r−K−1)` (corrected) with `r` the participation ratio of `R`.
- **Motivation:** array processing result, T01 section 3.
- **Support:** D only. L UCE reports an empirical capacity limit near 500 concepts in a different architecture.
- **Contradiction:** real keys are not isotropic. L 2608.09490 and 2610.03199 find interference driven by other geometry.
- **Difference:** published predictors (2505.11611, 2610.03199, 2608.09490) work on SAE features or trained adapters, not a closed-form gate capacity.
- **Prediction:** `gate_energy` flat for K≪r, rising near r.
- **Failure:** energy curve does not track `r/(r−K+1)`, or behaviour fails long before gate noise rises.
- **Difficulty:** low. **Falsifier:** E006.

## H04. Joint solve beats independent patches. `active`, novelty `known` in spirit
Multi-output least squares against separate fits. **Prediction:** gap widens with K. **Failure:** no gap at K=150. **Falsifier:** E006.

## H05. Patches scale past prompts. `partly supported` by E006 (beat prompts at every K≥4, but lose to router and probe past K≈16), novelty `combination`
- **Mechanism:** K rules cost K×about 25 tokens in a prompt, 0 tokens as patches.
- **Motivation:** the only regime where tokenless patches should clearly matter.
- **Contradiction:** D a router plus one retrieved rule also avoids token growth. If it wins, a patch is a worse skills system.
- **Prediction:** at K≥64, patches beat `prompt_all` on correctness or latency.
- **Failure:** `prompt_rag` dominates at all K. Then this reduces to retrieval.
- **Difficulty:** low. **Falsifier:** E006.

## H06. A logit bias captures most of the gain on a capable model. `rejected` by E003 on Qwen3-4B, novelty `known`
- **Mechanism:** constant bias on the `<tool_call>` logit at the decision position.
- **Support:** M Qwen3-4B schema-only AUROC .971 with TPR .50, FPR .00.
- **Prediction:** threshold at dev reaches J within .05 of the patch on Qwen3-4B.
- **Failure:** patch beats it on adversarial or style2 sets. **Falsifier:** E003 on Qwen3.

## H07. Reduced-rank compile of a prompt. `active`, novelty `known` (see In-Context Compilation, doi:10.18653/2026.findings-emnlp.983, per critique_01)
- **Mechanism:** ridge reduced-rank regression from unprompted to prompted activations: `h' = h + UVᵀh + b`, fit from paired forward passes.
- **Motivation:** a prompt's effect is a function of `h`; its best low-rank linear approximation has a closed form.
- **Support:** L 2605.03907 shows rank-1 gated mimicry works when trained. L 2605.10664 duality.
- **Contradiction:** L 2605.03907 rank-1 not expressive for all instructions. Different token positions are not aligned across prompted and unprompted runs.
- **Difference:** low-rank regression fit instead of sparse calibration. E007 tests it as a runtime hook at the decision token, not a weight edit, so it measures explainable variance only.
- **Prediction:** at the decision token, explains more variance than a constant shift.
- **Failure:** no more than the mean shift `dH`. **Falsifier:** E007.

## H08. Self-compile: a patch from the instruction alone. `failed as implemented` (E007; selection tied on easy generated data), novelty `open`
- **Mechanism:** run the model with the instruction over unlabeled utterances, take its own decisions as pseudo-labels, fit DKE to them. No human labels, no gradient.
- **Motivation:** assignment objective A: instruction to patch. Distills the prompt's decision boundary into a tokenless patch.
- **Support:** M Qwen3-4B few-shot prompt is accurate (dev AUROC .993), so its labels are worth copying.
- **Contradiction:** needs a capable model; garbage labels on 1.5B. Needs some unlabeled pool. Distillation is an old idea (`combination` risk).
- **Difference:** L 2410.00487 distills context to weights with gradients. L 2608.00218 labels from rollouts but uses a separate probe.
- **Prediction:** patch matches the prompt within .05 J on Qwen3-4B with zero labels, at 0 prompt tokens.
- **Failure:** large gap or a large drop on style2. **Difficulty:** low. **Falsifier:** E007.

## H09. A shared "unresolved reference" signal. `active`, novelty `known` in part
M E003 gate fires on can't-answer inputs, M E004 fixed by hard negatives at a TPR cost. L 2608.00218 reports separable over-calling and missing signals. **Failure:** a linear gate cannot separate memory-need from can't-answer on unseen kinds. **Falsifier:** E004 partly done (capability .55-.64, missing_input .35-.68 with unseen kinds).

## H10. The signed gate is what makes DKE work. `supported` by E008 on the 1.5B model, novelty `open`, minor
Clamp negative gate values; if the signed edge is the cause, J falls. Reduces to a small ablation. **Falsifier:** E008.

## H11. Last-block patches reduce to a linear classifier. `active`, novelty `known`
**D** near the output, `Δvᵀa` is a linear readout into one logit. **Prediction:** patch at the last layer ≈ probe. **Failure:** patch beats probe ceiling. **Falsifier:** E006 `probe` rows.

## H12. Merged edits are limited by weight precision. `partly rejected` by E008 (weight rounding small at bf16; inference arithmetic mattered), novelty `known` in part
M E003 hook and bf16 edit differ by 0.75 logits on a 5.9 effect. **Prediction:** the difference vanishes in float32 and worsens in 8-bit. **Falsifier:** E008, one forward pass per dtype.

## H13. Value directions, not gates, set the capacity. `supported` by E006 (probe on same keys keeps J .56-.68 where patches fall to .31-.48), novelty `open`
**S** even with gates perfectly nulled, `Δ`s interact through later layers. **Prediction:** at the last layer, capacity matches gate theory; at earlier layers behaviour degrades sooner. **Falsifier:** E006, compare `L(nl-1)` with `L(nl-4)`.

## H14. Patch transfer by closed-form alignment. `parked`, novelty `known` in part
Procrustes between models on shared prompts. A paper titled as functional transfer (2610.32581) is in the harvest, title only, not read. Parked: low expected value until H01 matters.

## H15. Set-point control instead of additive steering. `parked`, novelty `known`
L 2510.04309, 2604.19018, PID papers. Requires a hook, losing the plain-LoRA property.

## H16. A patch cannot supply what the schema supplied. `supported` by E009, novelty `combination`
- **Mechanism:** remove the tool schema; patch still pushes `<tool_call>`.
- **Motivation:** assignment objective F: separating selection from capability.
- **Prediction:** decision moves, but tool name and format are hallucinated. Without schema, correct calls are near zero unless the value carries the name.
- **Failure:** patch-only yields correct calls. That would mean the name can be stored cheaply in `Δ`, which is a larger positive result.
- **Difficulty:** low. **Falsifier:** E009.

## H17. Gate generalizes across writers better than text. `active`, novelty `open`
**Support:** M E004 unseen-kind wins on some kinds. **Contradiction:** M E004 overall loss. **Prediction:** on style2, activation gate beats TF-IDF. **Failure:** no difference. **Falsifier:** E003 on Qwen3 (style2 in control).

## H18. Few-example saturation is general. `active`, novelty `known` in part
M E003 saturation at n≈10 per class on 1.5B. **Prediction:** same on Qwen3-4B. **Falsifier:** E003 curve on Qwen3.

## H19. Closed-form reduced-rank LoRA from prompt pairs (RR-LoRA). `partly supported` (E010, E011: about half to 40% of the prompt gain, no MMLU change), novelty `known` in core (ICC, findings-emnlp.983, read 2026-11-03: ridge from zero-shot to few-shot last-token activations, sparse rows of attention output projection)
- **Mechanism:** `W' = W + BᵀR` where `B` minimizes `‖Xb − Y‖² + λ‖B‖²` subject to rank r. `X`: MLP keys. `Y`: prompted-minus-unprompted residual at the decision token, and 0 at every other position. Solved in the dual, then truncated by SVD. Strength comes from the regression; no alpha.
- **Motivation:** M E007 shows a constant shift explains nothing and a low-rank map explains .40-.50. The zero targets on background should substitute for the hook's position logic, using only what the model already computes.
- **Support:** M E007 hook version. T01 identity.
- **Contradiction:** a linear map on keys may not separate decision tokens from others. Keys differ from `h`.
- **Difference from ICC:** low rank rather than sparse rows; MLP output rather than attention output; explicit zero targets on background positions; a tool-decision task rather than classification and reasoning. These are variations, not a new idea. D2L needs a trained hypernetwork. PSR is trained by gradient and rank-1.
- **Prediction:** at rank 4-16, within .05 J of the few-shot prompt on style2 and contrast, no detectable MMLU drop.
- **Failure:** needs the hook to work, or breaks generation after the decision token.
- **Falsifier:** E010.

## Selected for falsification now (value per cost)
1. **H05 + H03 + H13 (E006):** the one regime where tokenless patches could matter, tested on human-written data with a retrieval rival.
2. **H08 (E007):** directly tests instruction-to-patch with no labels.
3. **H16 (E009):** a clean limit result whichever way it goes.
