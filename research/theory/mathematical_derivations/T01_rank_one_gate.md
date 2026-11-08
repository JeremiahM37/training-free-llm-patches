# T01. A conditional steering intervention is already a rank-one weight edit

Status: derivation (the algebra is exact; every empirical claim below is still a prediction). Written before any patch experiment.

## 1. Identity

A Qwen2-style block computes `h_out = h + W a`, where `a = σ(W_g x) ⊙ (W_u x) ∈ R^m` and `x` is the normalized post-attention residual and `W ∈ R^{d×m}` is `down_proj`, with no bias. Editing `W' = W + Δ vᵀ` gives

    h'_out = h_out + Δ · (vᵀ a(h))

This is the assignment's `h' = h + g(x,h)Δ` with `g = vᵀa(h)`. Consequences:
- The gate needs no separate classifier and no hook. It reads a nonlinear feature vector the model computes anyway, and `m >> d` (8960 versus 1536 in the 1.5B model).
- It is exactly a rank-one LoRA on `down_proj`, so any engine with LoRA support runs it unmodified.
- Storage: `(d+m)` values per rank. For Qwen2.5-1.5B that is 10,496 values, 21 KB at fp16. Extra work per token: `d+m` multiplications per patched layer, against about `2·d·m` for that block's MLP matrices alone.

Known relatives: ROME has this exact form (`W' = W + Λ(C⁻¹k)ᵀ`) but picks one key for one fact and fits `Λ` by gradient descent. CAST gates by cosine with a condition vector through a hook and threshold. Whether the population version below exists is a novelty question for the literature pass.

## 2. Gate as a linearly constrained minimum-variance filter

There is no bias term, so a gate `vᵀa` cannot subtract a threshold. It must respond with 0 on background and 1 on the trigger using only the direction of `v`. Let `R = E[a aᵀ]`, the **uncentered** second moment over all token positions in background text, including generated tokens. The perturbation energy injected on background is `‖Δ‖² vᵀRv`, which is the collateral-damage term. So:

    minimize vᵀRv   subject to   Mᵀv = g

where the columns of `M` are trigger key means (and optional nulls), and `g` holds the wanted responses. The closed form is the LCMV/MVDR beamformer from array signal processing:

    v* = R⁻¹M (MᵀR⁻¹M)⁻¹ g

With `n` background samples `X ∈ R^{n×m}` and `R_λ = λI + XᵀX/n`, Woodbury gives an `O(n³ + nm)` solve with no `m×m` matrix.

Single constraint `M=[μ₊], g=1`: `v ∝ R⁻¹μ₊`, close to LDA but with no separate threshold. Two constraints `M=[μ₊, μ₋], g=(1,0)` force zero mean response on the negative class, standing in for the missing bias.

## 3. Composition and a capacity prediction

For `K` patches, set `M = [μ₁…μ_K]`, `G = I`, `V = R⁻¹M(MᵀR⁻¹M)⁻¹`. Then `μ_jᵀ v_i = δ_ij`: mean crosstalk at the gate is exactly zero by construction, not hoped for. This holds only for the means placed in `M`. With centred means, every input also gets the offset `v_iᵀμ_all`, and with no bias in the weight that offset is always on, so the uncentred response is not `δ_ij`. The price is noise gain.

In whitened coordinates `ṽ = R^{1/2}v`, `vᵢᵀRvᵢ = [(MᵀR⁻¹M)⁻¹]_ii`. For iid isotropic real keys in `r` effective dimensions, `(MᵀR⁻¹M)` is Wishart, and the inverse-Wishart mean gives:

    vᵢᵀRvᵢ ≈ (1/‖μ̃‖²) · r/(r − K − 1)

(Corrected 2026-11-03. The first version had `r − K + 1`, a mistake caught by the critic and confirmed by simulation: r=200, K=190 gives 22.7, this formula 22.2, the old one 18.2.)

**Prediction P1.** Gate noise, and with it collateral damage, stays nearly flat for `K << r` and diverges as `K → r−1`, where `r` is the effective rank of the key distribution (measure it as the participation ratio of `R`, not as `m`). Real behaviour keys are correlated, so the practical limit should sit below `r`.
**Prediction P2.** Off-diagonal entries of `MᵀR⁻¹M` predict which pairs interfere before anything is run.
**Prediction P3.** Sequentially adding patches without joint re-solving degrades with `K` where joint solving does not.

Gate crosstalk is only half the story. Value directions `Δᵢ` act through later nonlinear layers and can interfere regardless. This analysis says nothing about that term, which has to be measured.

## 4. What this cannot do, by construction

- A gate linear in `a` can only fire on what is linearly decodable from `a`. If relevance is not decodable at any layer, the method fails for that reason. The cheapest test is the held-out AUROC of `vᵀa`.
- `Δ` selects among behaviours the downstream layers can already produce. It adds no capability those layers lack.
- The mean constraint gives no worst-case guarantee on individual inputs. False positives are the tail of `vᵀa`, not its mean.
- A patch tuned at one position must not disturb the rest of the generated sequence. This depends on background `R` covering generated tokens.

## 5. Falsifiers

1. Held-out gate AUROC at the best layer ≤ the prompt's AUROC by more than 0.05 → gate too weak.
2. The rank-one edit cannot match the hook version of the same intervention (would mean an algebra or implementation bug, since section 1 is an identity).
3. Any `Δ` scale that reaches a TPR of the prompt's level also breaks unrelated capability by more than prompts do.
4. A trivial alternative (keyword regex plus a prompt) matches it.
