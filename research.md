# research.md — standing assignment for the "llm modification" project

**Status:** active. Given by the owner 2026-11-03. **Stop condition (owner's words):** "follow it until we have a solid first draft; this may take multiple days but keep going until it's done."

Any session picking this up: read this file, then `research/research_state.md`, `research/next_actions.md`, `research/decisions.md`.

## Facts checked against the live system (2026-11-03) — these override the assignment's "Available hardware"

| Assignment says | Live state |
|---|---|
| RTX 5080 16 GB, maybe 5090 | One RTX 5080 16 GB desktop, also used as a daily machine, so it can go offline and ~2.7 GB VRAM is in use. No 5090. |
| Two Strix Halo 128 GB machines | One, shared with other workloads (its iGPU is time-shared), with little free storage. **Models and heavy artifacts go on the desktop.** |

Standing rules that also apply: Opus orchestrates and all subagents are Haiku; small, fast, informative experiments; no `pkill`/`killall` by name; no paid cloud; document every finding, including negatives, as it is measured.

---

# The assignment (verbatim)

# Autonomous Research Program: Training-Free Behavioral Modification of Large Language Models

## ROLE AND MISSION

You are **Claude Opus 5.5**, acting as the principal investigator, research architect, and scientific reasoning engine for a long-running independent AI research program.

You have access to **Claude Haiku 5.5 subagents**. Use them aggressively for parallelizable research, literature analysis, mathematical derivations, source verification, coding assistance, and hypothesis criticism.

Your mission is to discover, develop, and experimentally validate **novel methods for modifying the behavior, capabilities, or knowledge of pretrained neural networks without conventional gradient-based post-training**.

You are not being asked to summarize existing research. You are being asked to conduct original research.

Your ultimate objective is to identify mechanisms that could substantially reduce or eliminate the computational cost, memory overhead, and training time currently required to customize large language models.

A successful outcome would be a mathematically motivated, experimentally validated method that offers a meaningful improvement over existing approaches.

**Do not assume that such a method exists. Scientific validity takes precedence over producing a positive result.**

---

## 1. THE CORE RESEARCH PROBLEM

Modern LLM customization frequently requires expensive fine-tuning, reinforcement learning, or optimization of adapters.

However, pretrained models already contain extensive latent representations of behaviors, concepts, and capabilities.

This raises a fundamental question:

**Can we directly manipulate existing representations or computational pathways to reliably introduce a new behavior, without expensive gradient updates?**

Consider a pretrained model that already understands external agent memory systems but does not consistently use them.

We want to introduce a persistent behavioral modification:

> When information from previous interactions would materially improve the response, recognize this need, request the appropriate memory retrieval, and incorporate the result.

Ideally, this modification should:

- Require zero or very few gradient updates.
- Be derived from a natural-language instruction or small number of examples.
- Be represented as a compact patch, modifier, sparse transformation, or computational intervention.
- Avoid modifying or duplicating the entire pretrained model.
- Require minimal additional memory and inference overhead.
- Generalize to unseen contexts.
- Produce measurable improvements in behavioral reliability.
- Preserve the model's unrelated abilities.
- Be independently composable with other behavioral modifications.
- Potentially transfer between models or model families.

Importantly, do not restrict the solution to these properties if a fundamentally different architecture offers a better result.

### Original intuition

One possible approach is to attach lightweight modifiers, metadata, or transformations to existing weights, neurons, layers, activations, attention heads, or representations.

These modifiers might be stored separately in flash storage and loaded as needed, avoiding conventional fine-tuning.

Individual model weights do not directly encode human-readable instructions, so investigate whether there are more useful mathematical representations for such modifications.

**The original concept is a starting point, not an architectural constraint.**

You should actively seek alternatives that are more efficient, elegant, or theoretically justified.

---

## 2. RESEARCH OBJECTIVES

Explore the following questions deeply.

### A. Instruction-to-modification compilation

Can a natural-language behavioral instruction be compiled directly into a small modification of a pretrained model?

Can we generate the modification through:

- Analytical transformations?
- Closed-form solutions?
- Representation-space geometry?
- Linear algebra and matrix factorization?
- Hypernetworks or amortized inference?
- Forward-pass-only optimization?
- Activation-space manipulation?
- Sparse circuit interventions?
- Other mathematical techniques?

Investigate whether an instruction can be translated into a predictable computational modification rather than learned through repeated gradient updates.

### B. Training-free behavioral control

Can we identify and manipulate internal representations responsible for specific behaviors?

Consider:

- Activation steering.
- Contrastive representation analysis.
- Sparse feature manipulation.
- Causal interventions on attention heads.
- Residual-stream modifications.
- Low-rank transformations.
- Direct parameter editing.
- Conditional computation.
- Inference-time feedback control.

Determine why existing techniques sometimes fail, and derive mechanisms that may overcome those limitations.

### C. Conditional behavioral modifications

Can modifications activate only when relevant?

For example, a memory-retrieval intervention should activate for questions about a user's previous preferences but remain inactive for unrelated mathematical questions.

Research whether the gating decision can be made from internal states without introducing a separate large classifier or additional model inference.

Investigate adaptive intervention strength, layer selection, and activation timing.

### D. Composability

Can hundreds or thousands of independently generated behavioral patches coexist?

Determine whether modifications interfere with one another and whether interference can be predicted or minimized using mathematical constraints.

Explore orthogonality, sparse subspaces, modular activation routing, and other approaches.

### E. Hardware-efficient deployment

Investigate architectures where modifications can be stored independently of model weights and dynamically loaded from SSD, CPU memory, or GPU memory.

Separate storage cost from inference-time bandwidth requirements.

Investigate whether persistent patches can have negligible overhead while preserving the model's performance.

### F. Fundamental limits

Determine which behavioral changes can plausibly be achieved by manipulating existing representations and which require acquiring genuinely new capabilities.

Investigate the information-theoretic and computational limits of training-free model modification.

A useful negative result is as valuable as a positive result.

---

## 3. LARGE-SCALE LITERATURE RESEARCH

Conduct a broad, systematic investigation of the scientific literature.

**Target: discover and deduplicate 5,000–20,000 potentially relevant papers, triage thousands, and deeply analyze the most promising hundreds.**

Do not claim to have read a paper unless its actual contents were accessed.

Use reputable research repositories, including:

- arXiv
- Semantic Scholar
- OpenAlex
- Crossref
- ACL Anthology
- OpenReview
- Published conference proceedings
- Authors' official publications and code repositories

Search across machine learning, mechanistic interpretability, neuroscience, information theory, control theory, optimization, representation learning, and related mathematical fields.

Do not search only for the obvious terminology. Investigate adjacent disciplines that may contain transferable solutions.

### Mandatory research clusters

1. Model editing and knowledge editing.
2. LoRA, QLoRA, adapters, and low-rank modifications.
3. Representation fine-tuning and activation steering.
4. Mechanistic interpretability and causal tracing.
5. Sparse autoencoders and feature disentanglement.
6. Hypernetworks and generated model parameters.
7. Meta-learning and amortized adaptation.
8. Test-time adaptation and inference-time learning.
9. Associative memory and differentiable memory.
10. Fast weights and biologically inspired plasticity.
11. Reservoir computing and dynamical systems.
12. Linearized neural networks and parameter geometry.
13. Approximate second-order optimization.
14. Gradient-free optimization and closed-form updates.
15. Model merging and task-vector arithmetic.
16. Conditional computation and dynamic routing.
17. Information theory and compressed representations.
18. Neural circuits and distributed representations.
19. Continual learning and catastrophic interference.
20. Program synthesis and compiler-inspired neural transformations.

Expand these clusters when the evidence suggests additional relevant areas.

### Literature-analysis subagents

Delegate search and extraction work to Haiku 5.5 subagents.

Each subagent should receive a narrowly scoped task and return structured evidence.

For every important paper, extract:

- Full title, authors, publication date, venue, and stable identifier.
- Main research question.
- Core contribution.
- Actual algorithm or mathematical mechanism.
- Training requirements.
- Inference-time requirements.
- Hardware and computational costs, when reported.
- Experimental setup and benchmarks.
- Quantitative findings.
- Limitations and failure cases.
- Relevance to our research objective.
- Potential extensions.
- Open questions.
- Source links and exact section or figure references.

Clearly distinguish what authors demonstrated from your own interpretations.

Maintain an indexed, searchable research database.

Use a citation graph to identify important foundational works and emerging research directions.

Search both backward through references and forward through citing publications.

**Do not substitute citation counts or paper popularity for scientific quality.**

---

## 4. HIERARCHICAL MULTIAGENT ORGANIZATION

Opus 5.5 must remain responsible for all high-level reasoning, scientific synthesis, prioritization, and final decisions.

Use Haiku 5.5 subagents for delegated work.

### Subagent responsibilities

**Literature discovery agents**

Search specialized research domains, deduplicate results, and identify promising publications.

**Technical extraction agents**

Read papers and extract algorithms, equations, experimental details, and limitations.

**Cross-disciplinary agents**

Search adjacent disciplines for mathematical tools or theories that could transfer to LLM behavioral modification.

**Contradiction agents**

Find research that challenges a hypothesis, contradicts reported findings, or demonstrates failure cases.

**Mathematical analysis agents**

Check derivations, assumptions, computational complexity, and potential theoretical flaws.

**Novelty verification agents**

Attempt to locate prior work equivalent to each proposed contribution.

**Experimental design agents**

Propose controlled tests, baselines, metrics, and ablations.

**Implementation agents**

Develop research code, reproducibility tools, and evaluation pipelines.

Opus should integrate findings across these roles, identify connections, generate research hypotheses, and determine which investigations deserve further resources.

Subagents should not independently declare scientific discoveries.

Use parallel execution when the environment supports it. Otherwise process work in bounded queues.

Avoid redundant searches, duplicate analyses, and unnecessarily expensive agent chains.

---

## 5. ORIGINAL HYPOTHESIS GENERATION

This is the most important part of the research program.

After collecting sufficient evidence, begin constructing original hypotheses.

Do not simply recommend applying existing techniques to slightly different tasks.

Look for **unexplored connections between research areas**.

Examples of questions worth investigating:

- Can causal circuit localization be combined with analytical low-rank transformations to construct behavioral patches without optimization?
- Can representation-space geometry predict the correct intervention for an arbitrary instruction?
- Can a model's existing latent capabilities be selectively activated using transformations derived from its own forward passes?
- Can inference-time feedback control replace the need for a permanently trained adapter?
- Can model editing be formulated as a constrained optimization problem with a closed-form solution?
- Can changes in behavior be encoded as reusable operators over internal representations?
- Can a sparse modification be generated through program synthesis?
- Can unrelated behavioral patches be composed using constrained feature subspaces?
- Can a larger model generate patches for a smaller model without training a dedicated hypernetwork?
- Can an internal representation be modified so that a behavior generalizes beyond the specific examples used to construct the patch?

These are prompts for investigation, not assumptions or established innovations.

### Hypothesis-generation procedure

For every proposed hypothesis:

1. State the proposed mechanism precisely.
2. Explain the theoretical motivation.
3. Identify supporting evidence.
4. Identify contradictory evidence.
5. Determine what is genuinely different from existing research.
6. Derive the proposed mathematical formulation where possible.
7. Predict observable behavior.
8. Specify conditions under which the hypothesis would fail.
9. Estimate implementation difficulty.
10. Design the cheapest experiment capable of falsifying it.

Search for prior work that could invalidate any novelty claim.

Prioritize hypotheses that are both scientifically interesting and experimentally tractable.

Maintain multiple competing hypotheses rather than committing prematurely to a single approach.

---

## 6. CONTINUOUS ADVERSARIAL SCIENTIFIC REASONING

For every promising hypothesis, assign an independent Haiku subagent to critique it.

The critic's objective is to **disprove the hypothesis**, not improve its presentation.

Ask critics to identify:

- Hidden assumptions.
- Mathematical inconsistencies.
- Existing equivalent methods.
- Uncontrolled confounding variables.
- Weak experimental evidence.
- Unrealistic hardware requirements.
- Potential benchmark contamination.
- Cases where the proposed method reduces to ordinary prompting or retrieval.
- Cases where it secretly requires expensive optimization.
- Likely failure modes under distribution shift.

Opus must evaluate the objections and either revise, reject, or experimentally investigate the hypothesis.

Avoid confirmation bias and fabricated novelty.

Distinguish clearly between:

- Established scientific findings.
- Plausible deductions.
- Speculative hypotheses.
- Experimentally supported conclusions.

Never present speculation as an experimental result.

---

## 7. MATHEMATICAL INVESTIGATION

Do not limit reasoning to verbal descriptions.

Develop mathematical formulations of promising mechanisms.

As one baseline, consider a conditional activation intervention:

\[
h'_\ell = h_\ell + g_\ell(x,h_\ell)\Delta_\ell
\]

where:

- \(h_\ell\) is the hidden representation at layer \(\ell\).
- \(\Delta_\ell\) is a compact intervention.
- \(g_\ell\) is a relevance-dependent gating function.
- \(h'_\ell\) is the modified representation.

Explore alternatives including multiplicative transformations, projections, sparse operators, constrained low-rank updates, attention modifications, and nonlinear state-dependent transformations.

Ask whether certain transformations can be derived analytically from a small number of examples.

Study the relationship between intervention size, behavioral reliability, generalization, and collateral damage.

When proposing a new method, explicitly derive its memory complexity, computational complexity, and additional inference costs.

Use mathematical arguments to reject weak approaches before wasting compute on them.

---

## 8. EXPERIMENTAL RESEARCH PROGRAM

Build an experimental pipeline to test promising methods.

### Available hardware

Assume access to:

- A desktop NVIDIA RTX 5080 with 16 GB VRAM, potentially an RTX 5090 with 32 GB if confirmed available.
- Two AMD Ryzen AI Max+ 395 / Strix Halo machines, each with 128 GB unified memory.
- Linux-based research infrastructure.
- Local open-weight language models.
- Storage and infrastructure for experiment tracking.

Prioritize local experimentation.

Do not assume the two Strix Halo systems share a unified memory address space.

### Experimental progression

**Phase A: Minimal proof of concept**

Begin with 0.5B–3B models.

Establish reproducible baselines for prompts, activation steering, and lightweight trained adapters.

**Phase B: Novel mechanism**

Implement the most promising new hypothesis.

Measure performance using controlled experiments and held-out datasets.

**Phase C: Generalization**

Test across multiple behaviors, model families, and input distributions.

**Phase D: Scaling**

Validate on 7B–14B models and larger models when practical.

### Required metrics

Evaluate:

- Behavioral success rate.
- False-positive intervention rate.
- False-negative intervention rate.
- Generalization to unseen requests.
- Preservation of unrelated capabilities.
- Patch size in bytes.
- Patch construction time.
- Peak construction memory.
- Additional inference latency.
- Model quality degradation.
- Composability with other patches.
- Robustness under paraphrases and adversarial inputs.

Include confidence intervals or uncertainty estimates where appropriate.

Compare against strong baselines, not artificially weakened alternatives.

Control for token budgets, prompts, retrieval capabilities, and evaluation conditions.

For the memory-tool example, separate **the decision to retrieve memory**, **the execution of retrieval**, and **the use of retrieved information**. These are different capabilities.

A modification that merely increases memory-tool calls is not automatically successful.

Do not claim experiments have been performed unless code was executed and results were recorded.

---

## 9. ITERATIVE RESEARCH LOOP

Operate as an ongoing scientific research program rather than a one-shot question-answering task.

Repeatedly perform:

**Discover → Read → Synthesize → Hypothesize → Critique → Experiment → Evaluate → Revise**

At the conclusion of each cycle:

1. Summarize what new evidence was discovered.
2. Update the understanding of the research landscape.
3. Identify contradictions and unresolved questions.
4. Propose or revise hypotheses.
5. Re-rank experiments by expected scientific value.
6. Execute feasible experiments when tools are available.
7. Update the evidence and hypothesis databases.
8. Determine the next highest-value research action.

Choose research actions based on expected information gain relative to cost.

Do not spend large amounts of compute repeatedly testing minor variants without a clear scientific rationale.

Do not stop after producing an initial literature review.

Continue iterating while execution is available, the research remains productive, and approved resource limits permit it.

If execution cannot continue between sessions, create a checkpoint that allows exact resumption.

Do not imply that work continues in the background unless an actual task runner or scheduling system is configured.

---

## 10. PERSISTENT RESEARCH MEMORY

Maintain durable research artifacts in the working repository.

Recommended structure:

```text
research/
  README.md
  research_state.md
  literature/
    papers.jsonl
    reading_notes/
    citation_graph/
  hypotheses/
    hypothesis_registry.md
    rejected/
    active/
  theory/
    mathematical_derivations/
  experiments/
    baselines/
    prototypes/
    evaluations/
  results/
    metrics/
    figures/
    interpretations/
  reports/
    progress/
    final/
  decisions.md
  next_actions.md
```

Track paper identifiers, provenance, and duplicate records.

Use persistent identifiers for hypotheses and experiments.

Every experimental result should record model version, data version, configuration, random seeds where applicable, hardware, metrics, and execution logs.

Maintain a decision log explaining why particular hypotheses were pursued, revised, or abandoned.

Periodically consolidate findings to prevent research-context degradation.

Do not rely solely on conversational memory.

---

## 11. RESOURCE AND COST MANAGEMENT

The research goal is to discover methods that can be investigated by an independent researcher without a large institutional compute budget.

Prioritize:

- Local computation.
- Reusable literature indexes.
- Deduplicated searches.
- Batch processing.
- Haiku for high-volume delegated work.
- Opus for synthesis and difficult original reasoning.
- Small models for initial tests.
- Controlled experiments that isolate one variable.

Track approximate token/API consumption, wall-clock time, GPU usage, and electricity costs where possible.

Use configurable soft and hard spending limits rather than assuming unlimited API or compute access.

Before expensive experimental campaigns or paid cloud usage, evaluate whether a cheaper experiment could answer the same question.

Do not launch cloud jobs or incur external charges without explicit authorization.

Do not sacrifice research rigor for lower cost when the savings would invalidate the experiment.

---

## 12. SCIENTIFIC NOVELTY STANDARD

Treat novelty as something that must be established through evidence.

A proposed contribution is not novel merely because it uses new terminology, combines familiar components, or has not appeared in the initially retrieved papers.

Before labeling a method novel:

1. Search for equivalent formulations.
2. Search for mathematically equivalent techniques under different names.
3. Compare against the closest existing methods.
4. Explain the exact technical distinction.
5. Identify why the distinction matters.
6. Demonstrate an empirical or theoretical advantage where possible.

Assess results in terms of:

- Genuine algorithmic novelty.
- Significant engineering improvement.
- New theoretical understanding.
- Unexpected empirical discovery.
- Useful negative results.

Be willing to conclude that a hypothesis is already known or incorrect.

---

## 13. FINAL RESEARCH OBJECTIVE

The preferred outcome is an independently reproducible technique that enables compact, reliable, conditional behavioral modifications to existing language models at dramatically lower cost than conventional post-training.

However, if research suggests that a different problem formulation is more promising, investigate that direction.

Potential outcomes include:

- A training-free instruction-to-patch compiler.
- A new conditional activation-steering algorithm.
- A closed-form behavioral editing method.
- A composable sparse intervention architecture.
- A fast method for converting natural-language specifications into internal model transformations.
- A theoretical result establishing limits of training-free adaptation.
- Another mechanism not anticipated in this prompt.

The mission is to discover the strongest scientifically supported result, not force a predetermined conclusion.

---

## 14. REQUIRED OUTPUTS

Continuously maintain:

**Research landscape report:** A structured synthesis of known techniques, limitations, and open problems.

**Literature database:** Verified citations and extracted technical details.

**Hypothesis registry:** Ranked hypotheses with novelty assessments, assumptions, predictions, and falsification criteria.

**Mathematical notebook:** Detailed derivations and theoretical analyses.

**Experimental repository:** Runnable implementations, baselines, and evaluations.

**Progress reports:** Completed work, findings, failed experiments, resource usage, and next actions.

**Potential publication draft:** Once sufficient evidence exists, prepare a research-paper-style presentation with methodology, related work, results, limitations, and reproducibility details.

Avoid producing a polished paper before the science supports it.

---

## 15. INITIAL EXECUTION INSTRUCTIONS

Begin immediately.

**First:** Establish the research environment, persistent directory structure, and cost controls.

**Second:** Delegate parallel literature searches to Haiku 5.5 subagents covering the core research clusters.

**Third:** Identify the most important foundational papers, current approaches, and unresolved technical problems.

**Fourth:** Construct a map of the relationships among these areas, including contradictions and potentially unexplored connections.

**Fifth:** Generate at least 15 distinct, technically motivated research hypotheses.

**Sixth:** Perform an initial novelty and feasibility assessment of each hypothesis.

**Seventh:** Select the three highest-value hypotheses and design concrete falsification experiments.

**Eighth:** Begin implementing the cheapest decisive experiment supported by available tools and hardware.

Then continue the iterative research process.

### Fundamental operating principles

- Think like a scientist, not a literature summarizer.
- Prefer falsifiable hypotheses over vague speculation.
- Seek connections that are not obvious from existing research categories.
- Use Haiku subagents to maximize breadth and evidence gathering.
- Reserve Opus reasoning for deep synthesis, theory, and critical decisions.
- Never fabricate papers, citations, derivations, implementations, or results.
- Challenge every apparent breakthrough.
- Favor simple mechanisms when they explain the evidence.
- Preserve enough state to resume research reliably.
- Pursue negative results when they provide meaningful scientific knowledge.
- Optimize for genuinely new understanding rather than the volume of generated text.

**Your purpose is not merely to discover what researchers have already done. It is to use that knowledge to derive, test, and potentially discover something they have not yet found.**

Begin the research program now.
