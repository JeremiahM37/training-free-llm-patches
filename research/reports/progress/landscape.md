# Research landscape (2026-11-03)

Basis: this report used the first 18.5k deduplicated records (arXiv and part of OpenAlex; later harvests raised the corpus to 30.5k, not yet triaged, see D014), 17.4k triaged by embedding similarity with per-cluster quotas, and 125 papers read in extraction packets by Haiku agents plus 3 read directly (Steer2Edit, SteerEdit, ICC). Every extraction quote was checked mechanically against the text. Numbers below are what authors report unless marked; papers listed are those read.

**Coverage gaps:** this is far short of the assignment's "deeply analyze hundreds". Citation graph built from OpenAlex reference lists covers only 4,349 papers (see `literature/citation_graph/report.md`). Semantic Scholar was unavailable without a key. Some extractions came from two-column text with interleaved lines.

## Map: what each family gives up

| Family | Representative | Needs gradient | Conditional? | In weights? | Known weakness |
|---|---|---|---|---|---|
| Prompting | | no | yes, by reading | no | tokens every query; hierarchy failures (2502.15851) |
| Constant steering | CAA 2312.06681, ActAdd 2308.10248, ITI 2306.03341, RepE 2310.01405 | no | no | no | per-input reliability varies (2407.12404, 2505.22637) |
| Gated steering | CAST 2409.05907, GAPS 2609.01878, ORBIT 2606.22357, CHaRS 2603.02237 | no | yes | no | threshold and hook |
| Prompt mimicry | PSR 2605.03907, GCAD 2605.10664, instruction steering 2410.12877 | PSR yes | yes | no | rank-1 not expressive for all instructions |
| Weight compile of steering | Arditi 2406.11717, TFTV 2609.09054, Steer2Edit 2602.09870, SteerEdit 2604.12359 | no | SteerEdit by trigger | yes | dual-use; mostly unconditional |
| Weight compile of prompts | ICC findings-emnlp.983, SELF-PARAM 2410.00487, D2L 2602.15902 | ICC no; others yes | by linear map | yes | calibration data |
| Fact editing | ROME 2202.05262, WISE 2405.14768, AlphaEdit 2410.02355 | partly | key-driven | yes | key collisions (2410.09338), multi-hop limits (2601.04600) |
| Hypernetworks | HyperSteer 2506.03292, T2L 2506.06105 | trained once | varies | no/yes | meta-training cost |
| Low-rank training | LoRA 2106.09685, QLoRA 2311.12023 | yes | no | yes | training cost; interference (2606.11262) |
| Tool-use control | PRISMS 2608.00218, heading steering 2607.05790 | probes only | yes | no | separate detector, or suppression only |

## Contradictions and unresolved points
- **Is steering a concept or just an output push?** 2606.29522 finds steering can flip answers while missing the natural target state; 2602.06801 finds orthogonal perturbations do the same; 2407.12404 finds a steerability bias. This agrees with our E003 and E008: the patch's FPR gain came from pushing negatives away.
- **Does gating help?** GAPS and PRISMS report sizeable gains, on models and tasks with substantial base failure. Our Qwen3-4B results suggest little gain once a model already discriminates. Unresolved: a likely moderator is base-model capability.
- **Reversal between models:** in our data the same method ranks differently on 1.5B and 4B models. Many papers evaluate one or two models, so such effects could go unseen.
- **Interference prediction exists for adapters and SAE features** (2610.03199, 2608.09490, 2505.11611). We did not find a closed-form capacity law for gates (search incomplete).

## Possibly unexplored connections
1. Zero-forcing theory for K closed-form gates (T01 section 3).
2. Prompt-vs-patch-vs-router at large K, with retrieval as the serious rival.
3. Low-rank state dependence of a prompt's effect as an explanation of steering limits, consistent with ICC's remark that fixed vectors recover only part of the gain.

## Dual use
Closed-form conditional weight edits are the mechanism in SteerEdit's stealthy backdoors. Any write-up of this class of method should recommend separate, signed adapters.
