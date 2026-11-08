#!/bin/bash
# Full pipeline on Qwen3-4B. Selection on dev only, then frozen.
M=Qwen/Qwen3-4B-Instruct-2507
cd ~/llm-mod/experiments && . ../.venv/bin/activate
nice -n 10 python baselines/e001_prompt_baselines.py $M > e001q3.log 2>&1
nice -n 10 python prototypes/e002_rank_one_gate.py $M 12,18,22,26,29,31,33,34 > e002q3.log 2>&1
CFG=$(python - <<PY
import json
d=json.load(open("../results/metrics/E002_Qwen3-4B-Instruct-2507.json"))
rows=[r for r in d["rows"] if r["kind"]=="lda"]
best=max(rows,key=lambda r:r["J"])
other=max((r for r in rows if r["layer"]!=best["layer"]),key=lambda r:r["J"])
lc=max((r for r in d["rows"] if r["kind"]=="lcmv"),key=lambda r:r["J"])
print([("lda",best["layer"],best["alpha"]),("lda",other["layer"],other["alpha"]),("lcmv",lc["layer"],lc["alpha"])])
PY
)
echo "CFG $CFG"
nice -n 10 python evaluations/e003_try_to_break.py $M "$CFG" > e003q3.log 2>&1
nice -n 10 python evaluations/e004_latent_vs_text.py $M > e004q3.log 2>&1
echo QWEN3_PIPELINE_DONE
