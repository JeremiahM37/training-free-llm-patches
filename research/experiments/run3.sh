#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/llm-mod/experiments && . ../.venv/bin/activate
M=Qwen/Qwen3-4B-Instruct-2507
echo start
nice -n 10 python evaluations/e010_rr_lora.py $M fewshot > e010q3.log 2>&1; tail -n 1 e010q3.log | cut -c1-200
nice -n 10 python evaluations/e006_many_behaviours.py $M > e006q3.log 2>&1; tail -n 1 e006q3.log | cut -c1-200
nice -n 10 python evaluations/e003_try_to_break.py $M "[('lda',34,1,'dL'),('lda',26,0.5,'dH'),('lcmv',26,0.5,'dH')]" > e003q3.log 2>&1; tail -n 1 e003q3.log | cut -c1-200
nice -n 10 python evaluations/e003_try_to_break.py Qwen/Qwen2.5-1.5B-Instruct > e003.log 2>&1; tail -n 1 e003.log | cut -c1-200
echo RUN3_DONE
