#!/bin/bash
# Resumable queue for the desktop: skips anything whose result JSON already exists. Start with:
#   rsync -a --exclude __pycache__ research/experiments/ $GPU_HOST:llm-mod/experiments/
#   ssh $GPU_HOST 'nohup bash ~/llm-mod/experiments/run_resume.sh > ~/llm-mod/experiments/resume.log 2>&1 < /dev/null &'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/llm-mod/experiments && . ../.venv/bin/activate
M=Qwen/Qwen3-4B-Instruct-2507; R=../results/metrics
step() { [ -f "$R/$1" ] && grep -q "$2" "$R/$1" && { echo "skip $1"; return; }; shift 2; nice -n 10 "$@"; echo "ran: $* rc=$?"; }
step E010_Qwen3-4B-Instruct-2507_fewshot.json '"patches"' python evaluations/e010_rr_lora.py $M fewshot > e010q3.out 2>&1
step E006_Qwen3-4B-Instruct-2507.json 'E006_COMPLETE' python evaluations/e006_many_behaviours.py $M > e006q3.out 2>&1
step E003_Qwen3-4B-Instruct-2507.json '"curve"' python evaluations/e003_try_to_break.py $M "[('lda',34,1,'dL'),('lda',26,0.5,'dH'),('lcmv',26,0.5,'dH')]" > e003q3.out 2>&1
step E003_Qwen2.5-1.5B-Instruct.json '"logit_bias"' python evaluations/e003_try_to_break.py Qwen/Qwen2.5-1.5B-Instruct > e003.out 2>&1
echo RESUME_DONE
