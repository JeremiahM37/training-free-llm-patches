#!/bin/bash
cd ~/llm-mod/experiments && . ../.venv/bin/activate
until grep -q QWEN3_PIPELINE_DONE runq3.log; do sleep 30; done
M=Qwen/Qwen3-4B-Instruct-2507
nice -n 10 python evaluations/e005_prior_rank_one.py $M > e005q3.log 2>&1
nice -n 10 python evaluations/e006_many_behaviours.py $M > e006q3.log 2>&1
echo AFTER_DONE
