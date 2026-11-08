#!/bin/bash
cd ~/llm-mod/experiments && . ../.venv/bin/activate
until grep -q AFTER_DONE after.log; do sleep 30; done
M=Qwen/Qwen3-4B-Instruct-2507
nice -n 10 python evaluations/e007_self_compile.py $M > e007q3.log 2>&1
nice -n 10 python evaluations/e009_no_schema.py $M > e009q3.log 2>&1
nice -n 10 python evaluations/e008_precision_sign.py > e008.log 2>&1
echo AFTER2_DONE
