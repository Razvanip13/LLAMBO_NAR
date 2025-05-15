# Script to run LLAMBO on all HPOBench tasks.



#!/bin/bash
trap "kill -- -$BASHPID" EXIT

ENGINE="gpt35turbo_20230727"

for dataset in "blood_transfusion"
do
    for model in "rf" "xgb" "nn"
    do
        echo "dataset: $dataset, model: $model"

        python3 -m exp_hpo_bench.run_hpo_bench --dataset $dataset --model $model --seed 0 --num_seeds 1 --engine $ENGINE --sm_mode discriminative --acq_strategy EI --resume False
    done
done