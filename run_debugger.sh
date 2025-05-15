#!/bin/bash
# Debug configuration: Runs hpo_bench for a single dataset and model.

# Ensure background processes are killed if the script exits unexpectedly
trap "kill -- -$BASHPID" EXIT

# --- Configuration ---
# Define the engine to use
ENGINE="gpt35turbo_20230727"

# Define the specific dataset and model for this debug run
dataset="australian" # Hardcoded dataset
model="rf"          # Hardcoded model

# --- Execution ---
echo "Running debug configuration..."
echo "Dataset: $dataset"
echo "Model: $model"
echo "Engine: $ENGINE"
echo "---"

# Run the Python script with the specified parameters
python3 -m exp_hpo_bench.run_hpo_bench \
    --dataset "$dataset" \
    --model "$model" \
    --seed 0 \
    --num_seeds 1 \
    --engine "$ENGINE" \
    --sm_mode discriminative

# Check the exit status of the python script
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "---"
    echo "Error: Python script exited with status $exit_status."
    exit $exit_status # Propagate the error status
else
    echo "---"
    echo "Script finished successfully."
    exit 0
fi
