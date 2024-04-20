#!/usr/bin/env bash

# Expected command line argument values.
valid_systems=("ircot" "ircot_qa" "oner" "oner_qa" "nor_qa")
valid_models=("flan-t5-xxl" "flan-t5-xl" 'gpt' "none")
valid_datasets=("hotpotqa" "2wikimultihopqa" "musique" 'nq' "trivia" "squad")

# Function to check if an argument is valid
check_argument() {
    local arg="$1"
    local position="$2"
    local valid_values=("${!3}")
    if ! [[ " ${valid_values[*]} " =~ " $arg " ]]; then
        echo "argument number $position is not a valid. Please provide one of: ${valid_values[*]}"
        exit 1
    fi

    if [[ $position -eq 2 && $arg == "none" && $1 != "oner" ]]; then
        echo "The model argument can only be 'none' only if the system argument is 'oner'."
        exit 1
    fi
}

# Check the number of arguments
if [[ $# -ne 4 ]]; then
    echo "Error: Invalid number of arguments. Expected format: ./run_retrieval_dev.sh SYSTEM MODEL DATASET LLM_PORT_NUM"
    exit 1
fi

# Check the validity of arguments
check_argument "$1" 1 valid_systems[*]
check_argument "$2" 2 valid_models[*]
check_argument "$3" 3 valid_datasets[*]

echo ">>>> Instantiate experiment configs with different HPs and write them in files. <<<<"
python runner.py $1 $2 $3 write --prompt_set 1 --llm_port_num $4

echo ">>>> Run experiments for different HPs on the dev set. <<<<"
python runner.py $1 $2 $3 predict --prompt_set 1 --sample_size 500 --llm_port_num $4
## If prediction files already exist, it won't redo them. Pass --force if you want to redo.

echo ">>>> Run evaluation for different HPs on the dev set. <<<<"
python runner.py $1 $2 $3 evaluate --prompt_set 1 --sample_size 500 --llm_port_num $4
## This runs by default after prediction. This is mainly to show a standalone option.

echo ">>>> Show results for experiments with different HPs <<<<"
python runner.py $1 $2 $3 summarize --prompt_set 1 --sample_size 500 --llm_port_num $4