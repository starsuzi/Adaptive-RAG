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
        #exit 1
    fi
}

# Check the number of arguments
if [[ $# -ne 4 ]]; then
    echo "Error: Invalid number of arguments. Expected format: ./run_retrieval_test.sh SYSTEM MODEL DATASET LLM_PORT_NUM"
    exit 1
fi

# Check the validity of arguments
check_argument "$1" 1 valid_systems[*]
check_argument "$2" 2 valid_models[*]
check_argument "$3" 3 valid_datasets[*]

echo ">>>> Instantiate experiment configs with different HPs and write them in files. <<<<"
python runner.py $1 $2 $3 write --prompt_set 1 --llm_port_num $4

echo ">>>> Run the experiment with best HP on the test set <<<<"
python runner.py $1 $2 $3 predict --prompt_set 1 --eval_test --official --llm_port_num $4

echo ">>>> Run evaluation for the best HP on the test set <<<<"
python runner.py $1 $2 $3 evaluate --prompt_set 1 --eval_test --official --llm_port_num $4