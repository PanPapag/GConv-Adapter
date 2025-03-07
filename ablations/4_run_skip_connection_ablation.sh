#!/bin/bash
# Run: ./ablations/4_run_skip_connection_ablation.sh -d <dataset1,dataset2,...> -s <script> -t <type> -p <positions>
# E.g. ./ablations/4_run_skip_connection_ablation.sh -d esol,lipo,bace,muv -s scripts/finetune_inductive_learning.py -t sequential -p pre,post -n none -l True
# E.g. ./ablations/4_run_skip_connection_ablation.sh -d cora,citeseer,pubmed -s scripts/finetune_transductive_learning.py -t sequential -p pre,post -n none -l True

# You can modify the following:
# --------------------------------------------------------
# WANDB configuration
# wandb_project="Thesis-Inductive-Learning-Ablations"
wandb_project="Thesis-Transductive-Learning-Ablations"
wandb_name=""
wandb_tags=""
wandb_notes=""

# Seeds to be used
seeds=(0 1 2 3 4)
# --------------------------------------------------------

datasets=()

# Skip connection values for ablation studies
skip_connections=(True False)

# Function to run the Python script with specific parameters
run_ablation() {
    local dataset=$1
    local skip_connection=$2
    local seed=$3
    local finetune_script=$4
    local adapter_type=$5
    local adapter_positions=$6
    local normalization=$7
    local learnable_scalar=$8
    
    echo "Running 'Skip Connection' ablation for dataset: $dataset, seed: $seed, skip_connection: $skip_connection, adapter type: $adapter_type, adapter positions: $adapter_positions, normalization: $normalization, learnable_sclar: $learnable_scalar"
    
    # Update the YAML configuration using the Python script
    python ablations/configure_gconv_adapter.py \
        --skip_connection $skip_connection \
        --type $adapter_type \
        --positions ${adapter_positions//,/ } \
        --normalization ${normalization} \
        --learnable_scalar ${learnable_scalar}

    # Run the finetuning script
    python "$finetune_script" seed=$seed dataset.name=$dataset +finetune=gconv_adapter \
        wandb.project=$wandb_project \
        wandb.name=$wandb_name \
        wandb.tags=$wandb_tags \
        wandb.notes=$wandb_notes
}


# Usage message for the script
usage() {
    echo "Usage: $0 -d dataset1,dataset2,... -s /path/to/finetune_script.py -t <type> -p <positions> -n <normalization> -l <learnable_scalar>"
    exit 1
}

# Parse arguments for datasets, finetune script path, type, and positions
while getopts ":d:s:t:p:n:l:" opt; do
    case ${opt} in
        d )
            IFS=',' read -ra datasets <<< "$OPTARG"
            ;;
        s )
            finetune_script=$OPTARG
            ;;
        t )
            adapter_type=$OPTARG
            ;;
        p )
            adapter_positions=$OPTARG  
            ;;
        n )
            normalization=$OPTARG  
            ;;
        l )
            learnable_scalar=$OPTARG  
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if datasets, finetune_script, type, and positions are provided
if [ ${#datasets[@]} -eq 0 ] || [ -z "$finetune_script" ] || [ -z "$adapter_type" ] || [ -z "$adapter_positions" ] || [ -z "$normalization" ] || [ -z "$learnable_scalar" ]; then
    usage
fi

# Loop over all datasets, skip connection values, adapter types, adapter positions, and seeds
for dataset in "${datasets[@]}"; do
    for skip_connection in "${skip_connections[@]}"; do
        for seed in "${seeds[@]}"; do
            run_ablation "$dataset" "$skip_connection" "$seed" "$finetune_script" "$adapter_type" "$adapter_positions" "$normalization" "$learnable_scalar"
        done
    done
done