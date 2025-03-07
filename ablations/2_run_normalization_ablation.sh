#!/bin/bash
# Run: ./ablations/2_run_normalization_ablation.sh -d <dataset1, dataset2, etc.> -s <script> -t <type> -p <position>
# E.g. ./ablations/2_run_normalization_ablation.sh -d freesolv,lipo,bace,muv -s scripts/finetune_inductive_learning.py -t sequential -p pre,post
# E.g. ./ablations/2_run_normalization_ablation.sh -d cora,citeseer,pubmed -s scripts/finetune_transductive_learning.py -t sequential -p pre,post

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

# Normalizations for ablation studies
normalizations=("batch_norm" "layer_norm" "none")

# Function to run the Python script with specific parameters
run_ablation() {
    local dataset=$1
    local normalization=$2
    local seed=$3
    local finetune_script=$4
    local adapter_type=$5
    local adapter_positions=$6
    
    echo "Running 'Normalization' ablation for dataset: $dataset, seed: $seed, normalization: $normalization, adapter type: $adapter_type, adapter positions: $adapter_positions"

    # Update the YAML configuration using the Python script
    python ablations/configure_gconv_adapter.py \
        --normalization $normalization \
        --type $adapter_type \
        --positions ${adapter_positions//,/ } \
        --learnable_scalar False \
        --skip_connection True

    # Run the finetuning script
    python "$finetune_script" seed=$seed dataset.name=$dataset +finetune=gconv_adapter \
        wandb.project=$wandb_project \
        wandb.name=$wandb_name \
        wandb.tags=$wandb_tags \
        wandb.notes=$wandb_notes
}

# Usage message for script
usage() {
    echo "Usage: $0 -d dataset1,dataset2,... -s /path/to/finetune_script.py -t <type> -p <positions>"
    exit 1
}

# Parse arguments for datasets, finetune script path, type, and positions
while getopts ":d:s:t:p:" opt; do
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
        \? )
            usage
            ;;
    esac
done

# Check if datasets, finetune_script, type, and positions are provided
if [ ${#datasets[@]} -eq 0 ] || [ -z "$finetune_script" ] || [ -z "$adapter_type" ] || [ ${#adapter_positions[@]} -eq 0 ]; then
    usage
fi

# Loop over all datasets, normalization techniques, adapter types, adapter positions, and seeds
for dataset in "${datasets[@]}"; do
    for normalization in "${normalizations[@]}"; do
        for seed in "${seeds[@]}"; do
            run_ablation "$dataset" "$normalization" "$seed" "$finetune_script" "$adapter_type" "${adapter_positions[@]}"
        done
    done
done
