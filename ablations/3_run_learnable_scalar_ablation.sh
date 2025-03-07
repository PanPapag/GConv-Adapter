#!/bin/bash
# Run: ./ablations/3_run_learnable_scalar_ablation.sh -d <dataset1,dataset2,...> -s <script> -t <type> -p <positions> -n <normalization>
# E.g. ./ablations/3_run_learnable_scalar_ablation.sh -d esol,lipo,clintox,hiv -s scripts/finetune_inductive_learning.py -t sequential -p pre,post -n none
# E.g. ./ablations/3_run_learnable_scalar_ablation.sh -d cora,citeseer,pubmed -s scripts/finetune_transductive_learning.py -t sequential -p pre,post -n none 

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

# Learnable scalar values for ablation studies
learnable_scalars=(True False)

# Function to run the Python script with specific parameters
run_ablation() {
    local dataset=$1
    local learnable_scalar=$2
    local seed=$3
    local finetune_script=$4
    local adapter_type=$5
    local adapter_positions=$6
    local normalization=$7
    
    echo "Running 'Learnable Scalar' ablation for dataset: $dataset, seed: $seed, learnable_scalar: $learnable_scalar, adapter type: $adapter_type, adapter positions: $adapter_positions, normalization: $normalization"
    
    # Update the YAML configuration using the Python script
    python ablations/configure_gconv_adapter.py \
        --learnable_scalar $learnable_scalar \
        --type $adapter_type \
        --positions ${adapter_positions//,/ } \
        --normalization $normalization \
        --skip_connection True

    # Run the finetuning script
    python "$finetune_script" seed=$seed dataset.name=$dataset +finetune=gconv_adapter \
        wandb.project=$wandb_project \
        wandb.name=$wandb_name \
        wandb.tags=$wandb_tags \
        wandb.notes=$wandb_notes
}

# Usage message for the script
usage() {
    echo "Usage: $0 -d dataset1,dataset2,... -s /path/to/finetune_script.py -t <type> -p <positions> -n <normalization>"
    exit 1
}

# Parse arguments for datasets, finetune script path, type, and positions
while getopts ":d:s:t:p:n:" opt; do
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
        \? )
            usage
            ;;
    esac
done

# Check if datasets, finetune_script, type, positions and normalization are provided
if [ ${#datasets[@]} -eq 0 ] || [ -z "$finetune_script" ] || [ -z "$adapter_type" ] || [ -z "$adapter_positions" ] || [ -z "$normalization" ]; then
    usage
fi

# Loop over all datasets, learnable scalar values, adapter types, adapter positions, and seeds
for dataset in "${datasets[@]}"; do
    for learnable_scalar in "${learnable_scalars[@]}"; do
        for seed in "${seeds[@]}"; do
            run_ablation "$dataset" "$learnable_scalar" "$seed" "$finetune_script" "$adapter_type" "$adapter_positions" "$normalization"
        done
    done
done
