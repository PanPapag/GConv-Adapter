#!/bin/bash
# Run: ./ablations/5_run_normalized_adj_ablation.sh -d <dataset1,dataset2,...> -s <script> -t <type> -p <positions>
# E.g. ./ablations/5_run_normalized_adj_ablation.sh -d esol,lipo,tox21,hiv -s ./scripts/finetune_inductive.py -t sequential -p pre,post -n none -l True -c True
# E.g. ./ablations/5_run_normalized_adj_ablation.sh -d cora,citeseer,pubmed -s ./scripts/finetune_transductive_learning.py -t sequential -p pre,post -n none -l True -c True

# You can modify the following:
# --------------------------------------------------------
# WANDB configuration
wandb_project="Thesis-Inductive-Learning-Ablations"
# wandb_project="Thesis-Transductive-Learning-Ablations"
wandb_name=""
wandb_tags=""
wandb_notes=""

# Seeds to be used
seeds=(0 1 2 3 4)
# --------------------------------------------------------

datasets=()

# Skip connection values for ablation studies
normalized_adjacencies=(True False)

# Function to run the Python script with specific parameters
run_ablation() {
    local dataset=$1
    local normalized_adj=$2
    local seed=$3
    local finetune_script=$4
    local adapter_type=$5
    local adapter_positions=$6
    local normalization=$7
    local learnable_scalar=$8
    local skip_connection=$9
    
    echo "Running 'Normalized Adjacency' ablation for dataset: $dataset, seed: $seed, normalized_adj: $normalized_adj, adapter type: $adapter_type, adapter positions: $adapter_positions, normalization: $normalization, learnable_sclar: $learnable_scalar, skip_connection: $skip_connection"
    
    # Update the YAML configuration using the Python script
    python ablations/configure_gconv_adapter.py \
        --skip_connection $skip_connection \
        --type $adapter_type \
        --positions ${adapter_positions//,/ } \
        --normalization ${normalization} \
        --learnable_scalar ${learnable_scalar} \
        --normalize ${normalized_adj}

    # Run the finetuning script
    python "$finetune_script" seed=$seed dataset.name=$dataset +finetune=gconv_adapter \
        wandb.project=$wandb_project \
        wandb.name=$wandb_name \
        wandb.tags=$wandb_tags \
        wandb.notes=$wandb_notes
}


# Usage message for the script
usage() {
    echo "Usage: $0 -d dataset1,dataset2,... -s /path/to/finetune_script.py -t <type> -p <positions> -n <normalization> -l <learnable_scalar> -c <skip_connection>"
    exit 1
}

# Parse arguments for datasets, finetune script path, type, and positions
while getopts ":d:s:t:p:n:l:c:" opt; do
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
        c )
            skip_connection=$OPTARG  
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if datasets, finetune_script, type, and positions are provided
if [ ${#datasets[@]} -eq 0 ] || [ -z "$finetune_script" ] || [ -z "$adapter_type" ] || [ -z "$adapter_positions" ] || [ -z "$normalization" ] || [ -z "$learnable_scalar" ] || [ -z "$skip_connection" ]; then
    usage
fi

# Loop over all datasets, skip connection values, adapter types, adapter positions, and seeds
for dataset in "${datasets[@]}"; do
    for normalized_adj in "${normalized_adjacencies[@]}"; do
        for seed in "${seeds[@]}"; do
            run_ablation "$dataset" "$normalized_adj" "$seed" "$finetune_script" "$adapter_type" "$adapter_positions" "$normalization" "$learnable_scalar" "$skip_connection" 
        done
    done
done