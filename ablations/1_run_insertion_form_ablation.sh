#!/bin/bash
# Run: ./ablations/1_run_insertion_form_ablation.sh -d <dataset1, dataset2, etc.> -s <script>
# E.g. ./ablations/1_run_insertion_form_ablation.sh -d esol,lipo,sider,muv  -s scripts/finetune_inductive_learning.py
# E.g. ./ablations/1_run_insertion_form_ablation.sh -d cora,citeseer,pubmed  -s scripts/finetune_transductive_learning.py

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

# Datasets to be used (will be passed as an argument)
datasets=()

# Positions and types for ablation studies
positions=("pre" "post" "pre post")
types=("sequential" "parallel")

# Function to run the Python script with specific parameters
run_ablation() {
    local dataset=$1
    local position=$2
    local type=$3
    local seed=$4
    local finetune_script=$5

    echo "Running 'Insertion Form' ablation for dataset: $dataset, seed: $seed, type: $type, position: $position"
    
    python ablations/configure_gconv_adapter.py \
        --positions $position \
        --type $type \
        --normalization none \
        --learnable_scalar False \
        --skip_connection True

    python "$finetune_script" seed=$seed dataset.name=$dataset +finetune=gconv_adapter \
        wandb.project=$wandb_project \
        wandb.name=$wandb_name \
        wandb.tags=$wandb_tags \
        wandb.notes=$wandb_notes
}


# Usage message for script
usage() {
    echo "Usage: $0 -d dataset1,dataset2,... -s /path/to/finetune_chem.py"
    exit 1
}

# Parse arguments for datasets and finetune script path
while getopts ":d:s:" opt; do
    case ${opt} in
        d )
            IFS=',' read -ra datasets <<< "$OPTARG"
            ;;
        s )
            finetune_script=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if datasets and finetune_script are provided
if [ ${#datasets[@]} -eq 0 ] || [ -z "$finetune_script" ]; then
    usage
fi

# Loop over all datasets, positions, and types
for dataset in "${datasets[@]}"; do
    for type in "${types[@]}"; do
        for position in "${positions[@]}"; do
            for seed in "${seeds[@]}"; do
                run_ablation "$dataset" "$position" "$type" "$seed" "$finetune_script"
            done
        done
    done
done
