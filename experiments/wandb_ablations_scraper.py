import itertools
import wandb
import pandas as pd
import numpy as np
import argparse

# Define a dictionary to map W&B tags to their corresponding ablation parameters
ablation_parameters = {
    "insertion_form": {
        "finetune_keys": {
            "positions": [["pre"], ["post"], ["pre", "post"]],
            "type": ["sequential", "parallel"]
        }
    },
    "non_linearity": {
        "finetune_keys": {
            "adapter.non_linearity": ["none", "relu", "silu"]
        }
    },
    "normalization": {
        "finetune_keys": {
            "adapter.normalization": ["none", "batch_norm", "layer_norm"]
        }
    },
    "learnable_scalar": {
        "finetune_keys": {
            "adapter.learnable_scalar": [False, True]
        }
    },
    "skip_connection": {
        "finetune_keys": {
            "adapter.skip_connection": [False, True]
        }
    },
    "normalized_adj": {
        "finetune_keys": {
            "adapter.normalize": [False, True]
        }
    },
}

# Desired dataset order for final table
desired_dataset_order = ["esol", "freesolv", "lipo", "tox21", "sider", "clintox", "bace", "muv", "hiv", "cora", "citeseer", "pubmed"]

def scrape_results(project_name, entity=None, pretrained_model=None, run_param_values=None, wandb_tag=None):
    """
    Scrape and return results from a W&B project, dynamically detect dataset names, and ensure proper ablation filtering.

    Args:
        project_name (str): Name of the W&B project.
        entity (str): Name of the W&B entity (optional, default: None).
        pretrained_model (str): The name of the pretrained model used in the run config (optional, default: None).
        ablation_params (dict): Dictionary containing the ablation parameters for the study (e.g., 'positions', 'type').
        run_param_values (dict): The specific values of the ablation parameters for this particular run.
        wandb_tag (str): The exact W&B tag to filter runs.

    Returns:
        dict: A dictionary containing the configuration and results (mean ± std) for validation and test metrics.
        set: A set of valid datasets.
    """
    # Initialize W&B API
    api = wandb.Api()

    # Get the project
    runs = api.runs(f"{entity}/{project_name}") if entity else api.runs(project_name)

    # Filter runs based on config parameters, tags, and ablation parameters
    filtered_runs = []
    valid_datasets = set()  # Track only valid datasets based on the specified W&B tag

    for run in runs:
        run_config = run.config
        match = True

        # Check all ablation parameters in the config to see if they match the values for this run
        for param_key, param_value in run_param_values.items():
            # Handle nested keys (e.g., "adapter.non_linearity")
            if "." in param_key:
                top_key, sub_key = param_key.split(".")
                if run_config.get('finetune', {}).get(top_key, {}).get(sub_key) != param_value:
                    match = False
                    break
            else:
                # Handle regular keys
                if run_config.get('finetune', {}).get(param_key) != param_value:
                    match = False
                    break

        # Check if the pretrained model matches
        if pretrained_model and (
            run_config.get('training', {}).get('pretrained_model') != pretrained_model
            and run_config.get('pretrained_model') != pretrained_model
        ):
            match = False
        
        # If the parameters match, check the W&B tags
        run_tags = run.tags
        if match and wandb_tag in run_tags:  # Check if the run has the exact wandb_tag
            filtered_runs.append(run)
            valid_datasets.add(run.config.get('dataset', {}).get('name'))  # Track valid datasets
    
    if not filtered_runs:
        print(f"No runs found for project '{project_name}'" +
              (f" and pretrained model '{pretrained_model}'" if pretrained_model else "") +
              (f" and tag '{wandb_tag}'" if wandb_tag else "") +
              (f" with ablation parameters '{run_param_values}'"))
        return None, valid_datasets

    # Initialize lists to store the extracted data
    dataset_scores = {}

    # Loop over each run and extract the relevant metrics
    for run in filtered_runs:
        history = run.history(pandas=True)

        if history.empty:
            continue

        validation_columns = [col for col in history.columns if "Validation" in col]

        if not validation_columns:
            continue

        validation_column = validation_columns[0]  # Use the first match found

        # Determine whether to minimize or maximize based on the metric type
        if "RMSE" in validation_column or "Loss" in validation_column:
            best_row = history.loc[history[validation_column].idxmin()]
        else:
            best_row = history.loc[history[validation_column].idxmax()]

        # Derive Test metric column name and extract its value
        test_column = "Test_" + validation_column.split("Validation_")[-1]
        test_value = best_row.get(test_column, np.nan)
        
        if "ROC-AUC" in test_column:
            test_value *= 100
        if "Accuracy" in test_column:
            test_value *= 100

        dataset_name = run.config.get('dataset', {}).get('name')
        if dataset_name not in dataset_scores:
            dataset_scores[dataset_name] = []
        dataset_scores[dataset_name].append(test_value)

    # Calculate mean and standard deviation for each dataset
    results = {}
    for dataset, scores in dataset_scores.items():
        if scores:  # Only include datasets that have at least one score
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            results[dataset] = f"{mean_score:.3f} ± {std_score:.3f}"

    return results, valid_datasets

def main():
    # Set up argparse to handle command-line arguments
    parser = argparse.ArgumentParser(description="Scrape and summarize results from W&B for ablation studies.")
        
    parser.add_argument('--project_name', type=str, default="Thesis-Transductive-Learning-Ablations", help='W&B project name.')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity name.')
    parser.add_argument('--wandb_tag', type=str, required=True, help='W&B tag to determine which ablation parameters to use.')
    parser.add_argument('--pretrained_model', type=str, default="pretrained/ogbn-arxiv-nodeformer.pkl", help='Single pretrained model to use.')

    args = parser.parse_args()

    # Get ablation parameters based on the tag
    if args.wandb_tag not in ablation_parameters:
        print(f"Error: The tag '{args.wandb_tag}' does not have corresponding ablation parameters defined.")
        return

    ablation_params = ablation_parameters[args.wandb_tag]['finetune_keys']

    # Initialize a list to store results across all datasets
    all_results = []
    all_valid_datasets = set()  # Store valid datasets across all runs

    # Create a list of all combinations of ablation parameters for this study
    param_combinations = list(itertools.product(*ablation_params.values()))

    # Loop over all combinations of ablation parameters
    for param_combination in param_combinations:
        # Create a dictionary of the specific values for this run (e.g., position and type)
        run_param_values = dict(zip(ablation_params.keys(), param_combination))

        # Scrape the results for the given combination
        result, valid_datasets = scrape_results(
            project_name=args.project_name,
            entity=args.entity,
            pretrained_model=args.pretrained_model,
            run_param_values=run_param_values,
            wandb_tag=args.wandb_tag
        )

        # Add valid datasets from this run to the global list
        all_valid_datasets.update(valid_datasets)

        if result:
            for dataset, value in result.items():
                result_entry = {
                    'Pretrained Model': args.pretrained_model,
                    'Dataset': dataset,
                    'Result': value
                }
                # Dynamically add the ablation parameters to the result
                result_entry.update(run_param_values)
                all_results.append(result_entry)

    # Organize the results into a DataFrame
    df = pd.DataFrame(all_results)

    # Drop rows with None results and ensure only datasets relevant to the ablation study are included
    df.dropna(subset=['Result'], inplace=True)

    # Convert list values to strings so they can be used as categories
    for param_key in ablation_params.keys():
        df[param_key] = df[param_key].apply(lambda x: str(x))
        df[param_key] = pd.Categorical(df[param_key], categories=[str(p) for p in ablation_params[param_key]], ordered=True)

    # Pivot the DataFrame to show datasets as columns
    df_pivot = df.pivot(index=list(ablation_params.keys()), columns='Dataset', values='Result').reset_index()

    # Keep only relevant datasets in the final output, ordered by the desired dataset order
    valid_columns = list(ablation_params.keys()) + [dataset for dataset in desired_dataset_order if dataset in df_pivot.columns]
    df_pivot = df_pivot[valid_columns]

    # Display the results
    print(df_pivot.to_string(index=False))

if __name__ == "__main__":
    main()
