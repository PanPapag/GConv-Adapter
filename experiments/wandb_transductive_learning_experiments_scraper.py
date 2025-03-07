import wandb
import pandas as pd
import argparse
import re
import numpy as np

def scrape_results(project_name, entity=None, dataset_name=None, pretrained_model=None):
    """
    Scrape and return results from a W&B project, filtered by dataset name and pretrained model.
    Return a dictionary of mean and std for each fine-tuning method.
    """
    # Initialize W&B API
    api = wandb.Api()

    # Get the project
    runs = api.runs(f"{entity}/{project_name}") if entity else api.runs(project_name)

    # Filter runs based on config parameters if specified
    filtered_runs = []
    if dataset_name and pretrained_model:
        for run in runs:
            run_config = run.config
            if (dataset_name and run_config.get('dataset').get('name') == dataset_name) and \
               (pretrained_model and run_config.get('training', {}).get('pretrained_model') == pretrained_model):
                filtered_runs.append(run)
        runs = filtered_runs

    # If no matching runs found, return None
    if not runs:
        return None

    # Initialize a dictionary to store the extracted data for each fine-tuning method
    fine_tuning_results = {}

    # Group results by fine-tuning method
    method_results = {}

    # Loop over each run and extract the relevant metrics
    for run in runs:
        # Retrieve the history as a DataFrame
        history = run.history(pandas=True)
        
        if history.empty:
            continue

        # Find the column that contains "Validation"
        validation_columns = [col for col in history.columns if "Validation" in col]

        if not validation_columns:
            continue

        validation_column = validation_columns[0]  # Use the first match found
        test_column = "Test_" + validation_column.split("Validation_")[-1]
        # Determine the fine-tuning method
        fine_tuning_method = run.name.split("-")[0]

        best_row = history.loc[history[validation_column].idxmax()]  

        # Derive Test metric column name and extract its value
        best_test_scores = {test_column: best_row[test_column]} if test_column in best_row else {"N/A": "N/A"}

        # Adjust Test Accuracy values to percentage format
        for test_column, test_value in best_test_scores.items():
            if "Accuracy" in test_column:
                test_value *= 100

            # Group by fine-tuning method
            if fine_tuning_method not in method_results:
                method_results[fine_tuning_method] = []
            method_results[fine_tuning_method].append(test_value)

    # Compute the mean and std for each method
    for method, results in method_results.items():
        results = np.array(results, dtype=float)
        mean_val = np.mean(results)
        std_val = np.std(results)
        fine_tuning_results[method] = {
            "mean": mean_val,
            "std": std_val
        }

    return fine_tuning_results


def strip_ansi_codes(text):
    """Helper function to strip ANSI color codes from the string."""
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def apply_color_formatting(df, dataset):
    """
    Apply terminal color formatting:
    - Red, blue, and purple for the top three performing methods.
    - White for all other cells.
    """
    # Check if the value is already formatted as mean ± std and skip reformatting
    def format_value(x):
        if isinstance(x, str) and '±' in x:
            return x  # Already formatted as mean ± std
        elif isinstance(x, dict):
            return f"{x['mean']:.3f}±{x['std']:.3f}"
        else:
            return f"{float(x):.3f}"

    # Apply the format_value function to each cell in the dataset column
    df[dataset] = df[dataset].apply(format_value)

    # Find the maximum length of the formatted numbers (without ANSI codes) to pad the strings
    max_len = df[dataset].apply(lambda x: len(strip_ansi_codes(x))).max()

    # Accuracy: Higher is better
    sorted_df = df[dataset].sort_values(key=lambda x: x.apply(lambda val: float(strip_ansi_codes(val).split("±")[0])), ascending=False)

    # Apply color formatting for the top 3 methods and pad all values equally
    for idx in range(len(sorted_df)):
        value = sorted_df.iloc[idx].ljust(max_len)
        if idx == 0:
            df.loc[sorted_df.index[0], dataset] = f"\033[91m{value}\033[0m"  # Red for best
        elif idx == 1:
            df.loc[sorted_df.index[1], dataset] = f"\033[94m{value}\033[0m"  # Blue for second
        elif idx == 2:
            df.loc[sorted_df.index[2], dataset] = f"\033[95m{value}\033[0m"  # Purple for third
        else:
            df.loc[sorted_df.index[idx], dataset] = f"\033[97m{value}\033[0m"  # White for all others

    return df


def print_aligned_table(df):
    """
    Custom function to print the DataFrame with proper alignment, taking into account ANSI color codes.
    Apply white color to all elements including headers.
    """
    # Get column headers and data
    headers = df.columns.tolist()
    rows = df.index.tolist()

    # Find the maximum column width for proper alignment
    col_widths = {col: max(len(strip_ansi_codes(col)), df[col].apply(lambda x: len(strip_ansi_codes(x))).max()) for col in headers}
    method_width = max(len('Method'), max(len(strip_ansi_codes(row)) for row in rows))

    # Apply white color to headers and row labels
    header_str = " | ".join([f"\033[97m{col:<{col_widths[col]}}\033[0m" for col in headers])
    print(f"\033[97m{'Method':<{method_width}}\033[0m | {header_str}")
    print("\033[97m" + "=" * (method_width + len(strip_ansi_codes(header_str)) + 3) + "\033[0m")

    # Print each row with color formatting and proper alignment
    for row in rows:
        row_str = " | ".join([f"{df[col].loc[row]:<{col_widths[col]}}" for col in headers])
        print(f"\033[97m{row:<{method_width}}\033[0m | {row_str}")


def process_combination(args, datasets, pretrained_model, finetuning_order, finetuning_rename):
    # Create an empty DataFrame to store the final table
    final_results = pd.DataFrame()

    # Loop over datasets and collect the results
    for dataset in datasets:
        result = scrape_results(
            project_name=args.project,
            entity=args.entity,
            dataset_name=dataset,
            pretrained_model=pretrained_model
        )

        if result:
            # Convert the result dict to a DataFrame
            result_df = pd.DataFrame.from_dict(result, orient='index')
            
            # Concatenate the 'mean' and 'std' columns into a single column with format "mean±std"
            result_df[dataset] = result_df.apply(lambda row: f"{row['mean']:.3f}±{row['std']:.3f}", axis=1)

            # Drop the original 'mean' and 'std' columns
            result_df = result_df[[dataset]]  # Keep only the newly formatted column

            # Join results into the final table
            if final_results.empty:
                final_results = result_df
            else:
                final_results = final_results.join(result_df, how='outer')

            # Apply terminal color formatting for top 3 performing methods and white for others
            final_results = apply_color_formatting(final_results, dataset)

    # Reorder rows according to finetuning_order and rename the rows
    final_results = final_results.reindex(finetuning_order)
    final_results.rename(index=finetuning_rename, inplace=True)

    # Print the aligned table with colors
    print_aligned_table(final_results)


def main():
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Scrape W&B results based on specific parameters.")
    
    # Add arguments
    parser.add_argument("--project", required=True, help="Name of the W&B project.")
    parser.add_argument("--entity", default=None, help="Name of the W&B entity (optional).")
    parser.add_argument("--pretrained_model", required=True, help="Single pretrained model.")
    
    # Parse arguments
    args = parser.parse_args()

    datasets = ["cora", "citeseer", "pubmed"]

    # Order of fine-tuning methods
    finetuning_order = ["FT", "SurgicalFinetuning", "BitFit", "LoRA", "Adapter", "GAdapter", "AdapterGNN", "GConvAdapter"]

    # Dictionary to rename fine-tuning methods
    finetuning_rename = {
        "FT": "Full Fine-tuning",
        "SurgicalFinetuning": "Surgical Fine-tuning",
        "BitFit": "BitFit",
        "LoRA": "LoRA",
        "Adapter": "Adapter",
        "GAdapter": "G-Adapter",
        "AdapterGNN": "AdapterGNN",
        "GConvAdapter": "GConv-Adapter"
    }

    process_combination(args, datasets, args.pretrained_model, finetuning_order, finetuning_rename)


if __name__ == "__main__":
    main()
