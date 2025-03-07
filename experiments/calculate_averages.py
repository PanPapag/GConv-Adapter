import json
import sys
import re
from statistics import mean

def extract_value(metric_str):
    """
    Extract the numeric value before the ± symbol.
    """
    return float(re.split(r'±', metric_str)[0])

def calculate_averages_for_methods(data):
    """
    Calculate:
    1. The average value for each model for each method across all datasets.
    2. The average value across all models for each dataset in each method.
    3. The overall average across models and datasets for each method.
    """
    models = ['GCN', 'GraphSAGE', 'GAT', 'GIN']
    averages_per_method = {}
    dataset_averages_per_method = {}

    for method, model_data in data.items():
        method_averages = {model: [] for model in models}
        dataset_averages = {}

        for model in models:
            for dataset, metric_value in model_data[model].items():
                value = extract_value(metric_value)
                method_averages[model].append(value)

                # Accumulate values for each dataset
                if dataset not in dataset_averages:
                    dataset_averages[dataset] = []
                dataset_averages[dataset].append(value)

        # Calculate the average for each model for this method
        averages_per_method[method] = {model: mean(values) for model, values in method_averages.items() if values}

        # Calculate the average across all models for each dataset
        dataset_averages_per_method[method] = {dataset: mean(values) for dataset, values in dataset_averages.items()}

    return averages_per_method, dataset_averages_per_method

def calculate_overall_average(averages_per_method, dataset_averages_per_method):
    """
    Calculate the overall average across all models and datasets for each method.
    """
    overall_averages = {}

    for method in averages_per_method.keys():
        all_values = []
        
        # Gather all model averages
        for avg in averages_per_method[method].values():
            all_values.append(avg)
        
        # Gather all dataset averages
        for avg in dataset_averages_per_method[method].values():
            all_values.append(avg)

        overall_averages[method] = mean(all_values) if all_values else 0

    return overall_averages

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    # Load the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Calculate the averages for each method
    model_averages, dataset_averages = calculate_averages_for_methods(data)

    # Calculate the overall averages
    overall_averages = calculate_overall_average(model_averages, dataset_averages)

    # Print the results
    for method, model_averages in model_averages.items():
        print(f"\nAverages for method '{method}':")
        for model, avg in model_averages.items():
            print(f"  {model}: {avg:.3f}")

        print(f"\nDataset Averages for method '{method}':")
        for dataset, avg in dataset_averages[method].items():
            print(f"  {dataset}: {avg:.3f}")

        print(f"\nOverall Average for method '{method}': {overall_averages[method]:.3f}")
