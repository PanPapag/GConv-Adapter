import argparse
from tqdm import tqdm
import subprocess

# Function to get total combinations
def get_total_combinations(datasets, fine_tuning_methods, seeds):
    return len(datasets) * len(seeds) * len(fine_tuning_methods) 

def main(args):
    # Calculate the total number of iterations
    total_iterations = get_total_combinations(args.datasets, args.fine_tuning_methods, args.seeds)

    # Initialize a counter
    current_index = 0

    # Use tqdm to wrap the loop
    with tqdm(total=total_iterations, desc="Progress", unit="combination") as pbar:
        for dataset_name in args.datasets:
            for fine_tuning in args.fine_tuning_methods:
                for seed in args.seeds:
                    # Check if the current index is greater than or equal to the start index
                    if current_index >= args.start_index:
                        # Construct the command for the model
                        cmd = (
                            f"python scripts/finetune_transductive_learning.py "
                            f"++dataset.name={dataset_name} "
                            f"++seed={seed} "
                            f"+finetune={fine_tuning}"
                        )
                        # Print the command being run (optional)
                        print(f"Running: {cmd}")
                        # Run the command
                        subprocess.run(cmd, shell=True, check=True)
                        # Update the progress bar
                        pbar.update(1)
                    else:
                        # Skip this combination
                        pbar.update(1)

                    # Increment the current index
                    current_index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning GNN models with different configurations")

    # Define the arguments
    parser.add_argument("--datasets", nargs='+', default=["cora", "citeseer", "pubmed"], help="List of datasets")
    parser.add_argument("--fine_tuning_methods", nargs='+', default=["full", "bitfit", "surgical", "lora", "adapter", "g_adapter", "adapter_gnn", "gconv_adapter"], help="List of fine-tuning methods")
    parser.add_argument("--seeds", nargs='+', type=int, default=[1, 2, 3], help="List of seeds")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start execution from")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function
    main(args)
