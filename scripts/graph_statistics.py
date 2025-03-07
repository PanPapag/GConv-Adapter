import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

import numpy as np

from tqdm import tqdm
from torch_geometric.utils import to_networkx

from src.graphs.analysis import calculate_graph_statistics
from src.utils.graphs import print_graph_statistics, save_graph_statistics_to_json
from src.utils.plots import plot_histogram
from src.utils.logging import setup_logging


@hydra.main(
    config_path="../configs", config_name="graph_statistics", version_base="1.2"
)
def main(cfg: DictConfig):
    # Set up logging
    output_dir = HydraConfig.get().runtime.output_dir
    logger = setup_logging(output_dir)

    # Log the Hydra run directory
    logger.info(f"Hydra run directory: {output_dir}\n")

    # Change the current working directory to Hydra's runtime directory
    os.chdir(output_dir)

    # Create directories to save exported files if they don't exist
    os.makedirs("statistics", exist_ok=True)
    os.makedirs("descriptors", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Log the dataset selection
    logger.info(f"Selected dataset: {cfg.dataset._target_}\n")

    # Load the dataset using the configuration specified in Hydra
    dataset = instantiate(cfg.dataset)
    graphs = [to_networkx(data, to_undirected=True) for data in dataset]

    # If there's only one graph in the dataset
    if len(graphs) == 1:
        G, data = graphs[0], dataset[0]
        logger.info("Single graph in the dataset\n")

        # Calculate and log statistics of the single graph
        stats = calculate_graph_statistics(G, data)
        print_graph_statistics(stats, logger=logger)

        # Save graph statistics to a JSON file
        stats_json_path = os.path.join(output_dir, "statistics/graph_statistics.json")
        save_graph_statistics_to_json(stats, stats_json_path)
        logger.info(f"Saved graph statistics to: {stats_json_path}\n")

        # Compute and save histograms for each selected descriptor
        for name, desc_cfg in cfg.descriptors.items():
            hist = instantiate(
                desc_cfg, G=G
            )  # Instantiate the descriptor function with the graph
            hist_npy_path = os.path.join(output_dir, "descriptors", f"{name}.npy")
            np.save(hist_npy_path, hist)  # Save the histogram data as a numpy file
            logger.info(f"Saved {name} histogram data to: {hist_npy_path}")

            # Plot and save the histogram
            plot_filename = f"{name}.png"
            plot_histogram(
                hist,
                title=name.replace("_", " ").title(),
                xlabel=name.replace("_", " ").replace("distribution", "").title(),
                ylabel="Density",
                filename=plot_filename,
                save_path=os.path.join(output_dir, "plots"),
                show=cfg.plot.show,
                save=cfg.plot.save,
                legend_loc="upper right",
            )
            logger.info(
                f'Saved {name} plot to: {os.path.join(output_dir, "plots", plot_filename)}\n'
            )

    # If there are multiple graphs in the dataset
    else:
        logger.info(f"{len(graphs)} graphs in the dataset\n")

        counter = 1
        results = {}
        for G, data in tqdm(zip(graphs, dataset), leave=True, position=0):
            results[str(counter)] = calculate_graph_statistics(G, data)
            counter += 1

        # num_nodes = [g.number_of_nodes() for g in graphs]
        # num_edges = [g.number_of_edges() for g in graphs]

        # logger.info(f'Dataset size: {len(graphs)}\n')
        # logger.info(f'Mean number of nodes: {np.mean(num_nodes)}')
        # logger.info(f'Mean number of edges: {np.mean(num_edges)}')
        # logger.info(f'Max number of nodes: {np.max(num_nodes)}')
        # logger.info(f'Min number of nodes: {np.min(num_nodes)}')
        # logger.info(f'Max number of edges: {np.max(num_edges)}')
        # logger.info(f'Min number of edges: {np.min(num_edges)}\n')

        # TODO: Calculate and plot histograms for the entire dataset


if __name__ == "__main__":
    main()
