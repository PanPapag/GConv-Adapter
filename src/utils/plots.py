import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional


def plot_histogram(
    data: Union[List[float], np.ndarray],
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    filename: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    save: bool = True,
    legend_loc: str = "best",
):
    """
    Plot and optionally save or show a histogram.

    Args:
        data (list or np.ndarray): Data to be plotted.
        title (str, optional): Title of the plot. Default is 'Histogram'.
        xlabel (str, optional): X-axis label. Default is 'Value'.
        ylabel (str, optional): Y-axis label. Default is 'Frequency'.
        filename (str, optional): Filename to save the plot. Default is None.
        save_path (str, optional): Path to save the plot. Default is None.
        show (bool, optional): Whether to show the plot. Default is False.
        save (bool, optional): Whether to save the plot. Default is True.
        legend_loc (str, optional): Location of the legend. Default is 'best'.
    """
    plt.figure()
    plt.plot(data, label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)

    if save and filename and save_path:
        plt.savefig(os.path.join(save_path, filename))

    if show:
        plt.show()

    plt.close()