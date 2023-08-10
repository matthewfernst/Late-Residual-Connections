import os 
import glob as glob
from typing import List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import utilities.dataframe as df_utils


def graph_results(model: torch.nn.Module, learning_rate: int, network_architecture: List[int], width: int,
                  optimizer: str, iteration: int, training_style: str, did_converge: bool) -> None:
    """
    graphs the results of the experiment and saves them to the graphs folder.
    :param model: The model that was trained.
    :param learning_rate: The learning rate used for training.
    :param network_architecture: The network architecture used for training.
    :param width: The width of the network.
    :param optimizer: The optimizer used for training.
    :param iteration: The iteration of the experiment.
    :param training_style: The training style used for training.
    :param did_converge: Whether or not the model converged.
    """
    x, t = df_utils.load_abs_data()
    depth = f"{len(network_architecture)}"

    colors = {"Adam": "blue",
              "SGD": "red",
              "RMSprop": "green",
              "Adagrad": "yellow",
              "Adadelta": "magenta",
              "Adamax": "cyan"}
    color = colors[optimizer]

    convergence = "Convergence" if did_converge else "No-Convergence"

    directory_path = f"graphs/Width-{width}/{optimizer}/LearningRate-{learning_rate}/ \
        {training_style}/{convergence}/Depth-{depth}/"
    df_utils.make_directory_if_not_exists(directory_path)

    filename = f"Iteration-{iteration + 1}"
    full_path = os.path.join(directory_path, f"{filename}.jpeg")

    y = model.use(x)

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"{optimizer}-Width-{width}-Depth{depth}", fontsize=16)
    plt.subplot(1, 2, 1)

    if model.device != "cpu":
        model.error_trace = [tensor.cpu().detach().numpy() for tensor in model.error_trace]

    plt.plot(model.error_trace, color="orange", label=optimizer)
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.ylim((0.0, 0.3))
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(y, "-s", color=color, label=optimizer)
    plt.plot(t, "-o", color="green", label="Target")
    plt.xlabel("Sample")
    plt.ylabel("Target or Predicted")
    plt.legend()

    plt.savefig(full_path, bbox_inches="tight")
    plt.close("all")


def graph_all_results(width: int) -> None:
    """
    graphs all of the results for a given width.
    :param width: The width of the network.
    """
    def graph_bar_results(dead_neurons_data: np.ndarray, optimizer: str, training_style: str, learning_rate: float,
                          x_ticks: List[str]) -> None:
        """
        graphs the dead neuron results for a given experiment.
        :param dead_neurons_data: The data to graph.
        :param optimizer: The optimizer used for training.
        :param training_style: The training style used for training.
        :param learning_rate: The learning rate used for training.
        :param x_ticks: The x-ticks to use for spacing.
        """

        x = np.arange(len(dead_neurons_data[-1]))
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1.5, 1.5])
        ax.set_ylabel("Number of Dead Neurons")
        ax.set_xlabel("Network Architecture")
        ax.set_title(f"Number of Dead Neurons vs Non-Residual and Late Residual Networks\n\
            {training_style} - {optimizer} - Learning Rate - {learning_rate}")
        ax.bar(x, dead_neurons_data[0], color="steelblue", width=0.25)
        ax.bar(x + 0.25, dead_neurons_data[1], color="darkorange", width=0.25)
        ax.legend(labels=["Non-Residual", "Residual"])
        ax.set_xticks(np.arange(len(dead_neurons_data[-1])), x_ticks)

        directory_path = f"../graphs/Width-{width}/{optimizer}/LearningRate-{learning_rate}/{training_style}/"
        filename = f"All-Results-DeadNeurons-{training_style}"
        full_path = f"{directory_path}{filename}.jpeg"

        plt.savefig(full_path, bbox_inches="tight")
        plt.close("all")

    def filtered_converged_data(df: pd.DataFrame, training_style: str) -> pd.DataFrame:
        """
        Filters the data to only include converged data.
        :param df: The dataframe to filter.
        :param training_style: The training style to filter by.
        :return: The filtered dataframe.
        """

        dead_neurons_data = df[f"{training_style} - Amount of Dead Neurons"][:-1].astype(
            float).values
        total_converged = df[f"{training_style} - Total Converged"][:-
                                                                    1:2].astype(float).values
        x_ticks = list(map(lambda x: x[:x.index("-")], df["Network Architecture"].values[:-1:2]))
        indices_of_no_convergence = np.where(total_converged == 0)[0]

        dead_neurons_non_residual = np.delete(dead_neurons_data[:-1:2], indices_of_no_convergence)
        dead_neurons_residual = np.delete(dead_neurons_data[1::2], indices_of_no_convergence)
        x_ticks = np.delete(x_ticks, indices_of_no_convergence)

        return [dead_neurons_non_residual, dead_neurons_residual], x_ticks

    # Main graph running python
    all_csvs = glob.glob(f"Results/Width-{width}/*/*.csv")
    for csv in all_csvs:
        df = pd.read_csv(csv)
        optimizer = csv.split("/")[-1].split("-")[0]
        learning_rate = csv.split("/")[-1].split("-")[-1]
        learning_rate = learning_rate[:learning_rate.rindex(".")]

        for training_style in ["Iterative", "Batch"]:
            dead_neurons_data, x_ticks = filtered_converged_data(df, training_style)
            graph_bar_results(dead_neurons_data, optimizer, training_style, learning_rate, x_ticks)
