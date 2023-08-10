import os
import time
from typing import List, Tuple
import torch
import pandas as pd
from alive_progress import alive_bar

import late_residual_pytorch.late_residual_neural_network as lrnn
import utilities.common as lr_utils
import utilities.graphing as graph_utils
import utilities.dataframe as df_utils


def run_model(x: torch.tensor, t: torch.tensor, epochs: int, network_architecture: List[int],
              optimizer: str, learning_rate: float, connection_style: str,
              training_style: str, verbose=False) -> Tuple[bool, List[float], torch.nn.Module]:
    """
    This function runs a specific model with the given parameters and returns the results. Specifically, it returns if
    the model converged, the number of dead neurons, the dead layers, training time, and the model itself.
    :param x: The input data for the model
    :param t: The target data for the model
    :param epochs: The number of epochs to train for
    :param network_architecture: The network architecture, specifically a list for the number of neurons in each layer
    :param optimizer: The optimizer to use for training
    :param learning_rate: The learning rate to use
    :param connection_style: The connection style to use
    :param training_style: The training style to use
    :return: A tuple consisting of, if the model converged, a list with the number of dead neurons, the dead layers,
    training time, and the model
    """
    convergence_threshold = 0.07

    model = lrnn.NNet(1, network_architecture, 1, optimizer,
                      is_residually_connected=(connection_style == 'residual'), device='mps')
    start = time.time()
    model.train(x, t, epochs, learning_rate, training_style, verbose=verbose)
    end = time.time()
    total_time = end - start

    y = model.use(x)
    final_rmse = lr_utils.rmse(y, t)

    if verbose:
        print(f"Total Time to Train {(total_time):.3f} seconds")
        print(f"RMSE {final_rmse:.3f}\n")

    dead_neurons, dead_layers = model.dead_neurons()

    return final_rmse <= convergence_threshold, [dead_neurons, dead_layers, total_time], model


def heart_of_experiment(epochs: int, width: int, network_architecture: List[int], optimizer: str,
                        learning_rate: float, connection_style: str, training_style: str, iteration: int,
                        converged_data: List[List[float]]) -> None:
    """
    This function is the heart of the experiment. It loads the data, runs the specific model with the given training
    parameters. Once the model is trained, it the graphs the results and appends experiments results to the list.
    :param epochs: The number of epochs to train for
    :param width: The width of each layer
    :param network_architecture: The network architecture, specifically a list for the number of neurons in each layer
    :param optimizer: The optimizer to use for training
    :param learning_rate: The learning rate to use
    :param connection_style: The connection style to use
    :param training_style: The training style to use
    :param iteration: The iteration of the experiment
    :param converged_data: The list of experiments results
    """

    x, t = df_utils.load_abs_data()

    did_converge, results, model = run_model(x, t, epochs, network_architecture, optimizer, learning_rate,
                                             connection_style, training_style)

    graph_utils.graph_results(model, learning_rate, network_architecture,
                              width, optimizer, iteration, training_style, did_converge)

    converged_data.append(results)


def run_experiments(optimizers: List[str], learning_rates: List[float], network_architectures: List[List[int]],
                    connection_styles: List[str], training_styles: List[str], iterations: int,
                    epochs: int, width: int, depths: List[int]) -> None:
    """
    This function runs all the experiments and saves the results to a csv file.
    :param optimizers: The optimizers to be used
    :param learning_rates: The learning rates to be used
    :param network_architectures: The network architectures to be used
    :param connection_styles: The connection styles to be used
    :param training_styles: The training styles to be used
    :param iterations: The number of iterations to run
    :param epochs: The number of epochs to train for
    :param width: The width of the networks
    :param depths: The depths of the networks
    """

    df_column_names = [
        "Iterative - Total Converged", "Iterative - Amount of Dead Neurons",
        "Iterative - Amount of Dead Layers", "Iterative - Total Time", "Batch - Total Converged",
        "Batch - Amount of Dead Neurons", "Batch - Amount of Dead Layers", "Batch - Total Time"
    ]

    df_index_names = []
    for depth in depths:
        df_index_names.append(f"Depth {depth} - Non Residual")
        df_index_names.append(f"Depth {depth} - Residual")
    df_index_names.append(" ")

    lr_utils.log_starting_experiment_message()

    for optimizer in optimizers:
        for learning_rate in learning_rates:
            df = pd.DataFrame(columns=df_column_names, index=df_index_names)
            df.index.name = "Network Architecture"

            for network_architecture in network_architectures:
                converged_iterative_data = []
                converged_batch_data = []

                for connection_style in connection_styles:
                    lr_utils.log_current_training_architecture(network_architecture, learning_rate,
                                                               connection_style, optimizer)

                    for training_style in training_styles:
                        converged_data = converged_iterative_data if training_style == "Iterative" \
                            else converged_batch_data

                        with alive_bar(iterations, bar="classic", spinner="classic") as bar:
                            for iteration in range(iterations):
                                heart_of_experiment(epochs,
                                                    width,
                                                    network_architecture,
                                                    optimizer,
                                                    learning_rate,
                                                    connection_style,
                                                    training_style,
                                                    iteration,
                                                    converged_data)
                                bar()

                    df.loc[f"Depth {len(network_architecture)} - {connection_style}"] = \
                        lr_utils.concat_iterative_and_batch_data(converged_iterative_data, converged_batch_data)
                    os.system("cls" if os.name == "nt" else "clear")

            df_utils.save_df_to_csv(df, width, depths, optimizer, learning_rate)
            graph_utils.graph_all_results(width)

    df_utils.combine_all_dfs_to_csv(width, optimizers)
    os.system("cls" if os.name == "nt" else "clear")
    lr_utils.log_end_of_all_training_message()


def run(width: int, depths: List[int], learning_rates: List[float], optimizers, epochs: int):
    """
    This python file is intended to be used with the "run_notebook_script.py" file.
    Experiment parameters are read via the experiment_vars.yaml file.
    :param width: The width of the networks
    :param depths: The depths of the networks
    :param learning_rates: The learning rates to be used
    :param optimizers: The optimizers to be used
    :param epochs: The number of epochs to train for
    """

    network_architectures = [[width] * d for d in depths]

    connection_styles = ["non_residual", "residual"]
    training_styles = ["batch", "iterative"]
    iterations = 10

    run_experiments(optimizers, learning_rates, network_architectures,
                    connection_styles, training_styles, iterations, epochs, width, depths)
