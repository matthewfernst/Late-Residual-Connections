from typing import List

import torch
import numpy as np
from termcolor import colored


def rmse(y: torch.tensor, t: torch.tensor) -> np.ndarray:
    """
    Calculates the root mean squared error between the predicted values and the target values.
    :param y: The predicted values.
    :param t: The target values.
    """
    return np.sqrt(np.mean((t - y) ** 2))


def concat_iterative_and_batch_data(iterative_data: List[float], batch_data: List[float]) -> np.ndarray:
    """
    Concatenates the iterative and batch data and returns the result.
    If either of the iterative or batch data is empty, then the data is
    changed to [0, 0, 0, 0].
    :param iterative_data: The iterative data.
    :param batch_data: The batch data.
    """

    def configure_data(data):
        if len(data) != 0:
            data = np.insert(data, 0, len(data), axis=1)
            data = np.median(data, axis=0)
        else:
            data = np.array([0, 0, 0, 0])
        return np.around(data, decimals=3)

    return np.concatenate((configure_data(iterative_data), configure_data(batch_data)))


def log_starting_experiment_message():
    """
    Logs a message to the console indicating that the experiment is starting.
    """
    bar = 47 * '='
    print(colored(f"\t{bar} STARTING EXPERIMENT {bar}", "yellow", attrs=["blink"]))


def log_current_training_architecture(network_architecture: List[int], learning_rate: float, connection_style: str,
                                      optimizer: str, color: str = "blue") -> None:
    """
    Logs a message to the console indicating the current training architecture.
    :param network_architecture: The network architecture.
    :param learning_rate: The learning rate.
    :param connection_style: The connection style.
    :param optimizer: The optimizer.
    :param color: The color of the message.
    """
    training_hashes = '#' * len(" TRAINING ")
    training_info = f"\t\tNETWORK ARCHITECTURE: [WIDTH: {network_architecture[-1]}, DEPTH: {len(network_architecture)}] \
        OPTIMIZER: {optimizer} LEARNING RATE: {learning_rate} CONNECTION: {connection_style} "
    
    print(colored(f"\t{'#' * (len(training_info) // 2)} TRAINING {'#' * (len(training_info) // 2)}", color))
    print(colored(training_info, color))
    print(colored(f"\t{'#' * (len(training_info) // 2)}{training_hashes}{'#' * (len(training_info) // 2)}", color))

def log_end_of_all_training_message() -> None:
    """
    Logs a message to the console indicating that the experiment is ending.
    """
    bar = 47 * '='
    print(colored(f"\t{bar} END OF EXPERIMENT {bar}", "green", attrs=["blink"]))
