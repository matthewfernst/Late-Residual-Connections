import numpy as np
from termcolor import colored


def rmse(Y, T):
    """
    Calculates the root mean squared error between the predicted values and the target values.
    """
    return np.sqrt(np.mean((T - Y) ** 2))


def concat_iterative_and_batch_data(iterative_data, batch_data):
    """
    Concatenates the iterative and batch data and returns the result.
    If either of the iterative or batch data is empty, then the data is
    changed to [0, 0, 0, 0].
    """

    def configure_data(data):
        if len(data) != 0:
            data = np.insert(data, 0, len(data), axis=1)
            data = np.median(data, axis=0)
        else:
            data = np.array([0, 0, 0, 0])
        return np.around(data, decimals=3)

    return np.concatenate((configure_data(iterative_data), configure_data(batch_data)))


def print_starting_experiment_message():
    """
    Prints a message to the console indicating that the experiment is starting.
    """
    print(colored(
        '\t=============================================== STARTING EXPERIMENT '
        '===============================================',
        'yellow', attrs=['blink']))


def print_current_training_architecture(network_architecture, learning_rate, connection_style, optimizer, color='blue'):
    """
    Prints a message to the console indicating the current training architecture.
    """
    training_info = f"\t\tNETWORK ARCHITECTURE: [WIDTH: {network_architecture[-1]}, DEPTH: {len(network_architecture)}] OPTIMIZER: {optimizer} LEARNING RATE: {learning_rate} CONNECTION: {connection_style} "
    training_hashes = ''.join(['#' for _ in range(len(" TRAINING "))])

    print(colored(
        f"\t{''.join(['#' for _ in range(len(training_info) // 2)])} TRAINING {''.join(['#' for _ in range(len(training_info) // 2)])}", color))
    print(colored(training_info, color))
    print(colored(
        f"\t{''.join(['#' for _ in range(len(training_info) // 2)])}{training_hashes}{''.join(['#' for _ in range(len(training_info) // 2)])}", color))


def print_end_of_all_training_message():
    """
    Prints a message to the console indicating that the experiment is ending.
    """
    print(colored(
        '\t=============================================== EXPERIMENT COMPLETE '
        '===============================================',
        'green', attrs=['blink']))
