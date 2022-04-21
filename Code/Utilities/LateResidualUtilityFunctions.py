import numpy as np
from termcolor import colored

def rmse(Y, T):
    '''
    Calculates the root mean squared error between the predicted values and the target values.
    '''
    return np.sqrt(np.mean((T - Y)**2))

def concat_iterative_and_batch_data(iterative_data, batch_data):
    '''
    Concatenates the iterative and batch data and returns the result.
    If either of the iterative or batch data is empty, then the data is 
    changed to [0, 0, 0, 0].
    '''
    iterative_data = np.array(iterative_data)
    batch_data = np.array(batch_data)
    
    if iterative_data.size != 0:
        iterative_data = np.around(np.insert(iterative_data, 0, iterative_data.shape[0], axis=1), decimals=3)
        iterative_data = np.around(np.median(iterative_data, axis=0), decimals=3)
    else:
        iterative_data = np.array([0, 0, 0, 0])

    if batch_data.size != 0:
        batch_data = np.around(np.insert(batch_data, 0, batch_data.shape[0], axis=1), decimals=3)
        batch_data = np.around(np.median(batch_data, axis=0), decimals=3)
    else:
        batch_data = np.array([0, 0, 0, 0])

    return np.concatenate((iterative_data, batch_data))

def print_starting_experiment_message():
    '''
    Prints a message to the console indicating that the experiment is starting.
    '''
    print(colored('\t=============================================== STARTING EXPERIMENT ===============================================', 'yellow', attrs=['blink']))

def print_current_training_architecture(network_architecture, learning_rate, connection_style, optimizer, color='blue'):
    ''' 
    Prints a message to the console indicating the current training architecture.
    '''
    training_info = f"\t\tNETWORK ARCHITECTURE: [WIDTH: {network_architecture[-1]}, DEPTH: {len(network_architecture)}] OPTIMIZER: {optimizer} LEARNING RATE: {learning_rate} CONNECTION: {connection_style}"
    training_hashes = ''.join(['#' for _ in range(len(" TRAINING "))])

    print(colored(f"\t{''.join(['#' for _ in range(len(training_info) // 2)])} TRAINING {''.join(['#' for _ in range(len(training_info) // 2)])}", color))
    print(colored(training_info, color))
    print(colored(f"\t{''.join(['#' for _ in range(len(training_info) // 2)])}{training_hashes}{''.join(['#' for _ in range(len(training_info) // 2)])}", color))


def print_end_of_all_training_message():
    '''
    Prints a message to the console indicating that the experiment is ending.
    '''
    print(colored('\t=============================================== EXPERIMENT COMPLETE ===============================================', 'green', attrs=['blink']))
