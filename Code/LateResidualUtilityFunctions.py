import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import colored
import glob 

def load_abs_data():
    dataframe = pd.read_csv('ABSdata.csv')
    X = dataframe['X'].values
    X = X.reshape(-1,1)

    T = dataframe['T'].values
    T = T.reshape(-1, 1)

    assert(X.shape == (20, 1))
    assert(T.shape == (20, 1))
    return X, T

def make_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_dataframe_to_csv(dataframe, width, optimizer, filename):
    directory_path = f'Results/Width-{width}/{optimizer}'

    make_directory_if_not_exists(directory_path)
    
    full_path = f'{directory_path}/{filename}.csv'

    dataframe.to_csv(full_path, index=True)
    

def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

def graph_results(model, learning_rate, network_architecture, width, optimizer, iteration, training_style, did_converge):
    X, T = load_abs_data()
    depth = f'{len(network_architecture)}'
    color = 'red' if optimizer =='Adam' else 'blue'
    convergence = 'Convergence' if did_converge else 'No-Convergence'

    directory_path = f'../Graphs/Width-{width}/{optimizer}/LearningRate-{learning_rate}/{training_style}/{convergence}/Depth-{depth}/'
    make_directory_if_not_exists(directory_path)

    filename = f'Iteration-{iteration + 1}'
    full_path = f'{directory_path}{filename}.jpeg'

    Y = model.use(X)

    plt.figure(figsize=(10,5))

    plt.suptitle(f'{optimizer}-Width-{width}-Depth{depth}', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(model.error_trace, color='orange', label=optimizer)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.ylim((0.0, 0.3))
    plt.legend()

    plt.subplot(1, 2, 2)
    
    plt.plot(Y, '-s', color=color, label=optimizer)
    plt.plot(T, '-o', color='green', label='Target')
    plt.xlabel('Sample')
    plt.ylabel('Target or Predicted')
    plt.legend()
    
    plt.savefig(full_path, bbox_inches = 'tight')
    plt.close('all')

def concat_iterative_and_batch_data(iterative_data, batch_data):
    if iterative_data:
        iterative_data = np.around(np.median(iterative_data, axis=0), decimals=3)
        iterative_data = np.around(np.insert(iterative_data, 0, iterative_data.shape[0]), decimals=3)
    else:
        iterative_data = np.array([0, 0, 0, 0])

    if batch_data:
        batch_data = np.around(np.median(batch_data, axis=0), decimals=3)
        batch_data = np.around(np.insert(batch_data, 0, batch_data.shape[0]), decimals=3)
    else:
        batch_data = np.array([0, 0, 0, 0])

    return np.concatenate((iterative_data, batch_data))

def combine_all_dataframes_to_csv(width):
    adam_csvs = glob.glob(f'Results/Width-{width}/Adam/Adam-LearningRate-*.csv')
    sgd_csvs = glob.glob(f'Results/Width-{width}/SGD/SGD-LearningRate-*.csv')

    current_dataframes_holder = []

    for filename in adam_csvs:
        df = pd.read_csv(filename)
        current_dataframes_holder.append(df)

    adam_dataframe = pd.concat(current_dataframes_holder, axis=1)

    current_dataframes_holder = []

    for filename in sgd_csvs:
        df = pd.read_csv(filename)
        current_dataframes_holder.append(df)

    sgd_dataframe = pd.concat(current_dataframes_holder, axis=1)

    final = pd.concat([adam_dataframe, sgd_dataframe], axis=0)
    final.to_csv(f'Results/Width-{width}/All-Results.csv', index=False)


def print_starting_experiment_message():
    print(colored('\t=============================================== STARTING EXPERIMENT ===============================================', 'yellow', attrs=['blink']))

def print_current_training_architecture(network_architecture, learning_rate, connection_style, optimizer, color='blue'):
    training_info = f"\t\tNETWORK ARCHITECTURE: [WIDTH: {network_architecture[-1]}, DEPTH: {len(network_architecture)}] OPTIMIZER: {optimizer} LEARNING RATE: {learning_rate} CONNECTION: {connection_style}"
    training_hashes = ''.join(['#' for _ in range(len(" TRAINING "))])

    print(colored(f"\t{''.join(['#' for _ in range(len(training_info) // 2)])} TRAINING {''.join(['#' for _ in range(len(training_info) // 2)])}", color))
    print(colored(training_info, color))
    print(colored(f"\t{''.join(['#' for _ in range(len(training_info) // 2)])}{training_hashes}{''.join(['#' for _ in range(len(training_info) // 2)])}", color))


def print_end_of_all_training_message():
    print(colored('\t=============================================== EXPERIMENT COMPLETE ===============================================', 'green', attrs=['blink']))
