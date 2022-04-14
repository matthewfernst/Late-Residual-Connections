import os
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

def save_dataframe_to_csv(dataframe, width, optimizer, filename):
    directory_path = f'Results/Width-{width}/{optimizer}'

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    full_path = f'{directory_path}/{filename}.csv'

    dataframe.to_csv(full_path, index=False)
    

def rmse(Y, T):
    return np.sqrt(np.mean((T - Y)**2))

def display_graph(X, T, model, lr, n_hiddens, method, iteration, train_style, did_converge):
    depth = f'{len(n_hiddens)}'
    arch = f'[2 for _ in range({len(n_hiddens)})]'
    
    color = 'blue' if method =='Adam' else 'orange'

    conv_or_no_conv = 'c' if did_converge else 'nc'

    lr_dir = None
    if lr == 0.01:
        lr_dir = 'LR0_01'
    elif lr == 0.001:
        lr_dir = 'LR0_001'
    elif lr == 0.1:
        lr_dir = 'LR0_1'
    else:
        raise Exception('{lr} is not a correct learing rate')

    Y = model.use(X)

    plt.figure(figsize=(10,5))

    plt.suptitle(f'{arch}', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(model.error_trace, color=color, label=method)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.ylim((0.0, 0.3))
    plt.legend()

    plt.subplot(1, 2, 2)
    
    plt.plot(Y, '-s', color=color, label=method)
    plt.plot(T, '-o', color='green', label='Target')
    plt.xlabel('Sample')
    plt.ylabel('Target or Predicted')
    plt.legend()
    
    plt.savefig(f'Graphs/ResidualConnections/{train_style}/{lr_dir}/{method}_depth_{depth}_it{iteration + 1}_{conv_or_no_conv}.jpeg',  bbox_inches = 'tight')
    plt.close()

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

def combine_all_dataframes_to_csv():
    adam_csvs = glob.glob('Results/Adam/Adam-LearningRate-*.csv')
    sgd_csvs = glob.glob('Results/SGD/SGD-LearningRate-*.csv')

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
    final.to_csv('Results/All-Results.csv', index=False)


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
