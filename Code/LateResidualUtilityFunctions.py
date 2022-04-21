import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import colored
import glob 

def load_abs_data():
    '''
    Loads the data from the ABSdata.csv file.
    '''
    dataframe = pd.read_csv('ABSdata.csv')
    X = dataframe['X'].values
    X = X.reshape(-1,1)

    T = dataframe['T'].values
    T = T.reshape(-1, 1)

    assert(X.shape == (20, 1))
    assert(T.shape == (20, 1))
    return X, T

def make_directory_if_not_exists(directory_path):
    '''
    Creates a directory if it does not exist. The directory path is relative to the current working directory.
    '''
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_dataframe_to_csv(dataframe, width, optimizer, learning_rate):
    '''
    Saves a dataframe to a csv file. The file is saved based on the width, optimizer, and learning rate.
    '''
    directory_path = f'Results/Width-{width}/{optimizer}'

    make_directory_if_not_exists(directory_path)
    
    full_path = f'{directory_path}/{optimizer}-LearningRate-{learning_rate}.csv'

    dataframe.to_csv(full_path, index=True)
    

def rmse(Y, T):
    '''
    Calculates the root mean squared error between the predicted values and the target values.
    '''
    return np.sqrt(np.mean((T - Y)**2))


def graph_results(model, learning_rate, network_architecture, width, optimizer, iteration, training_style, did_converge):
    '''
    Graphs the results of the experiment.
    The model is run and the results are graphed.
    The graphs are saved to the Graphs directory.
    '''
    X, T = load_abs_data()
    depth = f'{len(network_architecture)}'

    colors = {'Adam': 'blue', 'SGD': 'red', 'RMSprop': 'green', 'Adagrad': 'yellow', 'Adadelta': 'magenta', 'Adamax': 'cyan'}
    color = colors[optimizer]

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
    '''
    Concatenates the iterative and batch data and returns the result.
    If either of the iterative or batch data is empty, then the data is 
    changed to [0, 0, 0, 0].
    '''
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

def combine_all_dataframes_to_csv(width, optimizers):
    '''
    Combines all of the dataframes for a given width and optimizer into one dataframe.
    The dataframe is saved to the Results directory as 'AllData.csv' in the directory for the width.
    '''

    def get_combined_optimizer_csvs(optimizer):
        optimizer_csvs = glob.glob(f'Results/Width-{width}/{optimizer}/{optimizer}-LearningRate-*.csv')
        dataframes_holder = []

        for filename in optimizer_csvs:
            df = pd.read_csv(filename)
            dataframes_holder.append(df)

        return pd.concat(dataframes_holder, axis=1)

    final = pd.concat([get_combined_optimizer_csvs(optimizer) for optimizer in optimizers], axis=0)
    final.to_csv(f'Results/Width-{width}/All-Results.csv', index=False)




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
