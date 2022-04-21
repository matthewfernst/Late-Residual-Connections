#!/usr/bin/env python
# coding: utf-8

import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# Personal Files Import 
import LateResidualPyTorch.LateResidualNeuralNetwork as LateResidualNeuralNetwork
import Utilities.LateResidualUtilityFunctions as lr_utils
import Utilities.GraphingCode.LateResidualGraphing as graph_utils
import Utilities.DataframeCode.LateResidualDataframe as dataframe_utils


def run_experiment(X, T, epochs, network_architecture, optimizer, learning_rate, connection_style, training_style, verbose=False):

    convergence_threshold = 0.07
    
    model = LateResidualNeuralNetwork.NNet(1, network_architecture, 1, optimizer, isResiduallyConnected=(connection_style == 'Residual')) 
    start = time.time()
    model.train(X, T, epochs, learning_rate, training_style, verbose=verbose)
    end = time.time()
    total_time = end - start
    Y = model.use(X)
    final_rmse = lr_utils.rmse(Y, T)
    
    if verbose:
        print(f'Total Time to Train {(total_time):.3f} seconds')
        print(f'RMSE {final_rmse:.3f}\n')

    dead_neurons, dead_layers = model.dead_neurons()
    
    return (final_rmse <= convergence_threshold, [dead_neurons, dead_layers, total_time], model)


def run_experiments(optimizers, learning_rates, network_architectures, connection_styles, training_styles, iterations, epochs, width, depths):
    X, T = dataframe_utils.load_abs_data()

    dataframe_column_names = ['Iterative - Total Converged', 'Iterative - Amount of Dead Neurons', 'Iterative - Amount of Dead Layers', 'Iterative - Total Time', 
                                    'Batch - Total Converged', 'Batch - Amount of Dead Neurons', 'Batch - Amount of Dead Layers', 'Batch - Total Time']

    dataframe_index_names = []
    for depth in depths:
        dataframe_index_names.append(f'Depth {depth} - Non Residual')
        dataframe_index_names.append(f'Depth {depth} - Residual')
    dataframe_index_names.append(' ')

    lr_utils.print_starting_experiment_message()
    
    for optimizer in optimizers:
        for learning_rate in learning_rates:
                dataframe = pd.DataFrame(columns=dataframe_column_names, index=dataframe_index_names)
                dataframe.index.name = "Network Architecture"

                for network_architecture in network_architectures:
                    converged_iterative_data = []
                    converged_batch_data = []

                    for connection_style in connection_styles:
                        lr_utils.print_current_training_architecture(network_architecture, learning_rate, connection_style, optimizer)

                        for training_style in training_styles: 
                            for iteration in tqdm(range(iterations)):
                                did_converge, results, model = run_experiment(X, T, epochs, network_architecture, optimizer, learning_rate, connection_style, training_style)
                                graph_utils.graph_results(model, learning_rate, network_architecture, width, optimizer, iteration, training_style, did_converge)
                                
                                if did_converge:
                                    if training_style == 'Iterative':
                                        converged_iterative_data.append(results)
                                    elif training_style == 'Batch':
                                        converged_batch_data.append(results)
                                    else:
                                        raise ValueError(f'Training Style {training_style} is not supported')

                        dataframe.loc[f'Depth {len(network_architecture)} - {connection_style}'] = \
                                            lr_utils.concat_iterative_and_batch_data(np.array(converged_iterative_data), np.array(converged_batch_data))
                        
                dataframe.replace(to_replace=np.nan, value=' ', inplace=True)
                dataframe.insert(len(dataframe.columns), ' ', [' ' for _ in range(len(depths) * 2 + 1)], True)
                dataframe_utils.save_dataframe_to_csv(dataframe, width, optimizer, learning_rate)
                
    dataframe_utils.combine_all_dataframes_to_csv(width, optimizers)
    lr_utils.print_end_of_all_training_message()




def run(width, depths, learning_rates, optimizers, epochs):
    '''
    This python file is intended to be used with the 'run_notebook_script.py' file. 
    Experiment parameters are passed as command line aruments.
    '''
    network_architectures = []
    for depth in depths:
        network_architectures.append([width for _ in range(depth)])

    connection_styles = ['Non Residual', 'Residual']
    training_styles = ['Batch', 'Iterative']
    iterations = 10

    run_experiments(optimizers, learning_rates, network_architectures, 
                        connection_styles, training_styles, iterations, epochs, width, depths)

