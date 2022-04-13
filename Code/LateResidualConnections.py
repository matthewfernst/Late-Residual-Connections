#!/usr/bin/env python
# coding: utf-8

# Personal Files Import 
import LateResidualNeuralNetwork
import LateResidualUtilityFunctions as lr_utils

import time
from IPython.display import clear_output
import pandas as pd
from tqdm import tqdm

def run_experiment(X, T, epochs, network_architecture, optimizer, learning_rate, connection_style, training_style, verbose=False):

    convergence_threshold = 0.07
    isResiduallyConnected = connection_style == 'Residual' 
    
    model = LateResidualNeuralNetwork.NNet(1, network_architecture, 1, optimizer, isResiduallyConnected=isResiduallyConnected) 
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

    return (final_rmse <= convergence_threshold, [dead_neurons, dead_layers, total_time])


def run_experiments(optimizers, learning_rates, network_architectures, connection_styles, training_styles, iterations, epochs):
    X, T = lr_utils.load_abs_data()

    dataframe_column_names = ['Iterative - Total Converged', 'Iterative - Amount of Dead Neurons', 'Iterative - Amount of Dead Layers', 'Iterative - Total Time', 
                                    'Batch - Total Converged', 'Batch - Amount of Dead Neurons', 'Batch - Amount of Dead Layers', 'Batch - Total Time']

    dataframe_index_names = ['Depth 2 - Non Residual', 'Depth 2 - Residual', 'Depth 5 - Non Residual', 'Depth 5 - Residual', 'Depth 10 - Non Residual', 'Depth 10 - Residual', 
                                    'Depth 20 - Non Residual', 'Depth 20 - Residual', 'Depth 25 - Non Residual', 'Depth 25 - Residual', 'Depth 30 - Non Residual', 'Depth 30 - Residual']


    lr_utils.print_starting_experiment_message()
    
    for optimizer in optimizers:
        for learning_rate in learning_rates:
                dataframe = pd.DataFrame(columns=dataframe_column_names, index=dataframe_index_names)
                dataframe_title = f"{optimizer}-LearningRate-{learning_rate}"

                for network_architecture in network_architectures:
                    converged_iterative_data = []
                    converged_batch_data = []

                    for connection_style in connection_styles:
                        lr_utils.print_current_training_architecture(network_architecture, learning_rate, connection_style, optimizer)
                        for training_style in training_styles: 
                            for _ in tqdm(range(iterations)):
                                did_converge, results = run_experiment(X, T, epochs, network_architecture, optimizer, learning_rate, connection_style, training_style)

                                if did_converge:
                                    if training_style == 'Iterative':
                                        converged_iterative_data.append(results)
                                    elif training_style == 'Batch':
                                        converged_batch_data.append(results)
                                    else:
                                        raise ValueError(f'Training Style {training_style} is not supported')

                        row_to_add = lr_utils.concat_iterative_and_batch_data(converged_iterative_data, converged_batch_data)
                        dataframe.loc[f'Depth {len(network_architecture)} - {connection_style}'] = row_to_add
                        
                    clear_output(wait=True)

                dataframe.to_csv(f'Results/Width-{network_architectures[-1][-1]}/{optimizer}/{dataframe_title}.csv')

    lr_utils.combine_all_dataframes_to_csv()
    lr_utils.print_end_of_all_training_message()





def run(width, depths, learning_rates, optimizers, epochs):
    # Experiment Parameters 
    network_architectures = []
    for depth in depths:
        network_architectures.append([width for _ in range(depth)])

    connection_styles = ['Non Residual', 'Residual']
    training_styles = ['Batch', 'Iterative']
    iterations = 10

    run_experiments(optimizers, learning_rates, network_architectures, 
                        connection_styles, training_styles, iterations, epochs)

