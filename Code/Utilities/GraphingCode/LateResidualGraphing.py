import os 

import glob as glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import Utilities.DataframeCode.LateResidualDataframe as df_utils


def graph_results(model, learning_rate, network_architecture, width, optimizer, iteration, training_style,
                  did_converge):
    """
    Graphs the results of the experiment.
    The model is run and the results are graphed.
    The graphs are saved to the Graphs directory.
    """
    X, T = df_utils.load_abs_data()
    depth = f'{len(network_architecture)}'

    colors = {'Adam': 'blue', 'SGD': 'red', 'RMSprop': 'green', 'Adagrad': 'yellow', 'Adadelta': 'magenta',
              'Adamax': 'cyan'}
    color = colors[optimizer]

    convergence = 'Convergence' if did_converge else 'No-Convergence'

    directory_path = f'Graphs/Width-{width}/{optimizer}/LearningRate-{learning_rate}/{training_style}/{convergence}/Depth-{depth}/ '
    df_utils.make_directory_if_not_exists(directory_path)

    filename = f'Iteration-{iteration + 1}'
    full_path = os.path.join(directory_path, f'{filename}.jpeg')

    Y = model.use(X)

    plt.figure(figsize=(10, 5))

    plt.suptitle(f'{optimizer}-Width-{width}-Depth{depth}', fontsize=16)
    plt.subplot(1, 2, 1)
    if model.device != 'cpu':
        model.error_trace = [tensor.cpu().detach().numpy()
                             for tensor in model.error_trace]
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

    plt.savefig(full_path, bbox_inches='tight')
    plt.close('all')


def graph_all_results(width):
    def graph_bar_results(dead_neurons_data, optimizer, training_style, learning_rate, x_ticks):
        X = np.arange(len(dead_neurons_data[-1]))
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1.5, 1.5])
        ax.set_ylabel('Number of Dead Neurons')
        ax.set_xlabel('Network Architecture')
        ax.set_title(f'Number of Dead Neurons vs Non-Residual and Late Residual Networks\n\
            {training_style} - {optimizer} - Learning Rate - {learning_rate}')
        ax.bar(X, dead_neurons_data[0], color='steelblue', width=0.25)
        ax.bar(X + 0.25, dead_neurons_data[1], color='darkorange', width=0.25)
        ax.legend(labels=['Non-Residual', 'Residual'])
        ax.set_xticks(np.arange(len(dead_neurons_data[-1])), x_ticks)

        directory_path = f'../Graphs/Width-{width}/{optimizer}/LearningRate-{learning_rate}/{training_style}/'
        filename = f'All-Results-DeadNeurons-{training_style}'
        full_path = f'{directory_path}{filename}.jpeg'

        plt.savefig(full_path, bbox_inches='tight')
        plt.close('all')

    def filtered_converged_data(df, training_style):
        dead_neurons_data = df[f'{training_style} - Amount of Dead Neurons'][:-1].astype(
            float).values
        total_converged = df[f'{training_style} - Total Converged'][:-
                                                                    1:2].astype(float).values
        x_ticks = list(
            map(lambda x: x[:x.index('-')], df['Network Architecture'].values[:-1:2]))
        indices_of_no_convergence = np.where(total_converged == 0)[0]

        dead_neurons_non_residual = np.delete(
            dead_neurons_data[:-1:2], indices_of_no_convergence)
        dead_neurons_residual = np.delete(
            dead_neurons_data[1::2], indices_of_no_convergence)
        x_ticks = np.delete(x_ticks, indices_of_no_convergence)

        return [dead_neurons_non_residual, dead_neurons_residual], x_ticks

    all_csvs = glob.glob(f'Results/Width-{width}/*/*.csv')
    for csv in all_csvs:
        df = pd.read_csv(csv)
        optimizer = csv.split('/')[-1].split('-')[0]
        learning_rate = csv.split('/')[-1].split('-')[-1]
        learning_rate = learning_rate[:learning_rate.rindex('.')]

        for training_style in ['Iterative', 'Batch']:
            dead_neurons_data, x_ticks = filtered_converged_data(
                df, training_style)
            graph_bar_results(dead_neurons_data, optimizer,
                              training_style, learning_rate, x_ticks)
