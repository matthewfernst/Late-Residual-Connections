import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Utilities.DataframeCode.LateResidualDataframe as dataframe_utils

def graph_results(model, learning_rate, network_architecture, width, optimizer, iteration, training_style, did_converge):
    '''
    Graphs the results of the experiment.
    The model is run and the results are graphed.
    The graphs are saved to the Graphs directory.
    '''
    X, T = dataframe_utils.load_abs_data()
    depth = f'{len(network_architecture)}'

    colors = {'Adam': 'blue', 'SGD': 'red', 'RMSprop': 'green', 'Adagrad': 'yellow', 'Adadelta': 'magenta', 'Adamax': 'cyan'}
    color = colors[optimizer]

    convergence = 'Convergence' if did_converge else 'No-Convergence'

    directory_path = f'../Graphs/Width-{width}/{optimizer}/LearningRate-{learning_rate}/{training_style}/{convergence}/Depth-{depth}/'
    dataframe_utils.make_directory_if_not_exists(directory_path)

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