import os
import pandas as pd
import glob

def load_abs_data():
    '''
    Loads the data from the ABSdata.csv file.
    '''
    # print current directory
    dataframe = pd.read_csv('Utilities/DataframeCode/ABSdata.csv')
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
