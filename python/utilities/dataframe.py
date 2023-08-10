import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import glob


def load_abs_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the data from the abs_data.csv file.
    """
    df = pd.read_csv(os.path.abspath('utilities/abs_data.csv'))
    x = df["X"].values
    x = x.reshape(-1, 1)

    t = df["T"].values
    t = t.reshape(-1, 1)

    assert x.shape == (20, 1)
    assert t.shape == (20, 1)
    return x, t


def make_directory_if_not_exists(directory_path: str) -> None:
    """
    Creates a directory if it does not exist. The directory path is relative to the current working directory.
    :param directory_path: The path of the directory to be created
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def save_df_to_csv(df: pd.DataFrame, width: int, depths: List[int], optimizer: str, learning_rate: float) -> None:
    """
    Saves a dataframe to a csv file. The file is saved based on the width, optimizer, and learning rate.
    :param df: The dataframe to be saved
    :param width: The width of the networks
    :param depths: The depths of the networks
    :param optimizer: The optimizer used
    :param learning_rate: The learning rate used
    """
    directory_path = f"Results/Width-{width}/{optimizer}"

    make_directory_if_not_exists(directory_path)

    full_path = f"{directory_path}/{optimizer}-LearningRate-{learning_rate}.csv"

    df.replace(to_replace=np.nan, value=" ", inplace=True)
    df.insert(len(df.columns), " ", [" " for _ in range(len(depths) * 2 + 1)], True)

    df.to_csv(full_path, index=True)


def combine_all_dfs_to_csv(width: int, optimizers: List[str]) -> None:
    """
    Combines all the dataframes for a given width and optimizer into one dataframe.
    The dataframe is saved to the Results' directory as 'AllData.csv' in the directory for the width.
    :param width: The width of the networks
    :param optimizers: The optimizers used
    """

    def get_combined_optimizer_csvs(optimizer: str) -> pd.DataFrame:
        """
        Gets all the csv files for a given optimizer and concatenates them into one dataframe.
        :param optimizer: The optimizer to get the csv files for
        :return: The concatenated dataframe
        """
        optimizer_csvs = glob.glob(f"Results/Width-{width}/{optimizer}/{optimizer}-LearningRate-*.csv")
        df_holder = []

        for filename in optimizer_csvs:
            df = pd.read_csv(filename)
            df_holder.append(df)

        return pd.concat(df_holder, axis=1)

    final = pd.concat([get_combined_optimizer_csvs(optimizer) for optimizer in optimizers], axis=0)
    final.to_csv(f"Results/Width-{width}/All-Results.csv", index=False)
