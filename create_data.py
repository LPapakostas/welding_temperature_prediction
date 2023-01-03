import pandas as pd
import random
import os
import sys
from sklearn.model_selection import train_test_split

COLUMN_NAMES = [
    'PlateThickness', 'InitialTemperature',
    'HeatInput', 'ElectrodeVelocity', 'X', 'Y', 'Z',
    'Max Temperature', 'Delta Time']

DATA_FILE_PATH = "/data/dataset_edited.csv"
NN1_DATASET_NAME = "/data/nn1/dataset.csv"
NN1_TRAIN_DATASET_NAME = "/data/nn1/train_dataset.csv"
NN1_TEST_DATASET_NAME = "/data/nn1/test_dataset.csv"

NN2_DATASET_NAME = "/data/nn2/dataset.csv"
NN2_TRAIN_DATASET_NAME = "/data/nn2/train_dataset.csv"
NN2_TEST_DATASET_NAME = "/data/nn2/test_dataset.csv"


if (__name__ == "__main__"):

    random.seed(45)

    # Create two datasets for each NN
    home_folder = os.getcwd()
    df = pd.read_csv(home_folder + DATA_FILE_PATH)

    nn1_df = df.drop("Delta Time", axis=1)
    nn2_df = df.drop("Max Temperature", axis=1)

    is_dt_one_exists = os.path.exists(home_folder + NN1_DATASET_NAME)
    is_dt_two_exists = os.path.exists(home_folder + NN2_DATASET_NAME)

    if not (is_dt_one_exists and is_dt_two_exists):
        nn1_df.to_csv(home_folder + NN1_DATASET_NAME,
                      index=False)
        nn2_df.to_csv(home_folder + NN2_DATASET_NAME,
                      index=False)

    # Split and save datasets into training/testing
    nn1_train_df, nn1_test_df = train_test_split(
        nn1_df, test_size=0.2, random_state=5)

    is_train_one_exists = os.path.exists(home_folder + NN1_TRAIN_DATASET_NAME)
    is_test_one_exists = os.path.exists(home_folder + NN1_TEST_DATASET_NAME)

    if not (is_train_one_exists and is_test_one_exists):
        nn1_train_df.to_csv(home_folder + NN1_TRAIN_DATASET_NAME, index=False)
        nn1_test_df.to_csv(home_folder + NN1_TEST_DATASET_NAME,
                           index=False)

    nn2_train_df, nn2_test_df = train_test_split(
        nn2_df, test_size=0.2, random_state=5)
    is_train_two_exists = os.path.exists(home_folder + NN2_TRAIN_DATASET_NAME)
    is_test_two_exists = os.path.exists(home_folder + NN2_TEST_DATASET_NAME)

    if not (is_train_two_exists and is_test_two_exists):
        nn2_train_df.to_csv(home_folder + NN2_TRAIN_DATASET_NAME, index=False)
        nn2_test_df.to_csv(home_folder + NN2_TEST_DATASET_NAME,
                           index=False)
