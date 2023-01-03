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
DEBUG = True

NN1_DATASET_FILEPATH = "/data/nn1/dataset.csv"
NN1_TRAIN_DATASET_FILEPATH = "/data/nn1/train_dataset.csv"
NN1_TEST_DATASET_FILEPATH = "/data/nn1/test_dataset.csv"
NN1_VALIDATION_DATASET_FILEPATH = "/data/nn1/validation_dataset.csv"

NN2_DATASET_FILEPATH = "/data/nn2/dataset.csv"
NN2_TRAIN_DATASET_FILEPATH = "/data/nn2/train_dataset.csv"
NN2_TEST_DATASET_FILEPATH = "/data/nn2/test_dataset.csv"
NN2_VALIDATION_DATASET_FILEPATH = "/data/nn2/validation_dataset.csv"


if (__name__ == "__main__"):

    random.seed(4)

    # Create two datasets for each NN
    home_folder = os.getcwd()
    df = pd.read_csv(home_folder + DATA_FILE_PATH)

    nn1_df = df.drop("Delta Time", axis=1)
    nn2_df = df.drop("Max Temperature", axis=1)

    is_dt_one_exists = os.path.exists(home_folder + NN1_DATASET_FILEPATH)
    is_dt_two_exists = os.path.exists(home_folder + NN2_DATASET_FILEPATH)

    if not (is_dt_one_exists and is_dt_two_exists):
        nn1_df.to_csv(home_folder + NN1_DATASET_FILEPATH,
                      index=False)
        nn2_df.to_csv(home_folder + NN2_DATASET_FILEPATH,
                      index=False)

    # Split and save datasets into training/testing
    nn1_temp_df, nn1_test_df = train_test_split(
        nn1_df, test_size=0.2, random_state=2, shuffle=True)
    nn1_train_df, nn1_validation_df = train_test_split(
        nn1_temp_df, test_size=0.25, random_state=2, shuffle=True)

    if (DEBUG):
        print("NN1 Train length is: ", len(nn1_train_df))
        print("NN1 Test length is: ", len(nn1_test_df))
        print("NN1 Validation length is: ", len(nn1_validation_df))

        print("NN1 total length is: ", len(nn1_train_df) +
              len(nn1_test_df) + len(nn1_validation_df))
        print("NN1 total length is: ", len(nn1_df))

    is_train_one_exists = os.path.exists(
        home_folder + NN1_TRAIN_DATASET_FILEPATH)
    is_test_one_exists = os.path.exists(
        home_folder + NN1_TEST_DATASET_FILEPATH)
    is_valid_one_exists = os.path.exists(
        home_folder + NN1_VALIDATION_DATASET_FILEPATH)

    if not (is_train_one_exists and is_test_one_exists and is_valid_one_exists):
        nn1_train_df.to_csv(
            home_folder + NN1_TRAIN_DATASET_FILEPATH, index=False)
        nn1_test_df.to_csv(home_folder + NN1_TEST_DATASET_FILEPATH,
                           index=False)
        nn1_validation_df.to_csv(home_folder + NN1_VALIDATION_DATASET_FILEPATH,
                                 index=False)

    nn2_temp_df, nn2_test_df = train_test_split(
        nn2_df, test_size=0.2, random_state=3, shuffle=True)
    nn2_train_df, nn2_validation_df = train_test_split(
        nn2_temp_df, test_size=0.25, random_state=3, shuffle=True)

    if (DEBUG):
        print("NN2 Train length is: ", len(nn2_train_df))
        print("NN2 Test length is: ", len(nn2_test_df))
        print("NN2 Validation length is: ", len(nn2_validation_df))

        print("NN2 total length is: ", len(nn2_train_df) +
              len(nn2_test_df) + len(nn2_validation_df))
        print("NN2 total length is: ", len(nn2_df))

    is_train_two_exists = os.path.exists(
        home_folder + NN2_TRAIN_DATASET_FILEPATH)
    is_test_two_exists = os.path.exists(
        home_folder + NN2_TEST_DATASET_FILEPATH)
    is_validation_two_exists = os.path.exists(
        home_folder + NN2_VALIDATION_DATASET_FILEPATH)

    if not (is_train_two_exists and is_test_two_exists and is_validation_two_exists):
        nn2_train_df.to_csv(
            home_folder + NN2_TRAIN_DATASET_FILEPATH, index=False)
        nn2_test_df.to_csv(home_folder + NN2_TEST_DATASET_FILEPATH,
                           index=False)
        nn2_validation_df.to_csv(home_folder + NN2_VALIDATION_DATASET_FILEPATH,
                                 index=False)
