import torch
import numpy as np

import os
import sys
import pickle
import argparse

# *==== Define Hyperparameters ====*
torch.manual_seed(42)
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.005

# *==== Define paths ====*

WEIGHTS_PATH = "/nn_weights/MaximumTemperatureNN_Model_Parameters.pt"
PREPROCESS_VALUES_PATH = "/nn_weights/MaxTemperatureNN_training_preprocess_values.pickle"

# *==== Class Definitions ====*


class MaxTemperatureNN(torch.nn.Module):
    """
    Neural Network Model for maximum temperature regression
    of a welding process
    """

    def __init__(self, input=13, output=1):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, output)
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction

# *==== Method definitions ====*


def normalize(input, mean, std):
    return (input - mean) / std


def load_values():
    # Define Network
    home_folder = os.getcwd()
    model = MaxTemperatureNN(input=7, output=1)
    model.load_state_dict(torch.load(
        home_folder + WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.to("cpu")

    # Read and save preporcessing values
    filename = home_folder + PREPROCESS_VALUES_PATH
    with open(filename, 'rb') as handle:
        train_values_data = pickle.load(handle)
    mean = train_values_data["mean"]
    std = train_values_data["std"]

    return model, mean, std


if (__name__ == "__main__"):

    argp = argparse.ArgumentParser(
        description="Predict maximum temperature of weilding process, given desired inputs")
    argp.add_argument("--plate_thickness", required=False,
                      help="Plate thickness in [m]")
    argp.add_argument("--initial_temperature", required=False,
                      help="Initial Temperature in [K]")
    argp.add_argument("--heat_input", required=False,
                      help="Heat Input ")
    argp.add_argument("--electrode_velocity", required=False,
                      help="Electrode Velocity in [m/s]")
    argp.add_argument("--X", required=False,
                      help="X coordinate in [m]")
    argp.add_argument("--Y", required=False,
                      help="Y coordinate in [m]")
    argp.add_argument("--Z", required=False,
                      help="Z coordinate in [m]")
    args = vars(argp.parse_args())

    # Error handling about given values in case a field is empty
    for keys, value in args.items():
        if (type(value) == type(None)):
            print("Please give value to all fields")
            print("Closing...")
            sys.exit(1)

    # Set input for prediction
    # 0.004, 180, 900, 0.004, 0.0, 0.02, 0.002,
    # 803.545124
    plate_thickness = float(args["plate_thickness"])
    initial_temperature = float(args["initial_temperature"])
    heat_input = float(args["heat_input"])
    electrode_velocity = float(args["electrode_velocity"])
    x, y, z = float(args["X"]), float(args["Y"]), float(args["Z"])
    input = np.array([[plate_thickness, initial_temperature, heat_input,
                     electrode_velocity, x, y, z]])

    # Load predifined model with trained weights,
    # as well as preprocessing values
    model, mean, std = load_values()

    # Apply preprocessing
    input_scaled = normalize(input, mean, std).astype(np.float32)

    # Predict maximum temperature
    max_temperature = model.predict(input_scaled)
    print(
        f"Maximum Temperature prediction for given input is {max_temperature[0][0]:.2f}")
