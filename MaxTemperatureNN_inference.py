import torch
import os
import numpy as np

# *==== Define Hyperparameters ====*
torch.manual_seed(42)
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.005

# *==== Define paths ====*

WEIGHTS_PATH = "/nn_weights/MaximumTemperatureNN_Model_Parameters.pt"

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


if (__name__ == "__main__"):

    home_folder = os.getcwd()

    # Define Network
    model = MaxTemperatureNN(input=7, output=1)
    model.load_state_dict(torch.load(
        home_folder + WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.to("cpu")

    # Set input for prediction
    #  NOTE: Max Temperature 618.150712
    plate_thickness = 0.003  # in [m]
    initial_temperatute = 180  # in [Celcius]
    heat_input = 1200  # in [Celcius]
    electrode_velocity = 0.004  # in [m/s]
    x = 0.05  # in [m]
    y = 0.025  # in [m]
    z = 0.0  # in [m]
    input = np.array([[plate_thickness, initial_temperatute, heat_input,
                     electrode_velocity, x, y, z]]).astype(np.float32)

    # TODO: Need to apply preprocess value

    # predict
    max_temperature = model.predict(input)
    print(
        f"Maximum Temperature prediction for given input is {max_temperature[0][0]}")

    # TODO: Use argparse to read input parameters
