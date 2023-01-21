# -*- coding: utf-8 -*-

# Imports
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tqdm
import os
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             median_absolute_error,
                             max_error, r2_score)
import pickle

# Class Declarations


class DataMaker(torch.utils.data.Dataset):
    """
    Prepare Dataset for regression model
    """

    def __init__(self, X, y):
        self.targets = X.astype(np.float32)
        self.labels = y.astype(np.float32)
        return

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.targets[i, :], self.labels[i]


class MaxTemperatureNN(nn.Module):
    """
    Neural Network Model for maximum temperature regression
    of a welding process
    """

    def __init__(self, input=13, output=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, output)
        )

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction


# HYPER PARAMETERS
torch.manual_seed(42)
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG = False
DATA_DIRECTORY = "/data/nn1/"
DATA_PATH = os.getcwd() + DATA_DIRECTORY
PARAMETERS_SAVE_PATH = os.getcwd(
) + "/nn_weights/MaximumTemperatureNN_training_preprocess_values.pickle"
MODEL_SAVE_PATH = os.getcwd() + "/nn_weights/MaximumTemperatureNN_Model_Parameters.pt"

# Load datasets into pandas DataFrame and split inputs/targets
train_df = pd.read_csv(DATA_PATH + "train_dataset.csv")
X_train, Y_train = train_df.iloc[:, 0:-1], train_df.iloc[:, -1]

validation_df = pd.read_csv(DATA_PATH + "validation_dataset.csv")
X_validation, Y_validation = validation_df.iloc[:,
                                                0:-1], validation_df.iloc[:, -1]

test_df = pd.read_csv(DATA_PATH + "test_dataset.csv")
X_test, Y_test = test_df.iloc[:, 0:-1], test_df.iloc[:, -1]

# Apply pre-processing in respect to training values (mean and std)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype("float32")

# NOTE: use mean and std of training set
X_validation_scaled = scaler.transform(X_validation).astype("float32")
X_test_scaled = scaler.transform(X_test).astype("float32")

Y_train_scaled = Y_train.astype("float32")
Y_validation_scaled = Y_validation.astype("float32")
Y_test_scaled = Y_test.astype("float32")

# Check if it works
if (DEBUG):
    print("Means of training set:\n", X_train_scaled.mean(axis=0), "\n")
    print("Standard deviations of training set:\n",
          X_train_scaled.std(axis=0), "\n\n")

    print("Means of validation set:\n", X_validation_scaled.mean(axis=0), "\n")
    print("Standard deviations of validation set:\n",
          X_validation_scaled.std(axis=0), "\n\n")

    print("Means of test set:\n", X_test_scaled.mean(axis=0), "\n")
    print("Standard deviations of test set:\n",
          X_test_scaled.std(axis=0), "\n\n")

# Save preprocessing values to pickle
mean_values = scaler.mean_
std_values = scaler.scale_
train_dict = {
    "mean": mean_values,
    "std": std_values
}

with open(PARAMETERS_SAVE_PATH, 'wb') as handle:
    pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create DataLoaders for NN
train_dataset = DataMaker(X_train_scaled, Y_train_scaled)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

validation_dataset = DataMaker(X_validation_scaled, Y_validation_scaled)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

# Define Neural Network model
input_size = len(X_train.columns)
model = MaxTemperatureNN(input=input_size, output=1)
model = model.to(device)
criterion = nn.functional.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

total_train_losses, total_validation_losses = [], []

for epoch in range(EPOCHS):
    training_progress_bar = tqdm.tqdm(train_loader, leave=False)
    validation_progress_bar = tqdm.tqdm(
        validation_loader, leave=False)
    train_loss, valid_loss = 0.0, 0.0

    # *==== Training Procedure ====*
    model.train()
    for data, targets in training_progress_bar:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, torch.unsqueeze(targets, dim=1))
        loss.backward()
        optimizer.step()
        training_progress_bar.set_description(
            f'Training Loss: {loss.item():.3f}')
        train_loss += loss.item()

    # *==== Validation Procedure ====*
    with torch.no_grad():
        model.eval()
        for data, targets in validation_progress_bar:
            data, targets = data.to(device), targets.to(device)
            y_pred = model(data)
            targets = torch.unsqueeze(targets, dim=1)
            loss = criterion(y_pred, targets)
            validation_progress_bar.set_description(
                f'Validation Loss: {loss.item():.3f}')
            valid_loss += loss.item()

        # If the validation loss of the model is lower than that of all the
        # previous epochs, save the model state
        mean_val_loss = valid_loss / len(validation_loader)
        if (epoch == 0):
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        elif (epoch > 0) and (mean_val_loss < np.min(total_validation_losses)):
            print("Model Selection!")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # if (epoch % 10 == 0) :
    message = f'Epoch {epoch} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(validation_loader)}'
    tqdm.tqdm.write(message)

    total_train_losses.append(train_loss / len(train_loader))
    total_validation_losses.append(valid_loss / len(validation_loader))

# Print validation/training/testing losses
epochs_list = [e+1 for e in range(EPOCHS)]
plt.plot(epochs_list, total_train_losses, label="Training")
plt.plot(epochs_list, total_validation_losses, label="Validation")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Neural Network Evaluation
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to("cpu")
y_pred = model.predict(X_test_scaled)

print("Performance report")

print(f"Mean Squared Error: {mean_squared_error(Y_test_scaled, y_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(Y_test_scaled, y_pred):.2f}")
print(
    f"Median Absolute Error: {median_absolute_error(Y_test_scaled, y_pred):.2f}")
print(f"Max Error: {max_error(Y_test_scaled, y_pred):.2f}")
print(f"R2 score: {r2_score(Y_test_scaled, y_pred):.2f}")

plt.errorbar(Y_test_scaled, y_pred, fmt='bo', label="True values")
plt.xlabel("True Max Temperature")
plt.ylabel("Predicted Max Temperature")
plt.legend(loc="upper right")
plt.show()
plt.close()
