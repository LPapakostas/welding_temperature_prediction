import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tqdm

# *==== Dataset Class ====*


class BostonDataset(torch.utils.data.Dataset):
    """
    Prepare Boston Dataset for regression model
    """

    def __init__(self, X, y, scale_data=True):
        """
        """
        are_tensors = torch.is_tensor(X) and torch.is_tensor(y)
        if (are_tensors):
            if (scale_data):
                X = StandardScaler.fit_transform(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# *==== Neural Network ====*


class MaxTemperatureNN(nn.Module):
    """
    Neural Network Model for maximum temperature regression
    of a welding process
    """

    def __init__(self, input=13, output=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output)
        )

    def forward(self, x):
        return self.layers(x)


if (__name__ == "__main__"):

    # Set fixed seed and define hyperparameters
    torch.manual_seed(42)
    BATCH_SIZE = 10
    EPOCHS = 20
    LEARNING_RATE = 0.01

    # Load Boston dataset and split it into train/test
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    train_dataset = BostonDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_dataset = BostonDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # Define Neural Network model
    model = MaxTemperatureNN()
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start training procedure
    total_losses = []
    for epoch in range(EPOCHS):
        progress_bar = tqdm.notebook.tqdm(train_loader, leave=False)
        total = 0
        losses = []
        for inputs, target in progress_bar:
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, torch.unsqueeze(target, dim=1))
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Loss: {loss.item():.3f}')
            losses.append(loss.item())
            total += 1
        epoch_loss = sum(losses) / total
        total_losses.append(epoch_loss)
        mess = f"Epoch #{epoch+1}\tLoss: {total_losses[-1]:.3f}"
        tqdm.tqdm.write(mess)

    # Testing procedure
    y_pred, y_true = [], []
    model.train(False)
    for inputs, targets in test_loader:
        y_pred.extend(model(inputs).data.numpy())
        y_true.extend(targets.numpy())
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("R^2:", r2_score(y_true, y_pred))
