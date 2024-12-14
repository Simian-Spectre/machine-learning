import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from csv_reader import np_arrays_from_csv

class MLP(nn.Module):
    def __init__(self, input_size, width, depth, activation):
        super(MLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, width))
        if activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(width, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_errors = []
    test_errors = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_errors.append(epoch_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for X, y in test_loader:
                outputs = model(X)
                loss = criterion(outputs, y)
                test_loss += loss.item()
            test_errors.append(test_loss / len(test_loader))

    return train_errors, test_errors


def prepare_data(X_train, y_train, X_test, y_test):
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=32,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_test, y_test),
        batch_size=32,
        shuffle=False
    )
    return train_loader, test_loader

def initialize_weights(m, activation):
    if isinstance(m, nn.Linear):
        if activation == 'tanh':
            nn.init.xavier_uniform_(m.weight)
        elif activation == 'relu':
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def run_batch(X_train, y_train, X_test, y_test, depths, widths, activation, epochs):
    results = {}

    train_loader, test_loader = prepare_data(X_train, y_train, X_test, y_test)
    input_size = X_train.shape[1]

    for depth in depths:
        for width in widths:
            print(f"Training model with depth={depth}, width={width}, activation={activation}")
            
            model = MLP(input_size, width, depth, activation)
            model.apply(lambda m: initialize_weights(m, activation))
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            train_errors, test_errors = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)

            results[(depth, width)] = {
                "final_train_error": train_errors[-1],
                "final_test_error": test_errors[-1],
            }

    return results


if __name__ == "__main__":
    # Loading data
    num_features = 4
    train_data_file = 'Perceptron\\Bank Note Data\\train.csv'
    eval_data_file = 'Perceptron\\Bank Note Data\\test.csv'
    train_features, train_labels = np_arrays_from_csv(train_data_file, num_features)
    eval_features, eval_labels = np_arrays_from_csv(eval_data_file, num_features)
    train_labels = train_labels.reshape(-1, 1)
    eval_labels = eval_labels.reshape(-1, 1)

    # Parameters
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    epochs = 100

    # tanh
    results_tanh = run_batch(train_features, train_labels, eval_features, eval_labels, depths, widths, "tanh", epochs)
    print("Results for tanh with Xavier initialization:", results_tanh)

    # relU
    results_relu = run_batch(train_features, train_labels, eval_features, eval_labels, depths, widths, "relu", epochs)
    print("Results for ReLU with He initialization:", results_relu)
