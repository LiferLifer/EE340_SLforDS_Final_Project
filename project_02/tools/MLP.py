import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, device='cpu'):
        super(MLP, self).__init__()
        self.device = torch.device(device)

        layers = []
        in_features = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def fit(self, X_train, y_train, epochs=100, lr=0.01):
        losses = []
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()

            # Forward pass
            outputs = self(X_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
                losses.append(loss.item())
        return np.array(losses)

    def predict(self, X):
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
        return predicted.to(self.device)