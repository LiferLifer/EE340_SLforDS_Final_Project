import torch

class KNN:
    def __init__(self, k=3, device='cpu'):
        self.k = k
        self.device = torch.device(device)

    def fit(self, X_train, y_train):
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)

    def predict(self, X_test):
        X_test = X_test.to(self.device)
        distances = self.compute_distances(X_test)
        return self.get_predictions(distances).to(self.device)

    def compute_distances(self, X_test):
        # Calculate L2 distances between each test point and all training points
        dists = torch.cdist(X_test, self.X_train)
        return dists

    def get_predictions(self, distances):
        # Get the k nearest neighbors and their corresponding labels
        _, knn_indices = distances.topk(self.k, largest=False)
        knn_labels = self.y_train[knn_indices]

        # Predict the most common class among the nearest neighbors
        knn_labels = knn_labels.cpu().numpy()
        mode, _ = torch.mode(torch.tensor(knn_labels), dim=1)
        return mode