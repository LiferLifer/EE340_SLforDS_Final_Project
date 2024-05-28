import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_classes, n_conv_layers=2, conv_channels=None, device='cpu'):
        super(CNN, self).__init__()
        self.device = torch.device(device)

        if conv_channels is None:
            conv_channels = [16, 32]

        self.conv_layers = nn.ModuleList()
        in_channels = 1  # 因为我们使用单通道输入

        for out_channels in conv_channels[:n_conv_layers]:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            in_channels = out_channels

        # 动态计算全连接层的输入大小
        self.fc_input_size = self._get_fc_input_size()
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.to(self.device)

    def _get_fc_input_size(self):
        # 创建一个假输入，计算卷积层输出大小
        with torch.no_grad():
            x = torch.randn(1, 1, 224, 224).to(self.device)
            for layer in self.conv_layers:
                x = layer(x)
        return x.numel()

    def forward(self, x):
        x = x.to(self.device)  # 确保输入在同一个设备上
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x.to(self.device)))  # 确保在设备上
        x = self.fc2(x)
        return x

    def fit(self, dataloader, epochs=10, lr=0.001, momentum=0.9):
        self.to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                labels -= 1  # 由于我们的标签从1开始，所以我们需要减去1
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 确保数据在同一个设备上
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 100 == 99:
                    print(f'[Epoch {epoch + 1}, Mini-batch {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
        print('Finished Training')

    def predict(self, dataloader):
        self.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                labels -= 1  # 由于我们的标签从1开始，所以我们需要减去1
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 确保数据在同一个设备上
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted)
                all_labels.append(labels)
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        return all_predictions, all_labels
