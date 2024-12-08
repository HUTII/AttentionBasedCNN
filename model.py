import torch.nn as nn


# Define SE module
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)  # Compression
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)  # Expansion
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Perform global average pooling
        y = self.global_avg_pool(x).view(batch_size, channels)
        # Generate attention weights via fully connected layers
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y))))
        # Channel weighting
        y = y.view(batch_size, channels, 1, 1)
        return x * y  # Apply channel weighting


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional layer -> BN -> ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        # Second convolutional layer -> BN -> ReLU -> Pooling
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer 1 -> BN -> ReLU -> Dropout
        x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))
        # Fully connected layer 2
        x = self.fc2(x)

        return x


class SE_CNN(nn.Module):
    def __init__(self, num_classes):
        super(SE_CNN, self).__init__()
        # First convolutional layer + SE module
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)  # Add SE module

        # Second convolutional layer + SE module
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64)  # Add SE module

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

        # Activation function and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional layer -> BN -> SE -> ReLU
        x = self.relu(self.se1(self.bn1(self.conv1(x))))
        # Second convolutional layer -> BN -> SE -> ReLU -> Pooling
        x = self.pool(self.relu(self.se2(self.bn2(self.conv2(x)))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer 1 -> BN -> ReLU -> Dropout
        x = self.dropout(self.relu(self.bn_fc1(self.fc1(x))))
        # Fully connected layer 2
        x = self.fc2(x)

        return x
