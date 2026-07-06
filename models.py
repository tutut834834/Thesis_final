import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1. BASELINE CNN
# =========================================================
class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================================================
# 2. DEEP CNN
# =========================================================
class CNN_DEEP(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================================================
# 3. WIDE CNN (PC22)
# =========================================================
class CNN_WIDE(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = self.pool(x)
        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        return self.fc2(x)


# =========================================================
# 4. NO DROPOUT CNN
# =========================================================
class CNN_NODROP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================================================
# 5. MLP
# =========================================================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =========================================================
# MODEL SELECTOR (IMPORTANT - OUTSIDE CLASSES)
# =========================================================
def get_model(model_type="cnn"):
    if model_type == "cnn":
        return CNN_MNIST()

    elif model_type == "cnn_deep":
        return CNN_DEEP()

    elif model_type == "cnn_wide":
        return CNN_WIDE()

    elif model_type == "cnn_nodrop":
        return CNN_NODROP()

    elif model_type == "mlp":
        return MLP()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")