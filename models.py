import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        # 🔥 FIX: compute flatten size dynamically
        self._to_linear = None

        x = torch.randn(1, 1, 28, 28)
        x = self._forward_conv(x)
        self._to_linear = x.shape[1]

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)