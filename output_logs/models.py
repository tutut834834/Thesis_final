import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# PC20 BASE CNN (kept for compatibility)
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
# PC26 NEW MODEL: CNN + SELF-ATTENTION
# =========================================================
class CNN_Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        # attention over spatial tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )

        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv(x)

        B, C, H, W = x.shape

        # reshape into sequence
        x = x.view(B, C, H * W).permute(0, 2, 1)

        # self-attention
        x, _ = self.attn(x, x, x)

        # global pooling
        x = x.mean(dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =========================================================
# MODEL SELECTOR
# =========================================================
def get_model(model_type="cnn"):
    if model_type == "cnn":
        return CNN_MNIST()

    elif model_type == "cnn_attention":
        return CNN_Attention()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")