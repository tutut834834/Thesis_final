import torch
import torch.nn as nn
import torch.nn.functional as F


def _in_channels(data):
    if data in ("fmnist", "fedemnist"):
        return 1
    if data == "cifar10":
        return 3
    raise ValueError(f"Unknown dataset: {data}")


def get_model(data, model_name="cnn_residual_se"):
    in_ch = _in_channels(data)
    num_classes = 10

    if model_name == "cnn_residual_se":
        return CuckooResidualSENet(in_ch=in_ch, num_classes=num_classes)

    if model_name == "cnn_vgg_bn":
        return CuckooVGGBNNet(in_ch=in_ch, num_classes=num_classes)

    if model_name == "cnn_inception":
        return CuckooInceptionNet(in_ch=in_ch, num_classes=num_classes)

    if model_name == "cnn_depthwise_se":
        return CuckooDepthwiseSENet(in_ch=in_ch, num_classes=num_classes)

    if model_name == "mlp_mixer":
        if in_ch != 1:
            raise ValueError("mlp_mixer here is designed for 28x28 grayscale FMNIST/FedEMNIST.")
        return CuckooMLPMixerNet(in_ch=in_ch, num_classes=num_classes)

    raise ValueError(f"Unknown model_name: {model_name}")


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1, dropout=0.0):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        w = self.pool(x)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ResidualSEBlock(nn.Module):
    def __init__(self, channels, dropout=0.05):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, dropout=dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + residual)


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.05):
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2, dropout=dropout),
            ResidualSEBlock(out_ch, dropout=dropout),
        )

    def forward(self, x):
        return self.main(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.05):
        super().__init__()
        self.depth = ConvBNAct(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch, dropout=dropout)
        self.point = ConvBNAct(in_ch, out_ch, kernel_size=1, padding=0, dropout=dropout)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.point(self.depth(x)))


class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.05):
        super().__init__()
        branch = out_ch // 4
        self.b1 = ConvBNAct(in_ch, branch, kernel_size=1, padding=0, dropout=dropout)
        self.b3 = nn.Sequential(
            ConvBNAct(in_ch, branch, kernel_size=1, padding=0, dropout=dropout),
            ConvBNAct(branch, branch, kernel_size=3, padding=1, dropout=dropout),
        )
        self.b5 = nn.Sequential(
            ConvBNAct(in_ch, branch, kernel_size=1, padding=0, dropout=dropout),
            ConvBNAct(branch, branch, kernel_size=3, padding=1, dropout=dropout),
            ConvBNAct(branch, branch, kernel_size=3, padding=1, dropout=dropout),
        )
        self.bp = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(in_ch, out_ch - 3 * branch, kernel_size=1, padding=0, dropout=dropout),
        )
        self.se = SEBlock(out_ch)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)
        return self.se(out)


class CuckooResidualSENet(nn.Module):
    """
    Strong CNN: residual blocks + squeeze-excitation.
    Good for testing whether Cuckoo survives a modern CNN with channel attention.
    """
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.stem = ConvBNAct(in_ch, 32, kernel_size=3, stride=1, dropout=0.02)
        self.block1 = nn.Sequential(
            ResidualSEBlock(32, dropout=0.03),
            ResidualSEBlock(32, dropout=0.03),
        )
        self.down1 = DownsampleBlock(32, 64, dropout=0.04)
        self.block2 = nn.Sequential(
            ResidualSEBlock(64, dropout=0.04),
            ResidualSEBlock(64, dropout=0.04),
        )
        self.down2 = DownsampleBlock(64, 128, dropout=0.05)
        self.block3 = nn.Sequential(
            ResidualSEBlock(128, dropout=0.05),
            ResidualSEBlock(128, dropout=0.05),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 192),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.pool(x)
        return self.classifier(x)


class CuckooVGGBNNet(nn.Module):
    """
    VGG-style BatchNorm model.
    High-capacity convolutional baseline with dense classifier.
    """
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_ch, 32, 3, dropout=0.02),
            ConvBNAct(32, 32, 3, dropout=0.02),
            nn.MaxPool2d(2),

            ConvBNAct(32, 64, 3, dropout=0.03),
            ConvBNAct(64, 64, 3, dropout=0.03),
            nn.MaxPool2d(2),

            ConvBNAct(64, 128, 3, dropout=0.04),
            ConvBNAct(128, 128, 3, dropout=0.04),
            ConvBNAct(128, 128, 3, dropout=0.04),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 384),
            nn.GELU(),
            nn.Dropout(0.45),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CuckooInceptionNet(nn.Module):
    """
    Inception-style CNN.
    Tests whether Cuckoo survives multi-scale feature extraction.
    """
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.stem = ConvBNAct(in_ch, 32, kernel_size=3, dropout=0.02)
        self.inc1 = InceptionBlock(32, 64, dropout=0.03)
        self.pool1 = nn.MaxPool2d(2)
        self.inc2 = InceptionBlock(64, 128, dropout=0.04)
        self.pool2 = nn.MaxPool2d(2)
        self.inc3 = InceptionBlock(128, 160, dropout=0.05)
        self.res = nn.Sequential(
            ConvBNAct(160, 160, kernel_size=3, dropout=0.05),
            ResidualSEBlock(160, dropout=0.05),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 192),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.pool1(x)
        x = self.inc2(x)
        x = self.pool2(x)
        x = self.inc3(x)
        x = self.res(x)
        x = self.pool(x)
        return self.classifier(x)


class CuckooDepthwiseSENet(nn.Module):
    """
    MobileNet-like depthwise separable CNN + SE.
    Tests Cuckoo under efficient convolutions and fewer full conv parameters.
    """
    def __init__(self, in_ch=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, 32, kernel_size=3, dropout=0.02),
            DepthwiseSeparableBlock(32, 48, stride=1, dropout=0.03),
            DepthwiseSeparableBlock(48, 64, stride=2, dropout=0.03),
            DepthwiseSeparableBlock(64, 96, stride=1, dropout=0.04),
            DepthwiseSeparableBlock(96, 128, stride=2, dropout=0.04),
            DepthwiseSeparableBlock(128, 160, stride=1, dropout=0.05),
            ResidualSEBlock(160, dropout=0.05),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 192),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.net(x))


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, dim, token_hidden=128, channel_hidden=256, dropout=0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden, num_tokens),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, channel_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: B, tokens, channels
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y
        x = x + self.channel_mlp(self.norm2(x))
        return x


class CuckooMLPMixerNet(nn.Module):
    """
    MLP-Mixer-like model for 28x28 FMNIST.
    This is intentionally non-convolutional after patch embedding.
    It tests whether Cuckoo depends on convolutional spatial bias.
    """
    def __init__(self, in_ch=1, num_classes=10, patch=4, dim=96, depth=4):
        super().__init__()
        self.patch = patch
        self.dim = dim
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        num_tokens = (28 // patch) * (28 // patch)
        self.blocks = nn.Sequential(*[
            MixerBlock(num_tokens=num_tokens, dim=dim, token_hidden=128, channel_hidden=256, dropout=0.15)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.Linear(dim, 192),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.patch_embed(x)          # B, dim, 7, 7
        x = x.flatten(2).transpose(1, 2) # B, 49, dim
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class CNN_CIFAR(nn.Module):
    """
    CIFAR fallback. Not used in the FMNIST smoke test.
    """
    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.stem = ConvBNAct(3, 64, kernel_size=3, dropout=0.02)
        self.block1 = nn.Sequential(ResidualSEBlock(64), ResidualSEBlock(64))
        self.down1 = DownsampleBlock(64, 128)
        self.block2 = nn.Sequential(ResidualSEBlock(128), ResidualSEBlock(128))
        self.down2 = DownsampleBlock(128, 192)
        self.block3 = nn.Sequential(ResidualSEBlock(192), ResidualSEBlock(192))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.pool(x)
        return self.classifier(x)