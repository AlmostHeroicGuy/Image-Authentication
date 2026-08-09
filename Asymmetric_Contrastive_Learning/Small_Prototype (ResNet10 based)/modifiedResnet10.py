import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock: 2x (3x3 conv -> BN -> ReLU), with residual add."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut needed whenever spatial size or channel count changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity
        out = self.relu(out)
        return out


class ResNet10(nn.Module):
    """
    ResNet10 (BasicBlock x [1,1,1,1]) adapted for 64x64 inputs.

    Spatial trace for 64x64 input:
        stem    : 64x64 (3x3, stride 1, no maxpool)
        layer1  : 64x64  (64 ch,  stride 1)
        layer2  : 32x32  (128 ch, stride 2)
        layer3  : 16x16  (256 ch, stride 2)
        layer4  : 8x8    (512 ch, stride 2)
        avgpool : 1x1 (512)
        embedding_head      : 512 -> embedding_dim
    """

    def __init__(self, embedding_dim=128, zero_init_residual=True):
        super().__init__()

        # Stem: 3x3 stride 1, preserves spatial resolution (no maxpool)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = BasicBlock(64, 64, stride=1)    # no downsampling
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.layer4 = BasicBlock(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_head = nn.Linear(512, embedding_dim)

        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            # Zero-init the gamma of the last BN in each block so the residual
            # branch starts as identity -> stabilizes early training (He et al.)
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.embedding_head(x)
        return x


if __name__ == "__main__":
    model = ResNet10(embedding_dim=128)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    print("Output shape:", out.shape)  # expected: torch.Size([4, 128])

    # Sanity check on intermediate spatial sizes
    feat = model.stem(x)
    print("After stem :", feat.shape)
    feat = model.layer1(feat)
    print("After layer1:", feat.shape)
    feat = model.layer2(feat)
    print("After layer2:", feat.shape)
    feat = model.layer3(feat)
    print("After layer3:", feat.shape)
    feat = model.layer4(feat)
    print("After layer4:", feat.shape)
    feat = model.avgpool(feat)
    print("After avgpool:", feat.shape)
    feat = model.embedding_head(torch.flatten(feat, 1))
    print("After embedding head:", feat.shape)