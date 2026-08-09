import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    """
    Standard ResNet Bottleneck block implemented from scratch.
    Consists of 1x1, 3x3, and 1x1 convolutions.
    """
    # The expansion factor dictates how much the output channels grow 
    # relative to the internal 3x3 convolution channels.
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # 1x1 Convolution: Dimensionality Reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 Convolution: Spatial Feature Extraction
        # Padding=1 ensures spatial dimensions are maintained unless stride > 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 Convolution: Dimensionality Expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=False)
        
        # The projection shortcut D(x) if dimensions don't match
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # F(x) mapping
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply D(x) to the identity shortcut if necessary
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection: sigma(D(x) + F(x))
        out =  out + identity
        out = self.relu(out)

        return out


class CustomTruncatedResNet(nn.Module):
    """
    Truncated ResNet-50 built from scratch. 
    Halts after layer3 to yield a feature map of depth 1024.
    For a 512x512 input, the output spatial grid is 32x32.
    """
    def __init__(self, block, layers):
        super(CustomTruncatedResNet, self).__init__()
        self.in_channels = 64

        # Initial Stem: Aggressively downsamples the 512x512 input
        # 7x7 Conv with stride 2 -> 256x256
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=False)
        # This is a common practice in ResNet implementations to optimize memory usage during training and inference.
        
        # 3x3 MaxPool with stride 2 -> 128x128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer 1: 3 Bottleneck blocks. Output spatial size remains 128x128. Channels expand to 256.
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # Layer 2: 4 Bottleneck blocks. Stride 2 downsamples spatial size to 64x64. Channels expand to 512.
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # Layer 3: 6 Bottleneck blocks. Stride 2 downsamples spatial size to 32x32. Channels expand to 1024.
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # We STOP here. No layer4, no adaptive pooling, no FC layer.

        # Initialize weights mathematically (Kaiming Normal for ReLU networks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        
        # If we are striding, or if the input channels don't match the expected expanded output channels,
        # we must create a downsample projection layer for the identity shortcut.
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # First block in the layer (handles the downsampling and channel expansion)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks in the layer (maintains spatial dims and channel dims)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x: [Batch, 3, 512, 512]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Output x: [Batch, 1024, 32, 32]
        return x

def create_resnet50_backbone():
    # ResNet-50 uses 3, 4, 6, and 3 blocks per layer respectively.
    # We only pass the first three counts since we truncated the network.
    return CustomTruncatedResNet(Bottleneck, [3, 4, 6])