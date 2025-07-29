import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the 'src' directory relative to the current script
# Assuming 'your_script.py' is in 'some_dir' and 'src' is one level up and into 'src'
# So, from 'some_dir', we go '..' then into 'src'
quant_layer_dir = os.path.join(current_script_dir, '..','..', 'src', 'Pytorch_ext')



quant_layer_dir = os.path.abspath(quant_layer_dir)

sys.path.append(quant_layer_dir)

try:
    # If Wuant_Layer.py defines LoFloatFakeQuant, you would import it like this:
    from Quant_Layer import LoFloatFakeQuant
    print("Successfully imported LoFloatFakeQuant from Wuant_Layer.py!")
except ImportError as e:
    print(f"Error importing Wuant_Layer: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted to add: {quant_layer_dir}")


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x
    
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__()

        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),  # Input: 224x224x3 → Output: 112x112x32

            DepthwiseSeparableConv(32, 64, 1),
            LoFloatFakeQuant(0.5, 0.0),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),
            LoFloatFakeQuant(0.5, 0.0),
            *[DepthwiseSeparableConv(512, 512, 1) for _ in range(5)],

            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),

            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


