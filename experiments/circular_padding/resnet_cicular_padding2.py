'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training import Boilerplate


class CylindricalConv(nn.Module):
    def __init__(self, channels_in, channels_out,kernel_size, stride=1, **kwargs):
        super().__init__()
        padding = kernel_size // 2
        self.zero_pad = nn.ZeroPad2d((padding, padding, 0, 0))
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, padding=(padding, 0),
                              padding_mode='circular', stride=(1,stride), **kwargs)

    def forward(self, x):
        return self.conv(self.zero_pad(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = CylindricalConv(in_planes, planes, kernel_size=3, stride=stride,
                                     bias=False)  # stride is always 1 in freq. axis
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CylindricalConv(planes, planes, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                CylindricalConv(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),  # stride is always 1 in freq. axis
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = CylindricalConv(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CylindricalConv(planes, planes, kernel_size=3,
                                     stride=stride,
                                     bias=False)  # stride is always 1 in freq. axis
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = CylindricalConv(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                CylindricalConv(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),  # stride is always 1 in freq. axis
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(Boilerplate):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = CylindricalConv(1, 64, kernel_size=3,  # only one input channel
                                     stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)  # adapts to any output size
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out,1)


def ResNet18Circular(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def ResNet34Circular(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def ResNet50Circular(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def ResNet101Circular(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def ResNet152Circular(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = ResNet18Circular()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
