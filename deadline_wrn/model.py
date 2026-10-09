"""CIFAR WRN reimplementation following Zagoruyko/Komodakis original Torch model.
Sources and deviations: README.md. No pretrained weights or dataset access.
"""
import math
import torch
from torch import nn

class WideBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride, dropout):
        super().__init__()
        self.projected = incoming != outgoing or stride != 1
        self.bn1 = nn.BatchNorm2d(incoming)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        self.relu2 = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.shortcut = nn.Conv2d(incoming, outgoing, 1, stride, bias=False) if self.projected else nn.Identity()

    def forward(self, x):
        activated = self.relu1(self.bn1(x))
        shortcut = self.shortcut(activated) if self.projected else x
        residual = self.conv2(self.dropout(self.relu2(self.bn2(self.conv1(activated)))))
        return shortcut + residual

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout=0.3, num_classes=10):
        super().__init__()
        if (depth - 4) % 6: raise ValueError('WRN depth must be 6n+4')
        blocks = (depth - 4) // 6
        widths = (16, 16*widen_factor, 32*widen_factor, 64*widen_factor)
        self.stem = nn.Conv2d(3,16,3,1,1,bias=False)
        groups=[]
        for group in range(3):
            layers=[WideBlock(widths[group],widths[group+1],1 if group==0 else 2,dropout)]
            layers += [WideBlock(widths[group+1],widths[group+1],1,dropout) for _ in range(blocks-1)]
            groups.append(nn.Sequential(*layers))
        self.groups=nn.ModuleList(groups)
        self.bn=nn.BatchNorm2d(widths[-1])
        self.relu=nn.ReLU(inplace=False)
        self.pool=nn.AvgPool2d(8)
        self.classifier=nn.Linear(widths[-1],num_classes)
        self.initialize()

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_in',nonlinearity='relu')
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.ones_(layer.weight);nn.init.zeros_(layer.bias)
                layer.reset_running_stats()
            elif isinstance(layer,nn.Linear):
                nn.init.uniform_(layer.weight,-1/math.sqrt(layer.in_features),1/math.sqrt(layer.in_features))
                nn.init.zeros_(layer.bias)

    def forward(self,x):
        x=self.stem(x)
        for group in self.groups:x=group(x)
        return self.classifier(self.pool(self.relu(self.bn(x))).flatten(1))

def expected_parameters(depth=28,widen_factor=10):
    """Independent arithmetic: conv weights, two BN affine vectors, projections, FC."""
    n=(depth-4)//6; widths=(16,16*widen_factor,32*widen_factor,64*widen_factor)
    count=3*16*9
    for group in range(3):
        incoming=widths[group];outgoing=widths[group+1]
        for block in range(n):
            ci=incoming if block==0 else outgoing
            count+=2*ci+ci*outgoing*9+2*outgoing+outgoing*outgoing*9
            if ci!=outgoing:count+=ci*outgoing
    return count+2*widths[-1]+widths[-1]*10+10
