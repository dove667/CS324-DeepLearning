from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    
    self.backbone = nn.Sequential(
        self.make_conv_layer(n_channels, 64),
        nn.MaxPool2d(3, 2, 1),
        self.make_conv_layer(64, 128),
        nn.MaxPool2d(3, 2, 1),
        self.make_conv_layer(128, 256),
        self.make_conv_layer(256, 256),
        nn.MaxPool2d(3, 2, 1),
        self.make_conv_layer(256, 512),
        self.make_conv_layer(512, 512),
        nn.MaxPool2d(3, 2, 1),
        self.make_conv_layer(512, 512),
        self.make_conv_layer(512, 512),
        nn.MaxPool2d(3, 2, 1),
    )
    self.fc = nn.Linear(512, n_classes)

  def make_conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    conv_layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv_layer
  
  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    out = self.backbone(x)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out
