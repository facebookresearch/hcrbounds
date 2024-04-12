"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This module provides a simple neural network for processing the data set,
"CIFAR-10."

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import collections
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(collections.OrderedDict([
            ('conv0', nn.Conv2d(3, 32, 3)),
            ('relu0', nn.ReLU()),
            ('maxpool0', nn.MaxPool2d(2)),
            ('conv1', nn.Conv2d(32, 1024, 5)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(3)),
            ('conv2', nn.Conv2d(1024, 3072, 3)),
            ('relu2', nn.ReLU()),
            ('flatten', nn.Flatten()),
            ('linear0', nn.Linear(3072, 3072)),
            ('relu3', nn.ReLU()),
            ('linear1', nn.Linear(3072, 10))]))

    def forward(self, x):
        logits = self.layers(x)
        return logits
