"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This module provides a simple neural network for processing the data set,
"MNIST."

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import collections
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relus = nn.Sequential(collections.OrderedDict([
            ('linear0', nn.Linear(784, 784)),
            ('relu0', nn.ReLU()),
            ('linear1', nn.Linear(784, 784)),
            ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(784, 10))]))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relus(x)
        return logits
