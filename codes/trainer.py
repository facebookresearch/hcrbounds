#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This script trains neural nets on CIFAR10 or MNIST. Invoking this script on the
command-line can accept a single argument, selected from among the choices,
"CIFAR10" and "MNIST." Omitting to provide an argument on the command-line will
select the default, "MNIST." The training sweeps through all examples (randomly
ordered) in the training set 6 or 7 times -- 6 epochs for MNIST, 7 for CIFAR10.
The optimizer is AdamW with a learning rate of 0.001 and with minibatches, each
containing 32 examples, from the training set. The models come from the PyTorch
modules cifar10_model and mnist_model. The script saves the fully trained model
to the current working directory in a file, either "cifar10_model.pth" or
"mnist_model.pth".

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import os

from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

import cifar10_model
import mnist_model


# Require deterministic computations.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
# Set the random seeds.
seed_numpy = 3820497
seed_python = 7892345
seed_torch = 8925934
# Set the seeds for the random number generators.
np.random.seed(seed=seed_numpy)
random.seed(seed_python)
torch.manual_seed(seed_torch)
gen = torch.Generator()
gen.manual_seed(seed_torch)
gent = torch.Generator()
gent.manual_seed(seed_torch)


def worker_init_fn(worker_id):
    """
    Initializes the random number generator for a PyTorch worker node

    Sets the seed for the random number generator on worker worker_id.

    Parameters
    ----------
    worker_id : int
        number of the worker to initialize

    Returns
    -------
    None
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class MNISTdir(MNIST):
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "processed")


# Set which GPU to use.
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Parse the command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    'dataset', default='MNIST', choices=['CIFAR10', 'MNIST'], nargs='?')
args = parser.parse_args()

# Set the dataset to consider.
dataset = args.dataset
print(f'dataset = {dataset}')

# Construct the model.
if dataset == 'CIFAR10':
    model = cifar10_model.SimpleModel()
elif dataset == 'MNIST':
    model = mnist_model.SimpleModel()
model = model.cuda()
print(f'model = {model}')

# Set the optimizer and learning rate.
learning_rate = 0.001
print(f'learning_rate = {learning_rate}')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Construct the data loader.
batch_size = 2**5
print(f'batch_size = {batch_size}')
if dataset == 'CIFAR10':
    infdir = '/datasets01/cifar-pytorch/11222017'
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataclass = CIFAR10
elif dataset == 'MNIST':
    infdir = '/datasets01/mnist-pytorch/11222017'
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    dataclass = MNISTdir
inf_loader = torch.utils.data.DataLoader(
    dataclass(infdir, train=True, download=False, transform=preprocess),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    worker_init_fn=worker_init_fn, generator=gen)
test_loader = torch.utils.data.DataLoader(
    dataclass(infdir, train=False, download=False, transform=preprocess),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    worker_init_fn=worker_init_fn, generator=gent)

# Train the model (printing details of the training, if desired).
details = False
# Set the number of digits of precision to print.
prec = 3
# Set the number of epochs (full sweeps through the training data set).
if dataset == 'CIFAR10':
    epochs = 7
elif dataset == 'MNIST':
    epochs = 6
# Sweep through the training set and check the accuracy on the test set.
for epoch in range(epochs):
    print(f'trained {epoch} of {epochs} epochs...\n')
    # Evaluate the training accuracy and loss and backprop.
    aaccuracy = 0
    aloss = 0
    batches = 0
    if details:
        print('(accuracy, loss) = ', end='')
    for inputs, target in inf_loader:
        batches += 1
        output = model(inputs.cuda()).cpu()
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).sum().item() / pred.shape[0]
        aaccuracy += accuracy
        loss = criterion(output, target)
        aloss += loss.item()
        if details:
            print(
                f'({round(accuracy, prec)}, {round(loss.item(), prec)}), ',
                end='')
        model.zero_grad()
        loss.backward()
        optimizer.step()
    aaccuracy /= batches
    aloss /= batches
    if details:
        print('\b\b  ')
    print(f'average training accuracy = {round(aaccuracy, prec)}')
    print(f'average training loss = {round(aloss, prec)}')
    # Evaluate the testing accuracy and loss.
    ataccuracy = 0
    atloss = 0
    batches = 0
    for inputs, target in test_loader:
        batches += 1
        output = model(inputs.cuda()).cpu()
        pred = output.argmax(dim=1, keepdim=True)
        taccuracy = pred.eq(target.view_as(pred)).sum().item() / pred.shape[0]
        ataccuracy += taccuracy
        tloss = criterion(output, target).item()
        atloss += tloss
    ataccuracy /= batches
    atloss /= batches
    print(f'average test accuracy = {round(ataccuracy, prec)}')
    print(f'average test loss = {round(atloss, prec)}')
print(f'trained {epoch + 1} of {epochs} epochs...')

# Save the model.
torch.save(model, dataset.lower() + '_model.pth')
