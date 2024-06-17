#!/usr/bin/env python3

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This script computes the Hammersley-Chapman-Robbins (HCR) bounds on the inputs
for inference with a ResNet18 or Swin_T pretrained on ImageNet or with small
neural nets pretrained via trainer.py on CIFAR10 or MNIST. All bounds are based
on dithering of the associated features. The HCR calculations in this script
bound the variance of the pixels in the discrete cosine transforms of the
images input from the validation set of ImageNet for the ResNet18 or Swin_T or
from the test sets of CIFAR10 or MNIST. The script saves files in the
subdirectory, "bounds_[dataset]_nolimit" or "bounds_[dataset]_limit", of the
current working directory. The former name of the subdirectory pertains to
running the script without the command-line flag, "--limit" (while the latter
pertains to running with the command-line flag, "--limit"). For each input, the
script saves four files as JPEGs, with names starting "clipped", "unclipped",
"perturbed", and "unperturbed". The clipped versions set pixels in the bounds
whose values exceed 16 gray levels to 15 (there are 256 gray levels in total,
ranging from 0 to 255). The unclipped versions display the original discrete
cosine transforms. The perturbed versions add to the original images the bounds
(without clipping) with uniformly random signs. The images input from ImageNet
get resampled to be 91 pixels wide, 91 pixels high, and have 3 color channels.
The images get upsampled by a factor of 32/13 to become 224 pixels wide and
224 pixels high upon input to the pretrained ResNet18 or Swin_T. Invoking this
script on the command-line requires providing a single argument, selected from
among the choices, "CIFAR10," "MNIST," "ResNet18," and "Swin_T". The optional
command-line flag ("--limit") alters the perturbation to the input used when
computing the HCR bounds; the resulting perturbation is very small when
invoking the script with the flag ("--limit"). This small perturbation
corresponds to the limit in which the HCR bounds become the Cramer-Rao bounds
involving the Fisher information (however, the computations performed here
never calculate Fisher information explicitly; instead they employ the usual
finite-difference approximation of the HCR bounds).

This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
"""

import argparse
import collections
import math
import numpy as np
import matplotlib
import random
import scipy
import os
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
from functorch.experimental import replace_all_batch_norm_modules_
from PIL import Image


import cifar10_model
import mnist_model


matplotlib.use('agg')
import matplotlib.pyplot as plt


# Require deterministic computations (except for the upsampling operation
# in the ResNet18 and Swin_T models for processing ImageNet data).
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
gent = torch.Generator()
gent.manual_seed(seed_torch)


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
    'name', choices=['CIFAR10', 'MNIST', 'ResNet18', 'Swin_T'])
parser.add_argument('--limit', action='store_true')
args = parser.parse_args()

# Set the name to consider.
name = args.name

# Determine whether to take the limit such that the Hammersley-Chapman-Robbins
# bounds become the Cramer-Rao bound (set limit = True to take that limit).
limit = args.limit

# Set the configuration parameters for the model of interest.
if name == 'CIFAR10':

    sigma_scale = 5
    diffdiv = 500
    batch_size = 2500
    num_batches = 1
    num_pits = 6

    def partition(full, ilen):
        # Note that ilen is superfluous for this particular instantiation.
        model = torch.nn.Sequential(collections.OrderedDict([
            *list(list(full.named_children())[0][1].named_children())[:-1]]))
        final = torch.nn.Sequential(collections.OrderedDict([
            list(list(full.named_children())[0][1].named_children())[-1]]))
        return model, final

elif name == 'MNIST':

    sigma_scale = 1
    diffdiv = 200
    batch_size = 5000
    num_batches = 2
    num_pits = 10

    def partition(full, ilen):
        # Note that ilen is superfluous for this particular instantiation.
        model = torch.nn.Sequential(collections.OrderedDict([
            *(list(full.named_children())[:-1]),
            *(list(list(
                full.named_children())[-1][1].named_children())[:-1])]))
        final = torch.nn.Sequential(collections.OrderedDict([
            *(list(list(
                full.named_children())[-1][1].named_children())[-1:])]))
        return model, final

elif name == 'ResNet18':

    # Allow the upsampling not to be strictly deterministic.
    torch.use_deterministic_algorithms(False)
    sigma_scale = 2
    diffdiv = 500
    batch_size = 16
    num_batches = 8
    num_pits = 10

    def partition(full, ilen):
        model = torch.nn.Sequential(collections.OrderedDict([
            ('interp', torch.nn.Upsample(size=[ilen, ilen], mode='bilinear')),
            *(list(full.named_children())[:-2])]))
        final = torch.nn.Sequential(collections.OrderedDict([
            *(list(full.named_children())[-2:-1]),
            ('flatten', torch.nn.Flatten(1)),
            list(full.named_children())[-1]]))
        return model, final

elif name == 'Swin_T':

    # Allow the upsampling not to be strictly deterministic.
    torch.use_deterministic_algorithms(False)
    sigma_scale = 3
    diffdiv = 500
    batch_size = 16
    num_batches = 8
    num_pits = 10

    def partition(full, ilen):
        model = torch.nn.Sequential(collections.OrderedDict([
            ('interp', torch.nn.Upsample(size=[ilen, ilen], mode='bilinear')),
            *(list(full.named_children())[:-5])]))
        final = torch.nn.Sequential(collections.OrderedDict([
            *(list(full.named_children())[-5:])]))
        return model, final

if limit:
    diffdiv = 1000
    if name == 'CIFAR10':
        num_pits = 4
    elif name == 'MNIST':
        num_pits = 10

print(f'name = {name}')
print(f'limit = {limit}')
print(f'sigma_scale = {sigma_scale}')
print(f'diffdiv = {diffdiv}')
print(f'batch_size = {batch_size}')


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


def scores_and_results(output, target):
    """
    Computes the scores and corresponding correctness

    Given output from a classifier, computes the score from the output vector
    and checks whether the most likely class matches target. This function
    assumes that the first dimensions of the inputs are minibatch dimensions.

    Parameters
    ----------
    output : array_like
        confidences (often in the form of a probability distribution
        over the classes) in classification of each example into every class
    target : array_like
        index of the correct class for each example

    Returns
    -------
    array_like
        scores
    array_like
        Boolean indicators of correctness of the classifications
    """
    # Select the scores.
    scores = torch.max(output, dim=1).values
    scores = scores.cpu().detach().numpy()
    # Calculate the results.
    argmaxs = torch.argmax(output, dim=1)
    results = argmaxs.eq(target)
    results = results.cpu().detach().numpy()

    return scores, results


def infer(inf_loader, model, final, num_batches):
    """
    Conducts inference given a model and data loader

    Runs model and the separate final classifier on data loaded via inf_loader.

    Parameters
    ----------
    inf_loader : class
        instance of torch.utils.data.DataLoader
    model : class
        torch model without the final classification layers
    final : class
        torch model for the final classification layers
    num_batches : int
        number of batches to process

    Returns
    -------
    array_like
        scores
    array_like
        Boolean indicators of correctness of the classifications
    list
        ndarrays of indices of the examples classified into each class
        (the i'th entry of the list is an array of the indices of the examples
        from the data set that got classified into the i'th class);
        each ndarray in the list corresponds to a minibatch
    list
        ndarrays of outputs from the model without the final classifier;
        each ndarray in the list corresponds to a minibatch
    """
    # Prepare the models for inference.
    model.eval()
    final.eval()
    # Track the offset for appending indices to indicators (by default,
    # each minibatch gets indexed starting from 0, rather than offset).
    offset = 0
    nclasses = list(final.named_children())[-1][1].out_features
    print(f'nclasses = {nclasses}')
    indicators = [None] * nclasses
    outputs = []
    for k, (input, target) in enumerate(inf_loader):
        print(f'{k} of {num_batches} batches processed.')
        # Run inference.
        target = target.cuda(non_blocking=True)
        input = input.cuda()
        print(f'input.shape = {input.shape}')
        output = model(input)
        print(f'output.shape = {output.shape}')
        outputsmax = final(output)
        # Store the scores and results from the current minibatch,
        # and record which entries have the desired target indices.
        s, r = scores_and_results(outputsmax, target)
        # Record the scores, results, and outputs.
        if k == 0:
            scores = s.copy()
            results = r.copy()
        else:
            scores = np.concatenate((scores, s))
            results = np.concatenate((results, r))
        outputs.append(output.cpu().detach().numpy())
        # Partition the results into the nclasses classes.
        for i in range(nclasses):
            inds = torch.nonzero(
                target == i, as_tuple=False).cpu().detach().numpy()
            if k == 0:
                indicators[i] = inds.copy()
            else:
                indicators[i] = np.concatenate((indicators[i], inds + offset))
        # Increment offset.
        offset += target.numel()
        if k == num_batches - 1:
            break
    print(f'{k + 1} of {num_batches} batches processed.')
    for i in range(nclasses):
        indicators[i] = np.squeeze(indicators[i], axis=1)
    print('m = *scores.shape = {}'.format(*scores.shape))
    return scores, results, indicators, outputs


def iterate(inf_loader, model, final, num_batches, outputsa, num_pits):
    """
    Optimizes HCR bounds given a model, data loader, and random outputs

    Iteratively minimizes the directional derivative (or, rather, the analogous
    finite differences) of the outputs of model with respect to the inputs,
    so that the Euclidean norm of the directional derivative is roughly equal
    to the spectral norm of the pseudoinverse of the Jacobian of the outputs
    relative to the inputs. The optimization happens via several iterations of
    the power method (or, rather, inverse iteration), started in the direction
    given by the difference of outputsa from the initial unperturbed output.
    (The returns also include the results of running inference on outputsa with
    the given final classifier.) Aside from outputsa being the starting point
    for the iterations, outputsa and the calculated perturbed output need not
    have any special relation, as Hammersley-Chapman-Robbins bounds do not
    require any relation.

    Parameters
    ----------
    inf_loader : class
        instance of torch.utils.data.DataLoader
    model : class
        torch model without the final classification layers
    final : class
        torch model for the final classification layers
    num_batches : int
        number of batches to process
    outputsa : array_like
        outputs passed to the final classifier to generate scores and responses
    num_pits : integer
        number of power iterations to conduct

    Returns
    -------
    list
        ndarrays of inputs whose corresponding outputs minimize the directional
        derivative
    list
        ndarrays of the exact outputs yielded by the perturbed inputs
    array_like
        constructed perturbations to the inputs
    array_like
        resulting perturbations to the outputs
    array_like
        scores from running the final classifier on outputsa
    array_like
        Boolean indicators of correctness of the classifications for outputsa
    list
        ndarrays of indices of the examples classified into each class
        (the i'th entry of the list is an array of the indices of the examples
        from the data set that got classified into the i'th class)
    """
    # Prepare the models for inference.
    model.eval()
    final.eval()
    # Track the offset for appending indices to indicators (by default,
    # each minibatch gets indexed starting from 0, rather than offset).
    offset = 0
    nclasses = list(final.named_children())[-1][1].out_features
    print(f'nclasses = {nclasses}')
    indicators = [None] * nclasses
    in_pert = []
    out_pert = []
    for k, (input, target) in enumerate(inf_loader):
        print(f'{k} of {num_batches} batches processed.')
        # Run inference.
        target = target.cuda(non_blocking=True)
        input = input.cuda()
        output = model(input)
        outsize = (output.size(0), output.numel() // output.size(0))
        print(f'input.size() = {input.size()}')
        print(f'output.size() = {output.size()}')
        print(f'outsize = {outsize}')
        # Calculate the starting perturbation to the output.
        diff = outputsa[k] - output.cpu().detach().numpy()
        diff /= math.sqrt(outsize[1])
        diff /= diffdiv
        # Solve for the perturbation to the input which yields an output,
        # defined as outputpert in the iterations, such that the Euclidean norm
        # of outputpert - output is close to that of cdiff (where cdiff starts
        # as diff and evolves during the iterations). Start the iterations with
        # suitable values for perturbed and cdiff, with perturbed starting as
        # the unperturbed input and becoming the perturbed input during
        # iterations of the coming for loop and where cdiff is the perturbation
        # to the output that the iterations calculate (cdiff is the difference
        # between outputpert and output after an iteration, to the extent that
        # the LSQR approximations have converged).
        perturbed = input.clone()
        cdiff = diff.reshape(outsize).copy()
        for _ in range(num_pits):
            # Construct the new target cdiff, normalizing it and then matching
            # the Euclidean norms of the examples in the minibatch from diff.
            cdiff /= np.linalg.norm(cdiff, axis=1)[:, None]
            cdiff *= np.linalg.norm(diff.reshape(outsize), axis=1)[:, None]
            atol = 2e-2 * np.min(np.linalg.norm(cdiff, axis=1))
            cdiff = cdiff.flatten()
            # Solve for the perturbation delta to the input which will yield
            # the desired difference cdiff at the output.
            (_, vjpfunc) = torch.func.vjp(model, perturbed)

            def rmatvec(v):
                return vjpfunc(torch.from_numpy(v.reshape(list(
                    output.shape))).cuda())[0].cpu().detach().numpy().flatten()

            def matvec(v):
                return torch.func.jvp(
                    model, (perturbed,), (torch.from_numpy(v.reshape(list(
                        perturbed.shape))).cuda(),))[1].cpu().detach().numpy(
                            ).flatten()

            linearop = scipy.sparse.linalg.LinearOperator(
                shape=(output.numel(), perturbed.numel()), matvec=matvec,
                rmatvec=rmatvec, matmat=matvec, dtype=np.float32,
                rmatmat=rmatvec)
            (delta, _, itn, r1norm) = scipy.sparse.linalg.lsqr(
                linearop, cdiff, atol=atol, damp=0)[:4]
            print(f'itn = {itn}')
            print(f'r1norm / norm(cdiff) = {r1norm / np.linalg.norm(cdiff)}')
            # Calculate the perturbed inputs.
            perturbed = input + torch.from_numpy(
                delta.reshape(list(input.shape))).float().cuda()
            # Run inference with the perturbed inputs.
            outputpert = model(perturbed)
            # Update the corresponding perturbations of the outputs.
            cdiff = outputpert - output
            cdiff = cdiff.cpu().detach().numpy().reshape(outsize)
        # Check the size of the difference.
        print('norm of output = {}'.format(np.linalg.norm(
            output.cpu().detach().numpy())))
        print('norm of output-outputpert = {}'.format(np.linalg.norm(
            (output - outputpert).cpu().detach().numpy())))
        print('norm of diff = {}'.format(np.linalg.norm(diff)))
        print(
            '[norm(diff)-norm(output-outputpert)] / [norm(diff)] = {}'.format(
                1 - np.linalg.norm(
                    (output - outputpert).cpu().detach().numpy())
                / np.linalg.norm(diff)))
        # Store the perturbed inputs and outputs from the current minibatch.
        in_pert.append(perturbed.cpu().detach().numpy())
        out_pert.append(outputpert.cpu().detach().numpy())
        # Store the perturbations from the current minibatch.
        indiff = (perturbed - input).cpu().detach().numpy()
        outdiff = (outputpert - output).cpu().detach().numpy()
        if k == 0:
            in_diff = indiff.copy()
            out_diff = outdiff.copy()
        else:
            in_diff = np.vstack((in_diff, indiff))
            out_diff = np.vstack((out_diff, outdiff))
        # Store the scores and results from the current (perturbed) minibatch,
        # and record which entries have the desired target indices.
        outputperts = final(torch.from_numpy(outputsa[k]).float().cuda())
        s, r = scores_and_results(outputperts, target)
        # Record the scores and results.
        if k == 0:
            scores = s.copy()
            results = r.copy()
        else:
            scores = np.concatenate((scores, s))
            results = np.concatenate((results, r))
        # Partition the results into the nclasses classes.
        for i in range(nclasses):
            inds = torch.nonzero(
                target == i, as_tuple=False).cpu().detach().numpy()
            if k == 0:
                indicators[i] = inds.copy()
            else:
                indicators[i] = np.concatenate((indicators[i], inds + offset))
        # Increment offset.
        offset += target.numel()
        if k == num_batches - 1:
            break
    print(f'{k + 1} of {num_batches} batches processed.')
    for i in range(nclasses):
        indicators[i] = np.squeeze(indicators[i], axis=1)
    return in_pert, out_pert, in_diff, out_diff, scores, results, indicators


# Conduct inference.
# Set the seeds for the random number generators.
torch.manual_seed(seed_torch)
np.random.seed(seed=seed_numpy)
gent.manual_seed(seed_torch)
# Load the pretrained model.
if name == 'CIFAR10':
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ilen = None
    full = torch.load('cifar10_model.pth')
elif name == 'MNIST':
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    ilen = None
    full = torch.load('mnist_model.pth')
elif name in ['ResNet18', 'Swin_T']:
    preprocess = getattr(models, name + '_Weights').DEFAULT.transforms()
    ilen = preprocess.crop_size[0]
    preprocess.crop_size = [13 * ilen // 32]
    preprocess.resize_size = [13 * preprocess.resize_size[0] // 32]
    full = getattr(models, name.lower())(
        weights=getattr(models, name + '_Weights').DEFAULT)
model, final = partition(full, ilen)
del full
replace_all_batch_norm_modules_(model)
model = model.cuda()
replace_all_batch_norm_modules_(final)
final = final.cuda()
print(f'model = {model}')
print(f'final = {final}')
print('finished constructing the model...')
# Construct the data loader.
if name == 'CIFAR10':
    infdir = '/datasets01/cifar-pytorch/11222017'
    dataclass = CIFAR10
    kwargs = {'train': False, 'download': False, 'transform': preprocess}
elif name == 'MNIST':
    infdir = '/datasets01/mnist-pytorch/11222017'
    dataclass = MNISTdir
    kwargs = {'train': False, 'download': False, 'transform': preprocess}
elif name in ['ResNet18', 'Swin_T']:
    infdir = '/datasets01/imagenet_full_size/061417/val'
    dataclass = ImageFolder
    kwargs = {'transform': preprocess}
inf_loader = torch.utils.data.DataLoader(
    dataclass(infdir, **kwargs), batch_size=batch_size, shuffle=True,
    num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,
    generator=gent)
# Generate the scores, results, subset indicators, and probs.
print('generating scores and results...')
torch.manual_seed(seed_torch)
gent.manual_seed(seed_torch)
s, r, inds, outputs = infer(inf_loader, model, final, num_batches)
print(f'r = \n{r}')
# Check that another run generates the same results.
torch.manual_seed(seed_torch)
gent.manual_seed(seed_torch)
_, r2, _, _ = infer(inf_loader, model, final, num_batches)
if not np.array_equal(r, r2):
    print(f'r2 = \n{r2}')
    raise ValueError('Running infer twice generated different results!')

# Create the directory "bounds..." for output, if necessary.
dir = 'bounds_' + name.lower()
if limit:
    dir += '_limit'
else:
    dir += '_nolimit'
try:
    os.mkdir(dir)
except FileExistsError:
    pass
dir = dir + '/'

# Add Gaussian noise to the outputs.
variance = 0
numels = 0
for outs in outputs:
    variance += np.linalg.norm(outs)**2
    numels += outs.size
variance /= numels
sigma = sigma_scale * math.sqrt(variance)
# Compute the Hammersley-Chapman-Robbins (HCR) bounds, maximizing over num_iter
# independent runs.
hcr = []
accold = []
accnew = []
accoldavg = 0
accnewavg = 0
num_iter = 25
print(f'num_iter = {num_iter}')
for iter in range(num_iter):
    print(f'\n\niter = {iter}')
    outputsa = []
    for outs in outputs:
        outputsa.append(outs + np.random.normal(scale=sigma, size=outs.shape))
    # Backprop from the perturbed outputs to the perturbations of the inputs.
    torch.manual_seed(seed_torch)
    gent.manual_seed(seed_torch)
    _, _, in_diff, out_diff, _, rpert, _ = iterate(
        inf_loader, model, final, num_batches, outputsa, num_pits)
    print(f'rpert = {rpert}')
    # Compare accuracies before and after perturbing.
    accold.append(np.sum(r[:rpert.size]) / rpert.size)
    accnew.append(np.sum(rpert) / rpert.size)
    accoldavg += accold[-1]
    accnewavg += accnew[-1]
    print(f'accold[-1] = {accold[-1]}')
    print(f'accnew[-1] = {accnew[-1]}')
    # Calculate the Hammersely-Chapman-Robbins bounds.
    numerator = scipy.fft.dctn(in_diff, axes=(2, 3), norm='ortho')
    numerator = np.square(numerator)
    print(f'numerator = \n{numerator}')
    print(f'numerator.shape = {numerator.shape}')
    # Be sure to use the same sigma as used to generate outputsa.
    denominator = np.exp(np.square(np.linalg.norm(out_diff.reshape(
        (out_diff.shape[0], out_diff.size // out_diff.shape[0])), axis=1))
        / sigma**2) - 1
    print(f'denominator = \n{denominator}')
    print(f'denominator.shape = {denominator.shape}')
    hcr.append(numerator / denominator[:, None, None, None])
    hcr[-1] = np.sqrt(hcr[-1])
    print(f'hcr[-1] = \n{hcr[-1]}')
    print(f'hcr[-1].shape = {hcr[-1].shape}')
    out_diff_view = out_diff.reshape(
        (out_diff.shape[0], out_diff.size // out_diff.shape[0]))
    print('np.linalg.norm(out_diff_view, axis=1) / sigma = \n{}'.format(
        np.linalg.norm(out_diff_view, axis=1) / sigma))
    print('np.linalg.norm(out_diff_view, axis=1) = \n{}'.format(
        np.linalg.norm(out_diff_view, axis=1)))
    print(f'sigma = {sigma}')
# Print the accuracies.
accoldavg /= num_iter
accnewavg /= num_iter
print(f'accoldavg = {accoldavg}')
print(f'accnewavg = {accnewavg}')
print(f'accold = {accold}')
print(f'accnew = {accnew}')
# For every image in the minibatch, identify the best bound attained over all
# perturbations indexed by iter in the for loop above; hcr is a list of these
# bounds, each specified for every entry in the image's DCT.
hcrmax = np.zeros(hcr[0].shape)
for numex in range(len(hcr)):
    hcrmax = np.maximum(hcrmax, hcr[numex])
print(f'hcrmax = \n{hcrmax}')
print(f'hcrmax.shape = {hcrmax.shape}')
if name in ['CIFAR10', 'MNIST']:
    # Read the input images, modify them via the bounds, and save both to disk.
    pert = np.zeros(hcrmax.shape)
    for k, (input, _) in enumerate(inf_loader):
        print(f'{k} of {num_batches} batches processed.')
        input = input.detach().numpy()
        for im in range(input.shape[0]):
            pert[im + k * input.shape[0], :, :, :] = input[im, :, :, :]
        if k == num_batches - 1:
            break
    # Save the original, unperturbed images and the modified, perturbed ones.
    # (iter = 0 for the unperturbed ones and iter = 1 for the perturbed ones.)
    for iter in range(2):
        for im in range(pert.shape[0]):
            # Save the image, reversing the normalization.
            if name == 'CIFAR10':
                unnorm = (pert[im, :, :, :] + 1) / 2
                unnorm = np.transpose(unnorm, (1, 2, 0))
            elif name == 'MNIST':
                unnorm = 0.3081 * pert[im, 0, :, :] + 0.1307
            unnorm = np.clip(255 * unnorm, 0, 255).astype(np.uint8)
            img = Image.fromarray(unnorm)
            filename = dir + '/'
            if iter == 0:
                filename = filename + 'unperturbed'
            else:
                filename = filename + 'perturbed'
            filename = filename + str(im) + '.jpg'
            img.save(filename)
        if iter == 0:
            # Perturb according to the HCR bounds.
            randbits = np.random.choice([-1, 1], size=hcrmax.shape)
            pert = scipy.fft.dctn(pert, axes=(2, 3), norm='ortho')
            pert = pert + randbits * hcrmax
            pert = scipy.fft.idctn(pert, axes=(2, 3), norm='ortho')
# Save summaries, first unfiltered, then filtering out low frequencies.
for iter in range(2):
    if iter == 1:
        # Discard high frequencies.
        if name in ['CIFAR10', 'MNIST']:
            lowfreq = 8
        elif name in ['ResNet18', 'Swin_T']:
            lowfreq = 32
        hcrmax = hcrmax[:, :, :lowfreq, :lowfreq]
    # Save the best bounds as images.
    for im in range(hcrmax.shape[0]):
        # Save the unclipped image, averaging over the (RGB) color channels.
        averaged = scipy.fft.idctn(
            hcrmax[im, :, :, :], axes=(1, 2), norm='ortho')
        averaged = 255 * np.sum(averaged, axis=0) / averaged.shape[0]
        img = Image.fromarray(averaged.astype(np.uint8))
        filename = dir + '/' + 'unclipped'
        if iter == 1:
            filename += '_filtered'
        filename += str(im) + '.png'
        img.save(filename)
        # Save a clipped image.
        averaged2 = np.minimum(averaged, 15)
        img2 = Image.fromarray(averaged2.astype(np.uint8))
        filename = dir + '/' + 'clipped'
        if iter == 1:
            filename += '_filtered'
        filename += str(im) + '.jpg'
        img2.save(filename)
    # Histogram the best bounds.
    hist = np.histogram(hcrmax, bins=256)
    hist = [hist[0], hist[1], hist[0] / np.sum(hist[0])]
    for ind in range(len(hist)):
        if iter == 1:
            print(f'filtered hist[{ind}] = \n{hist[ind]}')
        else:
            print(f'unfiltered hist[{ind}] = \n{hist[ind]}')
    # Plot histograms of the best bounds.
    plt.figure(figsize=(3, 3))
    title = name + ' '
    if iter == 0:
        title += 'unfiltered'
    else:
        title += 'filtered'
    plt.title(title)
    plt.hist(hcrmax.flatten(), bins=hist[1], color='k')
    plt.xlabel('bound on the standard deviation')
    plt.ylabel('number of modes')
    if name == 'CIFAR10':
        if iter == 0:
            plt.xlim((0, 0.25))
        else:
            plt.xlim((0, 0.375))
    elif name == 'MNIST':
        if iter == 0:
            plt.xlim((0, 1))
        else:
            plt.xlim((0, 1.5))
    elif name == 'ResNet18':
        plt.xlim((0, 0.02))
    elif name == 'Swin_T':
        plt.xlim((0, 0.01))
    filename = dir + '/' + title.replace(' ', '_') + '.jpg'
    plt.savefig(filename, bbox_inches='tight')
