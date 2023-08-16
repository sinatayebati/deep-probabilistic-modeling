import errno
import os
from functools import reduce

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyro.contrib.examples.util import MNIST, get_data_directory

# this file contains utilities for caching, transforming and splitting MNIST data
# efficiently. By default, a PyTorch DataLoader will apply the transform every epoch
# we avoid this by caching the data early on in MNISTCached class

# transfromations for MNIST data
def fn_x_mnist(x, use_cuda):
    # normalize pixel values of the image to be in [0,1] instead of [0,255]
    xp = x * (1.0 / 255)

    # transform x to a linear tensor from bx * a1 * a2 * ... --> bs * A
    xp_1d_size = reduce(lambda a, b: a * b, xp.size()[1:])
    xp = xp.view(-1, xp_1d_size)

    # send the data to GPU
    if use_cuda:
        xp = xp.cuda()

    return xp

def fn_y_mnist(y, use_cuda):
    yp = torch.zeros(y.size(0), 10)

    # send the data to GPU
    if use_cuda:
        yp = yp.cuda()
        y = y.cuda()

    # transform the lavel y (integer between 0 and 9) to a one-hot
    yp = yp.scatter_(1, y.view(-1, 1), 1.0)
    return yp