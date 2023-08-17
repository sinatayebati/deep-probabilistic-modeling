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

def get_ss_indices_per_class(y ,sup_per_class):
    # numer of indices to consider
    n_indx = y.size()[0]

    # calculate the indices per class
    idxs_per_class = {j: [] for j in range(10)}

    # for each index identify the class and add the index to the right class
    for i in range(n_indx):
        curr_y = y[i]
        for j in range(10):
            if curr_y[j] == 1:
                idxs_per_class[j].append(i)
                break
    
    idxs_sup = []
    idxs_unsup = []
    for j in range(10):
        np.random.shuffle(idxs_per_class[j])
        idxs_sup.extend(idxs_per_class[j][:sup_per_class])
        idxs_unsup.extend(idxs_per_class[j][sup_per_class: len(idxs_per_class[j])])

    return idxs_sup, idxs_unsup


def split_sup_unsup_valid(X, y, sup_num, validation_num=10000):
    """
    this is a helper a function for splitting the data into supervised, un-supervised and validation parts
    :param X: images
    :param y: labels (digits)
    :param sup_num: what number of last examples is supervised
    :param validation_num: what number of last examples to use for validation
    :return: splits of data by sup_num number of supervised examples
    """

    # validation set is the last 10,000 examples
    X_valid = X[-validation_num:]
    y_valid = y[-validation_num:]

    X = X[0:-validation_num]
    y = y[0:-validation_num]

    assert sup_num % 10 == 0, "unable to have equal number of images per class"

    # number of supervised examples per class
    sup_per_class = int(sup_num / 10)

    idxs_sup, idxs_unsup = get_ss_indices_per_class(y, sup_per_class)
    X_sup = X[idxs_sup]
    y_sup = y[idxs_sup]
    X_unsup = X[idxs_unsup]
    y_unsup = y[idxs_unsup]

    return X_sup, y_sup, X_unsup, y_unsup, X_valid

def print_distribution_labels(y):
    """
    helper function for printing the distribution of class labels in a dataset
    :param y: tensor of class labels given as one-hots
    :return: a dictionary of counts for each label from y
    """

    counts = {j: 0 for j in range(10)}
    for i in range(y.size()[0]):
        for j in range(10):
            if y[i][j] == 1:
                counts[j] += 1
                break
    print(counts)


class MNISTCached(MNIST):
    """
    a wrapper arounf MNIST to load and cache the transformed data
    once at the beginnig of the inference
    """

    # static class variables for caching training data
    train_data_size = 50000
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    validation_size = 10000
    data_valid, labels_valid = None, None
    test_size = 10000

    def __init__(self, mode, sup_num, use_cuda=True, *args, **kwargs):
            super().__init__(train=mode in ["sup", "unsup", "valid"], *args, **kwargs)

            # transformations on MNIST data (normalization and one-hot conversion for labels)
            def transform(x):
                return fn_x_mnist(x, use_cuda)