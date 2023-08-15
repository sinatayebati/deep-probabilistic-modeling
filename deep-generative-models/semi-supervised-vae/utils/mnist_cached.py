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