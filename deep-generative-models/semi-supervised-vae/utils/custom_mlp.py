
from inspect import isclass
import torch
import torch.nn as nn

from pyro.distributions.util import broadcast_shape

class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """
    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)
    
class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """
    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        # we have single object
        if len(input_args) == 1:
            # regadrless of type,
            # we don't care about single objects
            # we just index into the object
