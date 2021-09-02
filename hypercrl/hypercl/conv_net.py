import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mnet_interface import MainNetInterface
from .utils.batchnorm_layer import BatchNormLayer
from .utils.context_mod_layer import ContextModLayer
from .utils.torch_utils import init_params


class ConvNet(nn.Module, MainNetInterface):
    def __init__(self, dim_in, dim_out):
        super().__init__()
