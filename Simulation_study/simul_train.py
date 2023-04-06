import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from simul_util import make_reproducibility, sampling, simulation, MYTensorDataset
from simul_loss import log_t_normalizing_const, gamma_regularizer
from simul_model import Encoder, Decoder, gammaAE
from simul_shallow import Shallow_Encoder, Shallow_Decoder, Shallow_gammaAE

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device(f'cuda:0' if USE_CUDA else "cpu")
