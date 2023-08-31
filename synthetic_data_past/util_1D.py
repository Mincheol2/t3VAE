import os
import torch
import random
import numpy as np

class TensorDataset(torch.utils.data.Dataset) :
    def __init__(self, *tensors) -> None:
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)

def make_result_dir(dirname):
    os.makedirs(dirname, exist_ok=True)
    os.makedirs(dirname + '/VAE', exist_ok=True)
    os.makedirs(dirname + '/generations', exist_ok=True)

def make_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True