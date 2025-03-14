import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Union, Tuple, Optional
import numpy as np
from numpy import ndarray

to_numpy = lambda x : x.detach().cpu().numpy()

class SampleDataset(Dataset):
    def __init__(self, n_samples: int, lb: Tensor, ub: Tensor, alpha: float):
        super().__init__()
        self.n_samples = n_samples
        self.lb = lb
        self.ub = ub
        self.alpha = alpha
        self.box_range = self.ub - self.lb

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tensor:
        x = torch.rand((self.n_samples, len(self.lb)))*self.box_range + self.lb
        return x