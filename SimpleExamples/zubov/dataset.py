import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Union, Tuple, Optional
import numpy as np
from numpy import ndarray

to_numpy = lambda x : x.detach().cpu().numpy()

class SampleDataset(Dataset):
    def __init__(self, n_samples: int, batch_size: int, lb: Tensor, ub: Tensor, alpha: float, verbose:bool=True):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.lb = lb.reshape(1, -1)
        self.ub = ub.reshape(1, -1)
        self.input_dim = len(lb)
        self.alpha = alpha
        self.box_range = self.ub - self.lb
        # self.randomize_samples()
        if verbose:
            print(f"n_samples = {n_samples}, batch_size = {batch_size}")
            print(f"lb shape = {self.lb.shape}, ub shape = {self.ub.shape}")
            print(f"box_range shape = {self.box_range.shape}")
            print(f"input_dim = {self.input_dim}")

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __getitem__(self, idx: int):
        return self.randomize_samples(self.batch_size)
        # return self.x[idx*self.batch_size:(idx+1)*self.batch_size], self.x_b[idx*self.batch_size:(idx+1)*self.batch_size]

    @staticmethod
    def project_to_box_border(x: Tensor, x_L: Tensor, x_U: Tensor) -> Tensor:
        """

        :param x:       box samples
        :param x_L:     lower limit box to project onto
        :param x_U:     upper limit box to project onto
        :return:
        """

        # For points inside the box, project to the nearest boundary
        x_L = x_L.reshape(1, -1).expand(x.shape[0], -1)
        x_U = x_U.reshape(1, -1).expand(x.shape[0], -1)
        b = torch.arange(x.shape[0])

        # x = torch.clamp(x, min=x_L, max=x_U)
        min_l, min_l_idx = torch.min((x - x_L).abs(), dim=1)
        min_u, min_u_idx = torch.min((x - x_U).abs(), dim=1)

        mask = min_l < min_u
        min_idx = torch.where(mask, min_l_idx, min_u_idx)
        border = torch.where(mask, x_L[b, min_l_idx], x_U[b, min_u_idx])

        x[b, min_idx] = border  # perform projection

        return x

    def randomize_samples(self, num_samples: int):
        x = torch.rand((num_samples, self.input_dim)) * self.box_range + self.lb
        x_b = SampleDataset.project_to_box_border(self.alpha * x, self.lb, self.ub)
        return x, x_b

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    batch_size = 2048
    x_L = -torch.ones(2)
    x_U = torch.ones(2)
    dataset_args = {
        "n_samples": 10_000,
        "batch_size": batch_size,
        "lb": x_L,
        "ub": x_U,
        "alpha": 0.2,
    }
    zubov_dataset = SampleDataset(**dataset_args)
    train_loader = DataLoader(zubov_dataset, batch_size=1, shuffle=True, drop_last=True)
    for batch_x in train_loader:
        print(f"batch_x shape = {batch_x.shape}")