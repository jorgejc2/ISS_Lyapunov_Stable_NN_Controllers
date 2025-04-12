import torch
from torch.utils.data import Dataset
from torch import Tensor
from typing import Union, Tuple, Optional, Callable
import numpy as np
from numpy import ndarray
from math import ceil

to_numpy = lambda x : x.detach().cpu().numpy()

def get_dataset(advance_trajectories: bool, **args) -> Union['SampleDataset', 'SampleDatasetAdvancedTrajectories']:

    if advance_trajectories:
        return SampleDatasetAdvancedTrajectories(**args)
    else:
        return SampleDataset(**args)

class SampleDataset(Dataset):
    """
    This dataset generates a random set of samples of the fly. This class does not support generating
    trajectories in advance. This class should be used if the controller is being trained as well.
    """
    def __init__(self, n_samples: int, batch_size: int, lb: Tensor, ub: Tensor, verbose:bool=True):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.lb = lb.reshape(1, -1)
        self.ub = ub.reshape(1, -1)
        self.input_dim = len(lb)
        self.box_range = self.ub - self.lb
        if verbose:
            print(f"n_samples = {n_samples}, batch_size = {batch_size}")
            print(f"lb shape = {self.lb.shape}, ub shape = {self.ub.shape}")
            print(f"box_range shape = {self.box_range.shape}")
            print(f"input_dim = {self.input_dim}")

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __getitem__(self, idx: int):
        return self.randomize_samples(self.batch_size), None

    def reset_samples(self):
        pass

    def randomize_samples(self, num_samples: int):
        x = torch.rand((num_samples, self.input_dim)) * self.box_range + self.lb
        return x

class SampleDatasetAdvancedTrajectories(Dataset):
    """
    This dataset generates trajectories in advance as producing trajectories during batch training
    can be a large bottleneck. This can only be done for a static controller. If the controller is also being
    trained, then this dataset will negatively affect the training procedure.
    """
    def __init__(self, dynamical_system, controller_fn: Callable, max_steps: int, n_samples: int, lb: Tensor, ub: Tensor,
                 gpu_frac: float = 0.8, device=torch.device('cpu'), verbose:bool=True):
        """

        :param dynamical_system:
        :param controller_fn:
        :param n_samples:
        :param lb:
        :param ub:
        :param verbose:
        """
        super().__init__()
        self.n_samples = n_samples
        self.lb = lb.reshape(1, -1)
        self.ub = ub.reshape(1, -1)
        self.input_dim = len(lb)
        self.box_range = self.ub - self.lb
        self.max_steps = max_steps
        self.dynamical_system = dynamical_system
        self.controller_fn = controller_fn
        self.device = device

        # Get information regarding device memory in case we need to run trajectories in batches. Even if we must
        # get trajectories in batches, if the batches to generate these trajectories are smaller than the trajectories
        # used for training, then it is still much faster to do this
        self.device_id = torch.cuda.current_device()
        self.total_memory_bytes = torch.cuda.get_device_properties(self.device_id).total_memory
        self.total_available_bytes = int(self.total_memory_bytes * gpu_frac)
        self.traj_memory_bytes = self.n_samples * self.input_dim * self.max_steps * 4
        self.traj_batches = ceil(self.traj_memory_bytes / self.total_available_bytes)
        self.traj_step_size = self.n_samples // self.traj_batches

        # initialize the dataset
        self.reset_samples()

        if verbose:
            print(f"n_samples = {n_samples}")
            print(f"lb shape = {self.lb.shape}, ub shape = {self.ub.shape}")
            print(f"box_range shape = {self.box_range.shape}")
            print(f"input_dim = {self.input_dim}")
            print(f"device_id = {self.device_id}")
            print(f"total_memory_bytes = {self.total_memory_bytes}")
            print(f"total_available_bytes = {self.total_available_bytes}")
            print(f"traj_memory_bytes = {self.traj_memory_bytes}")
            print(f"traj_batches = {self.traj_batches}")
            print(f"traj_step_size = {self.traj_step_size}")


    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self._samples[idx], self._trajectories[idx]

    def get_trajectories(self, samples: Tensor):
        """
        Return a new set of trajectories
        :param samples:
        :return:
        """
        new_traj = torch.empty((self.n_samples, self.max_steps, self.input_dim))

        i = 0
        start_idx = 0
        while start_idx < self.n_samples:
            stop_idx = min((i+1)*self.traj_step_size, self.n_samples)
            curr_x = samples[start_idx:stop_idx, :].to(device=self.device)
            curr_traj_batch = self.dynamical_system.forward_trajectory(curr_x, self.controller_fn, self.max_steps)
            new_traj[start_idx:stop_idx, :, :] = curr_traj_batch.cpu().clone()
            i+=1
            start_idx = stop_idx

        return new_traj

    def reset_samples(self):
        # free up memory by discarding previous samples
        self._samples = None
        self._trajectories = None
        # generates new samples
        self._samples = self.randomize_samples(self.n_samples)
        self._trajectories = self.get_trajectories(self._samples)

    def randomize_samples(self, num_samples: int):
        x = torch.rand((num_samples, self.input_dim)) * self.box_range + self.lb
        return x

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