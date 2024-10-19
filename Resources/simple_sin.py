import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from tqdm import tqdm
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np

print(plt.style.available)
plt.rcParams['text.usetex'] = True
plt.style.use("seaborn-v0_8-white")

save_model = False
model_path = "./models/"
model_name = "SimpleSin_model"
device = torch.device('cuda')

class SimpleSin(nn.Module):
    def __init__(self,
                 lrate: float,
                 loss_fn,
                 ):
        """
        Initialize simple Sin model
        :param lrate:
        :param loss_fn:
        """
        super(SimpleSin, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.optimizer = optim.SGD(self.parameters(), lr=lrate, weight_decay=1e-5)

        # unpack bounding_loss dict
        self.loss_fn = loss_fn


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model
        :param x:
        :return:
        """
        return self.model(x)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Returns the loss of a single forward pass
        :param x:   (Batch, input size)
        :param y:   (Batch, output size)
        :return:    loss
        """
        # zero the gradients
        self.optimizer.zero_grad()

        # pass the batch through the model
        y_hat = self.forward(x)

        # compute the loss
        loss = self.loss_fn(y_hat, y)


        # update model
        loss.backward()
        self.optimizer.step()

        return loss.item()

class SinDataset(Dataset):
    def __init__(
            self,
            input_range: Tuple[float, float],
            num_samples: Optional[int] = 10000
    ):
        """
        Creates a dataset uniformly sampling the sin function
        :param input_range: Sin input range
        :param num_samples: Number of samples on the input range
        """
        self.x = torch.linspace(input_range[0], input_range[1], num_samples)
        self.x = self.x.unsqueeze(1) # turn into bathes
        self.y = torch.sin(self.x) + 1e-1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def fit_model(
        input_range: Tuple[float, float],
        epochs: int,
        lrate: float = 0.01,
        loss_fn = nn.MSELoss(),
        batch_size: int = 100,
        samples: int = 10000
) -> Tuple[list[float], SimpleSin]:
    """
    Fits a simple sin model to the ground truth sin model
    :param input_range:     Sin input range
    :param epochs:          Number of epochs to run
    :param lrate:           Learning rate
    :param loss_fn:         Loss function
    :param batch_size:      Batch size per epoch
    :param samples:         Number of samples on the input range
    :param bounding_loss:   Parameters for the bounding loss function indicating it should be used
    :return:                Losses and network object
    """

    # get data loaders
    train_dataset = SinDataset(input_range, samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # initialize net and send to device
    NetObject = SimpleSin(lrate, loss_fn).to(device)

    losses = []
    # for _ in range(2):
    epoch_progress_bar = tqdm(range(epochs), desc="Epochs", leave=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            curr_epoch_loss = NetObject.step(batch_x, batch_y)
            epoch_loss += curr_epoch_loss
            losses.append(curr_epoch_loss)
        epoch_loss /= len(train_loader)
        epoch_progress_bar.update(1)
        epoch_progress_bar.set_postfix({'Epoch Loss': epoch_loss})

        # NetObject.swap_loss_fn()

    return losses, NetObject

def visualize_model(
        input_range: Tuple[float, float],
        samples: int,
        losses: list[float],
        NetObject: SimpleSin
):
    """
    Plots a side-by-side graph of the network/sin output and training loss
    :param input_range: Sin input range
    :param samples:     Number of samples on the input range
    :param losses:      Losses seen over the number of training iterations
    :param NetObject:   The Neural Network object
    :return:            None
    """
    # Opens matplotlib graph comparing
    # how close our graph is to the true sin function
    # Define x values
    h_L = -5
    h_U = 5
    v_L = -1.
    v_U = 1.5
    x_L, x_U = input_range[0], input_range[1]
    x_samples = torch.linspace(x_L, x_U, samples).unsqueeze(1).to(device=device)

    # set the network to evaluation mode
    NetObject.eval()

    # gets outputs
    NetOut = NetObject.forward(x_samples).detach().cpu().numpy()
    GroundOut = torch.sin(x_samples).cpu().numpy()
    x_samples = x_samples.squeeze().cpu().numpy()

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the curves
    ax1.plot(x_samples, NetOut, color='blue', linewidth=2, label=r"$Net(x)$")
    ax1.plot(x_samples, GroundOut, color='red', linewidth=2, label=r"$Sin(x)$")

    # Fill the area between the red and green lines
    # ax1.fill_between(x, xL_h, xU_h, where=(xL_h < xU_h), interpolate=True, color='gray', alpha=0.3)

    # Customize the x and y limits
    ax1.set_xlim(h_L, h_U)
    ax1.set_ylim(v_L, v_U)

    # Add dashed lines for vertical and horizontal axes
    ax1.vlines(x=x_L, ymin=h_L, ymax=h_U, color='black', linestyle='--')
    ax1.vlines(x=x_U, ymin=h_L, ymax=h_U, color='black', linestyle='--')
    ax1.set_xticks(list(ax1.get_xticks()) + [x_L, x_U])

    # Label axes
    ax1.set_xlabel(r'$x$', fontsize=12)
    ax1.set_ylabel(r'$f(x)$', fontsize=12)
    ax1.set_title("SinModel vs Ground Truth Sin Function")
    ax1.grid()
    ax1.legend()

    # Plot the MLE Loss
    iterations = torch.arange(len(losses)).numpy()
    ax2.plot(iterations, losses, color='blue', linewidth=2)

    # Label axes
    ax2.set_xlabel(r'Iterations', fontsize=12)
    ax2.set_ylabel(r'Loss', fontsize=12)
    ax2.set_title("MLE Loss Over epochs")
    ax2.grid()

    plt.show()

if __name__ == '__main__':
    """
    Trains the SimpleSin network
    """
    input_range = (-3, 2)
    samples = 10000

    fit_params = {
        "input_range": input_range,
        "epochs": 50,
        "lrate": 0.01,
        "loss_fn": nn.MSELoss(),
        "batch_size": 100,
        "samples": samples
    }
    losses, NetObject = fit_model(**fit_params)

    visualize_params = {
        "input_range": input_range,
        "samples": samples,
        "losses": losses,
        "NetObject": NetObject
    }
    visualize_model(**visualize_params)
