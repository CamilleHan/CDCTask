'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        """Initialize the network structure."""
        super(LeNet, self).__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # Output: 6x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 6x14x14
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # Output: 16x10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x5x5
        )

        self.block_2 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        """Define the forward pass."""
        x = self.block_1(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.block_2(x)
        return x
