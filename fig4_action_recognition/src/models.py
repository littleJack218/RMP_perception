"""Network architectures of the action-recognition readouts."""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Single-scale readout: 6000-300-10 (paper Methods)."""

    def __init__(self, input_dim=6000, hidden=300, out=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc3 = nn.Linear(hidden, out)

    def forward(self, x):
        return self.fc3(F.relu(self.fc1(self.flatten(x))))


class SimpleCNN(nn.Module):
    """Mixed-scale readout: Conv3-8 / Conv3-12 + 100/10 (paper Methods)."""

    def __init__(self, in_ch=4, hw=(100, 60)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 12, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(12 * (hw[0] // 4) * (hw[1] // 4), 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(x))))))
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


def n_params(model):
    return sum(p.numel() for p in model.parameters())
