import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)

        self.value_head = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.value_head(x)
        a = self.advantage_head(x)
        a = a - torch.mean(a)
        return v + a
