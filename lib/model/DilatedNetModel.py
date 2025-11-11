import torch
import torch.nn as nn
import torch.nn.functional as F

# neural network with dilated convolutions

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)
        self.conv5 = nn.Conv2d(64, 1, 1)  # final output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        logits = self.conv5(x)  # final output layer
        return logits   

