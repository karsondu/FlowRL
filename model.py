import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):

    def __init__(self, num_actions, input_shape):
        super().__init__()

        channels, rows, cols = input_shape

        #first layer, from chanels to 32 features with 3x3 grid look at a time
        # Then ReLU and then going to 64 features 
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        #64 features for each cell then reduced down to 256 neurons to start prediction
        #Going from 256 neurons to number of actions where the output is a vector of scoring for each possible action
        self.fc = nn.Sequential(
            nn.Linear(64 * rows * cols, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x