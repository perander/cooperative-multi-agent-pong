import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        (width, height, channels) = input_dim

        self.network = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)
