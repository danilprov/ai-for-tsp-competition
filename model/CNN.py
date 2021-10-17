import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self, network_config):
        super(NeuralNetwork, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_actions = network_config.get("num_actions")
        self.threshold = network_config.get("threshold")

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
            nn.Threshold(self.threshold, self.threshold)
        )

    def forward(self, x):
        x = self.flatten(x)
        q_values = self.linear_relu_stack(x)
        return q_values
