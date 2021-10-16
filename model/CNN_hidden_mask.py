import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNetwork(nn.Module):
    def __init__(self, network_config):
        super(NeuralNetwork, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_actions = network_config.get("num_actions")
        self.threshold = network_config.get("threshold")

        self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(self.state_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.GELU(),
        #     nn.Linear(512, self.num_actions),
        #     nn.Tanh()
        # )
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


if __name__ == '__main__':
    import torch.optim as optim

    from baseline_rl.batch_env_rl import BatchEnvRL
    from model.get_td_error import get_td_error

    batch_size = 4
    n_nodes = 5
    discount = 0.99
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network_config = {'state_dim': 5 * n_nodes, 'num_actions': n_nodes}
    model = NeuralNetwork(network_config=network_config).to(device)

    env = BatchEnvRL(n_envs=batch_size, n_nodes=n_nodes, adaptive=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # GET INPUT
    batch = env.get_features1().copy()
    batch = torch.from_numpy(batch)
    batch = batch.to(device)
    batch = batch.transpose(-1, -2)

    visited_nodes = torch.zeros(batch_size, 1, n_nodes, device=device)
    # batch = torch.cat((batch, visited_nodes), 1)
    tours = torch.zeros(batch_size, 1, device=device)
    # loss = torch.zeros(batch_size, 1, device=device)

    network_input = batch.float()

    for step in range(n_nodes):
        with torch.no_grad():
            actions_values = model(network_input)

        probs = nn.Softmax(dim=1)(actions_values)
        next_nodes = torch.multinomial(probs, 1, replacement=True).to(device)
        available_nodes = probs * (1 - visited_nodes[:, 0, :])
        next_nodes_adj = torch.multinomial(available_nodes, 1, replacement=True).to(device)

        for b in range(batch_size):
            visited_nodes[b, 0, next_nodes_adj.squeeze(1)[b]] = 1
        # visited_nodes[:, 0, next_nodes.squeeze(1)] = 1
        tours = torch.cat((tours, next_nodes_adj), 1)

        network, current_q = model, model

        states = network_input.clone()
        actions = next_nodes_adj
        if step < n_nodes - 1:
            # option 1
            # rewards = torch.zeros_like(actions)
            # terminals = torch.zeros_like(actions).to(device)
            # option 2
            terminals = torch.zeros_like(actions).to(device)
            rwds, pens = env.step(torch.add(next_nodes_adj, 1).int())[2:4]
            R = rwds + pens
            rewards = torch.from_numpy(R).float().to(device)
        else:
            terminals = torch.ones_like(actions).to(device)
            tours = torch.add(tours, 1).int()
            rwds, pens = env.check_solution(tours)
            R = rwds + pens
            rewards = torch.from_numpy(R).float().to(device)
        # update decoder input
        # network_input[:, -1, :] = visited_nodes[:, -1, :]
        network_input = network_input * (1 - visited_nodes)
        next_states = network_input.clone()

        loss = get_td_error(states, next_states, actions, rewards,
                            discount, terminals, network, current_q)

        actor_loss = loss.mean()
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()
