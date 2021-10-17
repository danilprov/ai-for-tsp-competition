import os
import numpy as np
from copy import deepcopy
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from baseline_rl.batch_env_rl import BatchEnvRL
from model.get_td_error import get_td_error
from model.CNN import NeuralNetwork
from model.utilities.replay_memory import ReplayMemory
from model.utilities.softmax import softmax

experiment_config = {'n_nodes': 10,
                     'n_epochs': 200,
                     'n_sims': 100,
                     'minibatch_size': 64,
                     'num_replay': 10,
                     'discount': 0.95}
n_nodes = experiment_config['n_nodes']
n_epochs = experiment_config['n_epochs']
n_sims = experiment_config['n_sims']
batch_size = 1
minibatch_size = experiment_config['minibatch_size']
num_replay = experiment_config['num_replay']
discount = experiment_config['discount']

n_sims_val = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network_config = {'state_dim': 5 * n_nodes,
                  'num_actions': n_nodes,
                  'threshold': n_nodes * -20.}
model = NeuralNetwork(network_config=network_config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
buffer = ReplayMemory(500000)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))

model_dir = os.path.join('modelRelative', 'n_nodes_' + str(n_nodes))
if not os.path.exists(model_dir):
    print(f'Creating a new model directory: {model_dir}')
    os.makedirs(model_dir)

result = []
for e in range(0, n_epochs):
    print(f'Start epoch {e + 1}')
    env = BatchEnvRL(n_envs=batch_size, n_nodes=n_nodes, adaptive=False)
    av_reward_sim = []

    # get input
    batch = env.get_features2().copy()
    batch = torch.from_numpy(batch)
    batch = batch.to(device)
    batch = batch.transpose(-1, -2)

    min_window = batch[:, 0:1, :]
    max_window = batch[:, 1:2, :]
    max_time = batch[:, 3:4, :]
    dist_mat = batch[:, 4:, :]
    prizes = batch[:, 2:3, :]

    for b in range(0, n_sims):
        tours = torch.zeros(batch_size, 1, device=device)

        # will use it to change the original action and tour_t shapes
        multiplier = torch.ones(batch_size, 1, n_nodes, dtype=torch.int64, device=device)
        # start from depot with 0 time
        actions = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
        tour_time = torch.zeros(batch_size, 1, n_nodes, dtype=torch.int64, device=device)
        visited_nodes = torch.zeros(batch_size, 1, n_nodes, device=device)
        visited_nodes_ones = visited_nodes.clone()

        network_input = torch.cat(((min_window - tour_time) / max_time,
                                   (max_window - tour_time) / max_time,
                                   torch.gather(dist_mat, 1, actions.unsqueeze(1) * multiplier),
                                   prizes * (1 - visited_nodes_ones),
                                   visited_nodes), 1).float()

        step_ = 0
        if_break = False
        # making tour
        while step_ < n_nodes and not if_break:
            # choose a batch of actions (batch always equals to 1)
            with torch.no_grad():
                actions_values = model(network_input.float())
            # convert action values to probabilities
            probs = softmax(actions_values, tau=n_nodes / 20)

            # to enable exploration, pick action (next_node) randomly according to probs
            next_nodes = torch.multinomial(probs, 1, replacement=True).to(device)

            # update rl data
            for b in range(batch_size):
                visited_nodes[b, 0, next_nodes.squeeze(1)[b]] += 1
            visited_nodes_ones = torch.minimum(visited_nodes, multiplier)
            # append next node to tour
            tours = torch.cat((tours, next_nodes), 1)
            # states
            states = network_input.clone()
            # actions
            actions = next_nodes.clone()
            # rewards and terminals
            if step_ < n_nodes - 1 and next_nodes != 0:
                try:
                    terminals = torch.zeros_like(actions).to(device)
                    tour_time, time_t, rwd_t, pen_t, feas, violation_t = env.step(torch.add(next_nodes, 1).int())
                    R = rwd_t * 10 + pen_t
                    rewards = torch.from_numpy(R).float().to(device)
                    tour_time = torch.from_numpy(tour_time).to(device)
                except:
                    terminals = torch.ones_like(actions).to(device)
                    rewards = torch.ones_like(actions).to(device) * -3
                    tour_time = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
                    if_break = True
            else:
                if_break = True
                terminals = torch.ones_like(actions).to(device)
                # doesn't matter what is tour time now
                tour_time = torch.zeros(batch_size, 1, dtype=torch.int64, device=device)
                tours = torch.add(tours, 1).int()
                rwds, pens = env.check_solution(tours)
                R = rwds * 10 + pens + 3 * (len(tours[0]) - 3)
                rewards = torch.from_numpy(R).float().to(device)

            # next state
            next_states = torch.cat(((min_window - tour_time.unsqueeze(1) * multiplier) / max_time,
                                     (max_window - tour_time.unsqueeze(1) * multiplier) / max_time,
                                     torch.gather(dist_mat, 1, actions.unsqueeze(1) * multiplier),
                                     prizes * (1 - visited_nodes_ones),
                                     visited_nodes), 1).float()

            network_input = next_states.clone()

            # update buffer
            buffer.push(states, actions, next_states, rewards, terminals)
            # optimize network
            if buffer.__len__() > minibatch_size:
                current_q = deepcopy(model)
                for _ in range(num_replay):
                    # Get sample experiences from the replay buffer
                    transitions = buffer.sample(minibatch_size)
                    batch = Transition(*zip(*transitions))

                    state_batch = torch.cat(batch.state)
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)
                    next_state_batch = torch.cat(batch.next_state)
                    terminal_batch = torch.cat(batch.terminal)

                    loss = get_td_error(states, next_states, actions, rewards,
                                        discount, terminals, model, current_q)

                    actor_loss = loss.mean()
                    optimizer.zero_grad()
                    actor_loss.backward()
                    optimizer.step()

            step_ += 1
        av_reward_sim.append(rewards.float().mean())
        env.reset()

    print(f'Average reward during epoch {e + 1}: {sum(av_reward_sim) / max(1, len(av_reward_sim))}')
    result.append(sum(av_reward_sim) / max(1, len(av_reward_sim)))

    if e in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        checkpoint = {
            'actor': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'av_result': result,
            'experiment_config': experiment_config
        }
        torch.save(checkpoint, os.path.join(model_dir, f'relative-protocol-checkpoint-epoch-{e}.pt'))

    """
    ###########################################################
    VALIDATION STEP
    
    Fix an arbitrary environment and run the current model n_sims_val times to evaluate the model.
    Metric is an average reward over n_sims_val tours built by the model.
    ###########################################################
    """

    print('validation')
    env_val = BatchEnvRL(n_envs=1, n_nodes=n_nodes, adaptive=False)
    R_ = 0
    n_success_sims = 1
    while n_success_sims < n_sims_val + 1:
        tours = torch.zeros(1, 1, device=device)

        batch = env_val.get_features2().copy()
        batch = torch.from_numpy(batch)
        batch = batch.to(device)
        batch = batch.transpose(-1, -2)

        min_window = batch[:, 0:1, :]
        max_window = batch[:, 1:2, :]
        max_time = batch[:, 3:4, :]
        dist_mat = batch[:, 4:, :]
        prizes = batch[:, 2:3, :]

        # will use it to change the original action and tour_t shapes
        multiplier = torch.ones(1, 1, n_nodes, dtype=torch.int64, device=device)
        # start from depot
        actions = torch.zeros(1, 1, dtype=torch.int64, device=device)
        tour_time = torch.zeros(1, 1, n_nodes, dtype=torch.int64, device=device)
        visited_nodes = torch.zeros(1, 1, n_nodes, device=device)
        visited_nodes_ones = visited_nodes.clone()

        # first network input (staying in depot)
        network_input = torch.cat(((min_window - tour_time) / max_time,
                                   (max_window - tour_time) / max_time,
                                   torch.gather(dist_mat, 1, actions.unsqueeze(1) * multiplier),
                                   prizes * (1 - visited_nodes_ones),
                                   visited_nodes), 1).float()

        step_ = 0
        # make tour
        while step_ < n_nodes:
            with torch.no_grad():
                actions_values = model(network_input)
            # to make valid tours on the validation phase, taking into account only action values of unvisited nodes
            actions_values = torch.where(actions_values * (1 - visited_nodes_ones) == 0,
                                         2 * model.threshold, actions_values.double()).squeeze(1)
            probs = nn.Softmax(dim=1)(actions_values)

            # update next state
            next_nodes = torch.multinomial(probs, 1, replacement=True).to(device)
            visited_nodes[:, 0, next_nodes.squeeze(1)] += 1
            visited_nodes_ones = torch.minimum(visited_nodes, multiplier)
            tours = torch.cat((tours, next_nodes), 1)
            tour_time, time_t, rwd_t, pen_t, feas, violation_t = env_val.step(torch.add(next_nodes, 1).int())
            tour_time = torch.from_numpy(tour_time).to(device)

            next_states = torch.cat(((min_window - tour_time.unsqueeze(1) * multiplier) / max_time,
                                     (max_window - tour_time.unsqueeze(1) * multiplier) / max_time,
                                     torch.gather(dist_mat, 1, next_nodes.unsqueeze(1) * multiplier),
                                     prizes * (1 - visited_nodes_ones),
                                     visited_nodes), 1).float()
            network_input = next_states.clone()
            step_ += 1

        # evaluate tour
        tours = torch.add(tours, 1).int()
        rwds, pens = env_val.check_solution(tours)
        R = rwds + pens
        R_ += R
        n_success_sims += 1
        # reset env to start the model all over again
        env.reset()
    result.append(np.mean(R_) / n_sims_val)
    print(f'validation avg rwds {np.mean(R_) / n_sims_val}')

checkpoint = {
    'actor': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'av_result': result,
    'experiment_config': experiment_config
}
torch.save(checkpoint, os.path.join(model_dir, f'relative-protocol.pt'))
