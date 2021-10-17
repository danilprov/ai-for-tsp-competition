import random
from collections import namedtuple, deque

import torch


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    batch_size = 2
    n_nodes = 5

    buffer = ReplayMemory(5000)

    print(buffer.__len__())

    for _ in range(32):
        states = torch.rand(batch_size, 6, n_nodes)
        next_states = torch.rand(batch_size, 6, n_nodes)
        rewards = torch.rand(batch_size, 1, n_nodes)
        terminals = torch.rand(batch_size, 1, n_nodes)
        actions = torch.rand(batch_size, 1, n_nodes)

        buffer.push(states, actions, next_states, rewards, terminals)

    print(buffer.__len__())

    transitions = buffer.sample(16)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    terminal_batch = torch.cat(batch.terminal)

    print(batch)
    print(state_batch.shape)
    print(action_batch.shape)
    print(reward_batch.shape)
    print(next_state_batch.shape)
    print(terminal_batch.shape)
