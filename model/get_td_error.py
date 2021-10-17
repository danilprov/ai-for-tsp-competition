import numpy as np
import torch
from torch import nn


def get_td_error(states, next_states, actions, rewards,
                 discount, terminals, network, current_q):
    """
    Args:
        states (Numpy array): The batch of states with the shape (batch_size, state_dim).
        next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
        actions (Numpy array): The batch of actions with the shape (batch_size,).
        rewards (Numpy array): The batch of rewards with the shape (batch_size,).
        discount (float): The discount factor.
        terminals (Numpy array): The batch of terminals with the shape (batch_size,).
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                        and particularly, the action-values at the next-states.
    Returns:
        The TD errors (Numpy array) for actions taken, of shape (batch_size,)
    """

    # Compute action values at next states using current_q network
    with torch.no_grad():
        q_next_mat = current_q(next_states)

    # Note that nn.Softmax doesn't have parameter tau (tau=1, always)
    probs_mat = nn.Softmax(dim=1)(q_next_mat)

    # Compute the estimate of the next state value, v_next_vec.
    next_vec_weighted_probs_mat = torch.sum(q_next_mat * probs_mat, axis=1)
    v_next_vec = next_vec_weighted_probs_mat * (1 - terminals)

    # Compute Expected Sarsa target
    target_vec = rewards + discount * v_next_vec

    # Compute action values at the current states for all actions using network
    q_mat = network(states)

    # Batch Indices is an array from 0 to the batch size - 1.
    batch_indices = np.arange(q_mat.shape[0])

    # Compute q_vec by selecting q(s, a) from q_mat for taken actions
    # q_vec = q_mat[batch_indices, actions]
    q_vec = q_mat[batch_indices, actions.squeeze(1).cpu().numpy()]

    # Compute TD errors for actions taken
    criterion = nn.SmoothL1Loss()
    loss = criterion(target_vec.unsqueeze(1), q_vec.unsqueeze(1))

    return loss
