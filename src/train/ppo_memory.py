import numpy as np
import random
import torch
from collections import namedtuple, deque
from utils.utils import batchify_obs

Transition = namedtuple(
    "Transition", ("obs", "next_obs", "actions", "rewards", "probs", "vals")
)


class PPOMemory(object):
    def __init__(self, obs_dim, batch_size, capacity):
        self.obs = np.zeros((capacity, *obs_dim[::-1]))
        self.next_obs = np.zeros((capacity, *obs_dim[::-1]))
        self.actions = np.zeros(capacity)
        self.rewards = np.zeros(capacity)
        self.values = np.zeros(capacity)
        self.advantages = np.zeros(capacity)
        self.probs = np.zeros(capacity)

        self.capacity = capacity
        self.batch_size = batch_size
        self.obs_dim = obs_dim

    def generate_batches(self, device):
        n_obs = len(self.obs)
        batch_starts = np.arange(0, n_obs, self.batch_size)
        indices = np.arange(n_obs, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_starts]

        return (
            self.obs,
            self.next_obs,
            self.actions,
            self.probs,
            self.values,
            self.rewards,
            batches,
        )

    def store_memory(self, i, obs, next_obs, action, reward, prob, val):
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.probs[i] = prob
        self.values[i] = val
        self.rewards[i] = reward

    def clear_memory(self):
        self.obs = []
        self.next_obs = []
        self.probs = []
        self.actions = []
        self.rewards = []
        # self.dones = [] # TODO add
        self.vals = []

        self.obs = np.zeros((self.capacity, *self.obs_dim[::-1]))
        self.next_obs = np.zeros((self.capacity, *self.obs_dim[::-1]))
        self.actions = np.zeros(self.capacity)
        self.rewards = np.zeros(self.capacity)
        self.values = np.zeros(self.capacity)
        self.advantages = np.zeros(self.capacity)
        self.probs = np.zeros(self.capacity)

    def __len__(self):
        return len(self.obs)
