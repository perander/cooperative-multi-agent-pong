import random
from collections import namedtuple, deque

Transition = namedtuple(
    "Transition",
    ("obs", "next_obs", "actions", "rewards"),
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
