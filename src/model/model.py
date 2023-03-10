import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from train.replay_memory import ReplayMemory, Transition
from utils.utils import (
    batchify_obs,
    get_q_values,
    get_expected_q_values,
    soft_update_target_network,
    linear_schedule,
)


class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, lr):
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

        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        return self.network(x / 255.0)


class DQNAgent:
    def __init__(
        self,
        observation_size,
        n_actions,
        action_space,
        batch_size,
        memory_size,
        lr,
        start_e,
        end_e,
        exploration_fraction,
        gamma,
        tau,
        device,
    ):
        self.policy_network = QNetwork(observation_size, n_actions, lr)
        self.target_network = QNetwork(observation_size, n_actions, lr)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size

        self.action_space = action_space
        self.epsilon = start_e  # will be modified during training
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.gamma = gamma
        self.tau = tau

        self.device = device
        self.policy_network.to(self.device)
        self.target_network.to(self.device)

    def remember(self, obs, next_obs, action, reward):
        obs = batchify_obs(obs, self.device)
        obs = obs[0].unsqueeze(0)
        next_obs = batchify_obs(next_obs, self.device)
        next_obs = next_obs[0].unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward])

        self.memory.push(obs, next_obs, action, reward)

    def choose_action(self, obs):
        obs = batchify_obs(obs, self.device)
        obs = obs[0].unsqueeze(0)

        if np.random.uniform() < self.epsilon:
            action = np.array(self.action_space.sample())
            action = torch.tensor(action).to(self.device)
            return action.item()
        else:
            with torch.no_grad():
                q_values = self.policy_network(obs)
            action = torch.argmax(q_values, dim=1)
            return action[0].item()

    def learn(self, step, t_learning_starts, train_frequency, max_cycles):
        self.epsilon = linear_schedule(
            self.start_e, self.end_e, self.exploration_fraction * max_cycles, step
        )

        if step > t_learning_starts:
            if step % train_frequency == 0:

                transitions = self.memory.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

                obs_batch = torch.cat(batch.obs)
                action_batch = torch.cat(batch.actions)
                reward_batch = torch.cat(batch.rewards)

                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_obs)),
                    device=self.device,
                    dtype=torch.bool,
                )

                non_final_next_obs = torch.cat(
                    [s for s in batch.next_obs if s is not None]
                )

                q_values = get_q_values(self.policy_network, action_batch, obs_batch)

                expected_q_values = get_expected_q_values(
                    non_final_mask,
                    non_final_next_obs,
                    self.batch_size,
                    self.target_network,
                    reward_batch,
                    self.gamma,
                    self.device,
                )

                loss = F.mse_loss(q_values, expected_q_values)
                self.policy_network.optimizer.zero_grad()
                loss.backward()

                for param in self.policy_network.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.policy_network.optimizer.step()

                self.target_network = soft_update_target_network(
                    self.policy_network, self.target_network, self.tau
                )

                return loss.item(), q_values.mean().item()

        # find another way to do this
        return None, None
