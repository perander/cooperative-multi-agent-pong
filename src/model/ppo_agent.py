import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from train.ppo_memory import PPOMemory, Transition
from utils.utils import batchify_obs


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr):
        super().__init__()

        (_, _, channels) = input_dims

        self.actor = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        dist = self.actor(x / 255.0)
        dist = Categorical(dist)

        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, lr):
        super(CriticNetwork, self).__init__()

        (_, _, channels) = input_dims

        self.critic = nn.Sequential(
            nn.Conv2d(channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.critic(x / 255.0)


class Agent:
    def __init__(
        self,
        n_actions,
        input_dims,
        gamma=0.99,
        lr=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=64,
        n_epochs=10,
        memory_size=1000000,
        ent_coef=0.01,
        vf_coef=0.5,
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions, input_dims, lr)
        self.critic = CriticNetwork(input_dims, lr)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = PPOMemory(input_dims, self.batch_size, self.memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

    def remember(self, i, obs, next_obs, action, reward, prob, val):
        obs = batchify_obs(obs, self.device)
        obs = obs[0].unsqueeze(0)
        next_obs = batchify_obs(next_obs, self.device)
        next_obs = next_obs[0].unsqueeze(0)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        prob = torch.tensor([prob])
        val = torch.tensor([val])

        self.memory.store_memory(i, obs, next_obs, action, reward, prob, val)

    def choose_action(self, obs):
        obs = batchify_obs(obs, self.device)
        obs = obs[0].unsqueeze(0)

        dist = self.actor(obs)
        value = self.critic(obs)

        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, dist.entropy(), value

    def compute_gae(self, rewards, values, next_values, gamma, gae_lambda):
        num_steps = len(rewards)
        advantages = np.zeros(num_steps)
        last_gae = 0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                delta = rewards[t] + gamma * next_values[t] - values[t]
            else:
                delta = rewards[t] + gamma * values[t + 1] - values[t]

            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae

        return advantages

    def normalize_advantages(self, advantages):
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def get_policy_loss(self, states, actions, old_probs, advantages, policy_clip):
        dist = self.actor(states)
        new_probs = dist.log_prob(actions)
        prob_ratio = (new_probs - old_probs).exp()

        weighted_probs = -advantages * prob_ratio
        weighted_clipped_probs = -advantages * torch.clamp(
            prob_ratio, 1 - policy_clip, 1 + policy_clip
        )
        policy_loss = torch.max(weighted_probs, weighted_clipped_probs).mean()

        return policy_loss

    def get_value_loss(self, states, advantages, values):
        returns = advantages + values
        critic_value = self.critic(states)
        critic_value = torch.squeeze(critic_value)

        critic_loss = (returns - critic_value) ** 2
        critic_loss = critic_loss.mean()

        return critic_loss

    def learn(self):
        for epoch in range(self.n_epochs):
            (
                state_arr,
                next_state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                # dones_arr, TODO add
                batches,
            ) = self.memory.generate_batches(self.device)

            next_states = torch.Tensor(next_state_arr).to(self.device)
            next_values = self.critic(next_states)
            advantages_arr = self.compute_gae(
                reward_arr, vals_arr, next_values, self.gamma, self.gae_lambda
            )

            # For each batch, update actor and critic
            for batch in batches:
                states = torch.Tensor(state_arr[batch]).to(self.device)
                old_probs = torch.Tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.Tensor(action_arr[batch]).to(self.device)
                values = torch.Tensor(vals_arr[batch]).to(self.device)
                advantages = torch.Tensor(advantages_arr[batch]).to(self.device)

                advantages = self.normalize_advantages(advantages)

                # losses
                policy_loss = self.get_policy_loss(
                    states, actions, old_probs, advantages, self.policy_clip
                )

                value_loss = self.get_value_loss(states, advantages, values)

                dist = self.actor(states)
                entropy = dist.entropy()
                entropy_loss = entropy.mean()

                total_loss = (
                    policy_loss
                    - self.ent_coef * entropy_loss
                    + value_loss * self.vf_coef
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                # nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                # nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
        return total_loss.item(), None
