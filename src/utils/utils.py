import numpy as np
import torch


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def get_expected_q_values(
    non_final_mask, non_final_next_obs, batch_size, target, reward_batch, gamma, device
):

    target_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        target_values[non_final_mask] = target(non_final_next_obs).max(1)[0]

    expected_q_values = reward_batch + target_values * gamma

    return expected_q_values


def get_q_values(network, action_batch, obs_batch):
    return (
        network(obs_batch)
        .gather(1, action_batch.unsqueeze(1).type(torch.int64))
        .squeeze()
    )

def soft_update_target_network(network, target, tau):
    network_state_dict = network.state_dict()
    target_state_dict = target.state_dict()

    for key in network_state_dict:
        target_state_dict[key] = network_state_dict[key] * tau + target_state_dict[
            key
        ] * (1 - tau)
    target.load_state_dict(target_state_dict)

    return target