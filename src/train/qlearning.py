import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from model.model import QNetwork
from utils.utils import linear_schedule, batchify_obs
from train.replay_memory import ReplayMemory, Transition


def get_expected_q_values(
    non_final_mask, non_final_next_obs, batch_size, target, reward_batch, gamma, device
):

    target_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        target_values[non_final_mask] = target(non_final_next_obs).max(1)[0]

    expected_q_values = target_values * gamma + reward_batch

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


def select_action(env, i, agent, network, obs, epsilon, device):
    if np.random.uniform() < epsilon:
        action = np.array(env.action_space(agent).sample())
        action = torch.tensor(action).to(device)
        return action.item()
    else:
        with torch.no_grad():
            q_values = network(obs)
        action = torch.argmax(q_values, dim=1)
        return action[i].item()


def optimize_model(i, memory, batch_size, gamma, network, target, optimizer, device):
    if len(memory) < batch_size:
        return None, None

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    obs_batch = torch.cat(batch.obs)
    actions_batch = torch.cat(batch.actions)
    rewards_batch = torch.cat(batch.rewards)

    action_batch = actions_batch[:, i]
    reward_batch = rewards_batch[:, i]

    # print("rewards", reward_batch)

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_obs)),
        device=device,
        dtype=torch.bool,
    )

    non_final_next_obs = torch.cat([s for s in batch.next_obs if s is not None])

    q_values = get_q_values(network, action_batch, obs_batch)

    expected_q_values = get_expected_q_values(
        non_final_mask,
        non_final_next_obs,
        batch_size,
        target,
        reward_batch,
        gamma,
        device,
    )

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()

    for param in network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item(), q_values.mean().item()


def qlearning(
    env,
    env_seed,
    writer,
    device,
    n_agents=2,
    planner=None,
    batch_size=32,
    buffer_size=1000000,
    lr=0.0001,
    gamma=0.99,
    start_e=1,
    end_e=0.1,
    exploration_fraction=0.1,
    total_episodes=1,
    max_cycles=10000,
    t_learning_starts=1000,
    plot_every_k_steps=100,
    train_frequency=4,
    tau=0.005,
):

    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape
    agents = env.possible_agents  # names
    print("agents", agents)

    networks = []
    optimizers = []
    targets = []

    for agent in range(n_agents):
        network = QNetwork(observation_size, num_actions).to(device)
        optimizer = optim.AdamW(network.parameters(), lr=lr)

        target = QNetwork(observation_size, num_actions).to(device)
        target.load_state_dict(network.state_dict())
        target.eval()

        networks.append(network)
        optimizers.append(optimizer)
        targets.append(target)

    memory = ReplayMemory(buffer_size)

    for episode in range(total_episodes):
        paddle_hits = 0
        paddle_hits_per_score = []
        losses_left = []
        losses_right = []
        score_left = 0
        score_right = 0

        obs = env.reset(seed=env_seed)
        obs = batchify_obs(obs, device)

        for step in range(0, max_cycles):
            actions = {}
            for i, network in enumerate(networks):
                agent = env.possible_agents[i]

                epsilon = linear_schedule(
                    start_e, end_e, exploration_fraction * max_cycles, step
                )

                # TODO agent is referred to in 3 different ways (i, agent name, network), can i simplify this
                actions[agent] = select_action(
                    env, i, agent, network, obs, epsilon, device
                )

            next_obs, rewards, terms, truncs, infos = env.step(actions)

            # print("rewards", rewards)
            # print("infos", infos)

            next_obs = batchify_obs(next_obs, device)
            next_obs_formatted = next_obs[0].unsqueeze(0)

            # TODO planner selects action r_p, it is added to replay memory with env rewards

            actions_values = [v for _, v in sorted(actions.items())]
            rewards_values = [v for _, v in sorted(rewards.items())]

            memory.push(
                obs[0].unsqueeze(0),
                next_obs_formatted,
                torch.tensor([actions_values]),
                torch.tensor([rewards_values]),
            )

            obs = next_obs

            wall = infos["paddle_0"][1]
            paddle_hit = infos["paddle_0"][2]

            if paddle_hit:
                # print("paddle hit")
                paddle_hits += 1
                # print("paddle hits so far", paddle_hits)
            if wall:
                paddle_hits_per_score.append(paddle_hits)
                paddle_hits = 0
                if wall == "left":
                    score_right += 1
                if wall == "right":
                    score_left += 1

            env.render()

            if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                end_step = step
                obs = batchify_obs(env.reset(seed=None), device)
                continue

            if step > t_learning_starts:

                if step % train_frequency == 0:
                    losses = []
                    mean_q_values = []

                    for i in range(n_agents):
                        loss, mean_q_value = optimize_model(
                            i,
                            memory,
                            batch_size,
                            gamma,
                            networks[i],
                            targets[i],
                            optimizers[i],
                            device,
                        )

                        # TODO simplify
                        if loss != None:
                            losses.append(loss)
                            mean_q_values.append(mean_q_value)

                    if loss != None:

                        mean_q_value_left, mean_q_value_right = mean_q_values
                        loss_left, loss_right = losses
                        losses_left.append(loss_left)
                        losses_right.append(loss_right)

                    if step % plot_every_k_steps == 0:
                        print(
                            f"step {step}, avg paddle hits cooperative {np.mean(paddle_hits_per_score[-plot_every_k_steps:])}, score: {score_left}, {score_right}"
                        )
                        writer.add_scalar(f"loss_coop_left", loss_left, step)
                        writer.add_scalar(f"loss_coop_right", loss_right, step)
                        writer.add_scalar(
                            f"q_values_coop_left", mean_q_value_left, step
                        )
                        writer.add_scalar(
                            f"q_values_coop_right", mean_q_value_right, step
                        )
                        writer.add_scalar(
                            f"avg_paddle_hits_per_score_cooperative_agents",
                            np.mean(paddle_hits_per_score[-plot_every_k_steps:]),
                            step,
                        )

                    targets[0] = soft_update_target_network(
                        networks[0], targets[0], tau
                    )
                    targets[1] = soft_update_target_network(
                        networks[1], targets[1], tau
                    )

        writer.add_scalar(
            "avg_paddle_hits_per_score_per_episode",
            np.mean(paddle_hits_per_score),
            episode,
        )
        writer.add_scalar("steps_per_episode", step, episode)
        writer.add_scalar("avg_loss_left_per_epoch", np.mean(losses_left), episode)
        writer.add_scalar("avg_loss_right_per_epoch", np.mean(losses_right), episode)
        writer.add_scalar("q_values_left_per_epoch", mean_q_value_left, episode)
        writer.add_scalar("q_values_right_per_epoch", mean_q_value_right, episode)

    env.close()
    writer.close()
