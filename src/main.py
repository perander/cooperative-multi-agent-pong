import numpy as np
import torch
from environment import pong
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.utils.tensorboard import SummaryWriter
from model.model import DQNAgent

if __name__ == "__main__":
    writer = SummaryWriter("src/runs/dqn")

    seed = 123
    torch_deterministic = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    env_seed = 123

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_size = (84, 84)
    stack_size = 4

    total_episodes = 1
    max_cycles = 2000
    plot_every_k_steps = 50
    train_frequency = 4

    batch_size = 32
    buffer_size = 1000000

    # dqn specific?
    t_learning_starts = batch_size
    target_network_frequency = 10000
    # lower -> target network changes more slowly
    tau = 0.005
    lr = 0.0001
    gamma = 0.99
    start_e = 1
    end_e = 0.05
    exploration_fraction = 0.1

    # ppo specific
    alpha = 0.0003
    gae_lambda = 0.95
    policy_clip = 0.2

    # competitive, cooperative, pd
    reward_structure = "competitive"

    env = pong.parallel_env(
        render_mode="human",
        max_cycles=max_cycles,
        bounce_randomness=True,
        reward_structure=reward_structure,
        ball_direction_randomness=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)

    n_agents = 2

    agent_names = env.possible_agents
    print(agent_names)

    agents = [
        (
            name,
            DQNAgent(
                env.observation_space(env.possible_agents[0]).shape,
                env.action_space(env.possible_agents[0]).n,
                env.action_space(env.possible_agents[0]),
                batch_size,
                buffer_size,
                lr,
                start_e,
                end_e,
                exploration_fraction,
                gamma,
                tau,
                device,
            ),
        )
        for name in agent_names
    ]

    print("agents", agents)

    obs = env.reset(seed=env_seed)
    loss = 0
    mean_q = 0

    for step in range(max_cycles):
        actions = {}

        for name, agent in agents:
            actions[name] = agent.choose_action(obs)

        next_obs, rewards, terms, truncs, infos = env.step(actions)
        # print("rewards", rewards)

        for i, (name, agent) in enumerate(agents):
            # print("obs:", obs.shape, "next_obs:", next_obs[name].shape, "actions:", actions[name], "rewards:", rewards[name])

            agent.remember(obs, next_obs, actions[name], rewards[name])

            loss, mean_q = agent.learn(
                step, t_learning_starts, train_frequency, max_cycles
            )

        print("step", step, "loss", loss, "mean q value", mean_q)

        obs = next_obs
        env.render()
