import numpy as np
import torch
from environment import pong
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.utils.tensorboard import SummaryWriter
from train.qlearning import qlearning

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
    max_cycles = 200000
    t_learning_starts = 1
    target_network_frequency = 10000
    plot_every_k_steps = 50
    train_frequency = 4

    # lower -> target network changes more slowly
    tau = 0.005

    lr = 0.0001
    gamma = 0.99

    start_e = 1
    end_e = 0.05
    exploration_fraction = 0.1

    batch_size = 32
    buffer_size = 1000000

    env = pong.parallel_env(
        render_mode="human",
        max_cycles=max_cycles,
        bounce_randomness=True,
    )

    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)

    n_agents = 2

    qlearning(
        env,
        env_seed,
        writer,
        device,
        n_agents,
        None,
        batch_size,
        buffer_size,
        lr,
        gamma,
        start_e,
        end_e,
        exploration_fraction,
        total_episodes,
        max_cycles,
        t_learning_starts,
        plot_every_k_steps,
        train_frequency,
        tau,
    )
