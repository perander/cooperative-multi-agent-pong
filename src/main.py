import numpy as np
import torch
from environment import pong
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.utils.tensorboard import SummaryWriter
from model.model import DQNAgent
from model.ppo_agent import Agent as PPOAgent


if __name__ == "__main__":
    writer = SummaryWriter("src/runs/ppo")
    writer.close()

    seed = 124
    torch_deterministic = True

    dummy_metrics = [1, 2]  # action numbers

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    env_seed = seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_size = (84, 84)
    stack_size = 4

    epochs = 2
    n_steps = 50000

    batch_size = 32
    buffer_size = 1000000

    plot_every_k_steps = 50
    train_frequency = 256
    rolling_avg_over_paddle_hits = 100

    lr = 0.0003

    # dqn specific
    t_learning_starts = batch_size
    target_network_frequency = 10000
    tau = 0.005  # lower -> target network changes more slowly
    gamma = 0.99
    start_e = 1
    end_e = 0.05
    exploration_fraction = 0.1

    # ppo specific
    gae_lambda = 0.95
    policy_clip = 0.2
    ent_coef = 0.1
    vf_coef = 0.5

    # competitive, cooperative, pd, dummy
    reward_structure = "competitive"
    reward_structure = "dummy"

    env = pong.parallel_env(
        render_mode="human",
        max_cycles=n_steps,
        bounce_randomness=True,
        reward_structure=reward_structure,
        ball_direction_randomness=0.5,
        dummy_metrics=dummy_metrics
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)

    n_agents = 2

    agent_names = env.possible_agents

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

    agents = [
        (
            name,
            PPOAgent(
                env.action_space(env.possible_agents[0]).n,
                env.observation_space(env.possible_agents[0]).shape,
                gamma,
                lr,
                gae_lambda,
                policy_clip,
                batch_size,
                epochs,
                memory_size=train_frequency,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
            ),
        )
        for name in agent_names
    ]

    # print("agents", agents)

    obs = env.reset(seed=env_seed)
    loss = 0
    mean_q = 0
    paddle_hits_per_score = [0]
    paddle_hits = 0
    scores = [0 for _ in range(n_agents)]

    all_actions_left = [0]
    all_actions_right = [0]

    for step in range(n_steps):
        actions = {}
        probs = {}
        values = {}

        # take actions
        for name, agent in agents:
            with torch.no_grad():
                action, prob, entropy, value = agent.choose_action(obs)
            actions[name] = action
            probs[name] = prob
            values[name] = value

        # step
        next_obs, rewards, terms, truncs, infos = env.step(actions)

        print("actions", actions, "rewards", rewards, "values", values)

        # measure performance TODO in a separate function
        wall = infos["paddle_0"][1]
        paddle_hit = infos["paddle_0"][2]

        if paddle_hit:
            paddle_hits += 1
        if wall:
            if wall == "left":
                paddle_hits_per_score.append(paddle_hits)
                paddle_hits = 0
                scores[0] += 1
            if wall == "right":
                paddle_hits_per_score.append(paddle_hits)
                paddle_hits = 0
                scores[1] += 1

        for i, (name, agent) in enumerate(agents):
            # add trajectory to buffer
            agent.remember(
                step % train_frequency,
                obs,
                next_obs,
                actions[name],
                rewards[name],
                probs[name],
                values[name],
            )

            # (dummy)
            if i == 0:
                all_actions_left.append(actions[name])
            else:
                all_actions_right.append(actions[name])

            if step > 0 and step % train_frequency == 0:
                # anneal lr TODO rather in train?
                frac = 1.0 - (step - 1.0) / n_steps
                new_lr = frac * lr
                agent.actor.optimizer.param_groups[0]["lr"] = new_lr

                loss, mean_q = agent.learn(
                    # step
                    # t_learning_starts, train_frequency, n_steps
                )
                print("step", step, f"{name} loss", loss)
                print(
                    f"fraction of action {dummy_metrics[0]} left",
                    all_actions_left[-1000:].count(dummy_metrics[0]) / 1000,
                )
                print(
                    f"fraction of action {dummy_metrics[1]} right",
                    all_actions_right[-1000:].count(dummy_metrics[1]) / 1000,
                )
                print(
                    "paddle hits",
                    np.mean(paddle_hits_per_score[-rolling_avg_over_paddle_hits:]),
                )

                writer.add_scalar(f"loss_{name}", loss, step)
                writer.add_scalar(
                    "paddle hits per score",
                    np.mean(paddle_hits_per_score[-rolling_avg_over_paddle_hits:]),
                    step,
                )
                writer.add_scalar(
                    f"fraction of dummy action left",
                    all_actions_left[-1000:].count(dummy_metrics[0]) / 1000,
                    step,
                )
                writer.add_scalar(
                    f"fraction of dummy action right",
                    all_actions_right[-1000:].count(dummy_metrics[1]) / 1000,
                    step,
                )

        obs = next_obs
        env.render()

    env.close()
    writer.close()
