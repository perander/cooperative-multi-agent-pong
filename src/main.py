import numpy as np
import torch
from environment import pong
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.utils.tensorboard import SummaryWriter
from model.model import DQNAgent
from model.ppo_agent import Agent as PPOAgent
from utils.agent_factory import create_agent

def get_dummy_rewards(rewards, actions, dummy_metrics):
    if actions['paddle_0'] == dummy_metrics[0] and actions['paddle_1'] == dummy_metrics[1]:
        rewards['paddle_0'] = 0.5
        rewards['paddle_1'] = 0.5
    
    elif actions['paddle_0'] == dummy_metrics[0]:
        rewards['paddle_0'] = -1
        rewards['paddle_1'] = 1
    
    elif actions['paddle_1'] == dummy_metrics[1]:
        rewards['paddle_1'] = -1
        rewards['paddle_0'] = 1
    
    else:
        rewards['paddle_0'] = -0.1
        rewards['paddle_1'] = -0.1
    
    return rewards

if __name__ == "__main__":
    # seeds
    seed = 124
    torch_deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    env_seed = seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alg = "ppo"
    alg = "dqn"

    # preprocessing
    frame_size = (84, 84)
    stack_size = 4

    # training hyperparameters
    epochs = 2
    episodes = 20000
    n_steps = 300
    batch_size = 32
    buffer_size = 20000

    # dqn specific
    train_frequency = 4
    t_learning_starts = 1000

    # ppo specific
    if alg == "ppo":
        train_frequency = 256  # buffer size
        t_learning_starts = 0

    # plotting, performance measuring
    writer = SummaryWriter("src/runs/ppo")
    writer.close()
    plot_every_k_steps = 50
    rolling_avg_over_paddle_hits = 100

    # rewards (competitive, cooperative, pd, dummy)
    reward_structure = "competitive"
    reward_structure = "dummy"
    dummy_metrics = [1, 2]  # action numbers

    env = pong.parallel_env(
        render_mode="human",
        max_cycles=n_steps,
        bounce_randomness=True,
        reward_structure=reward_structure,
        ball_direction_randomness=0.5,
        dummy_metrics=dummy_metrics
    )

    # preprocess
    env = color_reduction_v0(env)
    env = resize_v1(env, frame_size[0], frame_size[1])
    env = frame_stack_v1(env, stack_size=stack_size)

    # initialize agents
    n_agents = 2
    agent_names = env.possible_agents

    agents = [
        (
            name,
            create_agent(alg, env, device),
        )
        for name in agent_names
    ]

    # initialize training
    obs = env.reset(seed=env_seed)
    loss = 0
    mean_q = 0
    paddle_hits_per_score = [0]
    paddle_hits = 0
    scores = [0 for _ in range(n_agents)]
    cum_rewards = [0 for _ in range(n_agents)]

    all_actions_left = [0]
    all_actions_right = [0]

    total_steps = 0

    for episode in range(episodes):
        print("episode", episode)
        obs = env.reset(seed=env_seed)

        # training
        for step in range(n_steps):
            total_steps += 1
            # print(total_steps)
            actions = {}
            probs = {}
            values = {}

            # take actions
            for name, agent in agents:
                with torch.no_grad():
                    if alg == "dqn":
                        action = agent.choose_action(obs)
                    elif alg == "ppo":
                        action, prob, entropy, value = agent.choose_action(obs)
                        probs[name] = prob
                        values[name] = value
                        
                actions[name] = action

            # step
            next_obs, rewards, terms, truncs, infos = env.step(actions)

            # dummy rewards
            rewards = get_dummy_rewards(rewards, actions, dummy_metrics)

            print("actions", actions, "rewards", rewards, "truncs", truncs)
            cum_rewards[0] += rewards[agent_names[0]]
            cum_rewards[1] += rewards[agent_names[1]]

            # measure performance (TODO in a separate function)
            # print("infos", infos)
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
                
                if alg == "dqn":
                    agent.remember(
                        obs,
                        next_obs,
                        actions[name],
                        rewards[name]
                    )

                elif alg == "ppo":
                    agent.remember(
                        step % train_frequency,
                        obs,
                        next_obs,
                        actions[name],
                        rewards[name],
                        probs[name],
                        values[name],
                    )

                # performance measuring for dummy
                if i == 0:
                    all_actions_left.append(actions[name])
                else:
                    all_actions_right.append(actions[name])


                if total_steps > t_learning_starts and total_steps % train_frequency == 0:
                    # learn
                    loss, mean_q = agent.learn(total_steps, n_steps * 0.5 * episodes)

            obs = next_obs
            env.render()

            if np.any(list(terms.values())):
                # print("end")
                break

        # plotting
        writer.add_scalar(f"episode length", step, episode)
        writer.add_scalar(f"loss_{name}", loss, episode)
        writer.add_scalar(
            "paddle hits per score",
            np.mean(paddle_hits_per_score[-rolling_avg_over_paddle_hits:]),
            episode,
        )
        writer.add_scalar("cumulative rewards left", cum_rewards[0], episode)
        writer.add_scalar("cumulative rewards right", cum_rewards[1], episode)

        writer.add_scalar(
            f"fraction of dummy action left",
            all_actions_left[-1000:].count(dummy_metrics[0]) / 1000,
            episode,
        )
        writer.add_scalar(
            f"fraction of dummy action right",
            all_actions_right[-1000:].count(dummy_metrics[1]) / 1000,
            episode,
        )

        print(
            "paddle hits",
            np.mean(paddle_hits_per_score[-rolling_avg_over_paddle_hits:]),
        )
        print("total step", total_steps, "episode length", step, "loss", loss)
        print("memory size", len(agent.memory))



    env.close()
    writer.close()
