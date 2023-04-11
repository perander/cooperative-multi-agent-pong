import json

from model.ppo_agent import Agent as PPOAgent
from model.model import DQNAgent

def create_agent(algorithm_name, env, device):
    with open("hyperparameters.json", "r") as config_file:
        hyperparameters = json.load(config_file)

    if algorithm_name == "ppo":
        params = hyperparameters["ppo"]
        return PPOAgent(
            env.action_space(env.possible_agents[0]).n,
            env.observation_space(env.possible_agents[0]).shape,
            device=device,
            gamma=params["gamma"],
            lr=params["lr"],
            gae_lambda=params["gae_lambda"],
            policy_clip=params["policy_clip"],
            batch_size=params["batch_size"],
            n_epochs=params["epochs"],
            memory_size=params["train_frequency"],
            ent_coef=params["ent_coef"],
            vf_coef=params["vf_coef"]
        )
    elif algorithm_name == "dqn":
        params = hyperparameters["dqn"]
        return DQNAgent(
            env.observation_space(env.possible_agents[0]).shape,
            env.action_space(env.possible_agents[0]).n,
            env.action_space(env.possible_agents[0]),
            batch_size=params["batch_size"],
            memory_size=params["buffer_size"],
            lr=params["lr"],
            start_e=params["start_e"],
            end_e=params["end_e"],
            exploration_fraction=params["exploration_fraction"],
            gamma=params["gamma"],
            tau=params["tau"],
            device=device
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")