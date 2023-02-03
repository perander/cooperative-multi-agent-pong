import gymnasium
import numpy as np
import pygame
from gymnasium.utils import EzPickle, seeding

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, ParallelEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .ball import Ball
from .paddle import Paddle


def get_action_space(num_agents):
    return [gymnasium.spaces.Discrete(3) for _ in range(num_agents)]


def get_observation_space(
    screen_width, screen_height, kernel_window_length, num_agents
):
    original_color_shape = (
        int(screen_height * 2 / kernel_window_length),
        int(screen_width * 2 / kernel_window_length),
        3,
    )

    observation_space = [
        gymnasium.spaces.Box(
            low=0, high=255, shape=(original_color_shape), dtype=np.uint8
        )
        for _ in range(num_agents)
    ]

    return observation_space


def get_state_space(height, width):
    return gymnasium.spaces.Box(
        low=0, high=255, shape=((height, width, 3)), dtype=np.uint8
    )


def get_rewards(wall, reward_structure):
    rewards = {}
    if reward_structure == "competitive":
        if wall == "left":
            print("hit left wall")
            rewards["paddle_1"] = 1
            rewards["paddle_0"] = -1
        elif wall == "right":
            print("hit right wall")
            rewards["paddle_0"] = 1
            rewards["paddle_1"] = -1
    elif reward_structure == "cooperative":
        if wall == "left":
            print("hit left wall")
            rewards["paddle_1"] = -1
            rewards["paddle_0"] = -1
        elif wall == "right":
            print("hit right wall")
            rewards["paddle_0"] = -1
            rewards["paddle_1"] = -1
    elif reward_structure == "pd":
        if wall == "left":
            print("hit left wall")
            rewards["paddle_1"] = -2
            rewards["paddle_0"] = 1
        elif wall == "right":
            print("hit right wall")
            rewards["paddle_0"] = -2
            rewards["paddle_1"] = 1
    else:
        raise Exception(
            f"Reward structure should be one of the following: competitive, cooperative or pd. You gave {reward_structure}"
        )

    return rewards


FPS = 15


def get_flat_shape(width, height, kernel_window_length=2):
    return int(width * height / (kernel_window_length * kernel_window_length))


class Pong(ParallelEnv):
    def __init__(
        self,
        randomizer,
        ball_speed=9,
        num_agents=2,
        paddle_speeds=[12, 12],
        max_cycles=300,
        bounce_randomness=False,
        render_mode=None,
        render_ratio=2,
        kernel_window_length=2,
        reward_structure="competitive",
        ball_direction_randomness=1,
    ):
        super().__init__()

        pygame.init()

        self.ball_speed = ball_speed
        self.bounce_randomness = bounce_randomness
        self.paddle_speeds = paddle_speeds
        self.ball_direction_randomness = ball_direction_randomness

        self.render_ratio = render_ratio
        self.kernel_window_length = kernel_window_length

        self.render_mode = render_mode
        self.renderOn = False

        self.max_cycles = max_cycles
        self.randomizer = randomizer

        self.screen_width, self.screen_height = 960 // render_ratio, 560 // render_ratio
        self.paddle_dims = (20 // render_ratio, 80 // render_ratio)
        self.ball_dims = (20 // render_ratio, 20 // render_ratio)

        self.speed = [ball_speed, *self.paddle_speeds]

        self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.area = self.screen.get_rect()

        self.action_space = get_action_space(num_agents)

        self.observation_space = get_observation_space(
            self.screen_width, self.screen_height, kernel_window_length, num_agents
        )

        self.state_space = get_state_space(self.screen_height, self.screen_width)

        # self.p0, self.p1 = [
        #     Paddle(self.paddle_dims, paddle_speeds[i])
        #     for i in range(num_agents)
        # ]

        self.p0 = Paddle(self.paddle_dims, paddle_speeds[0], "left")
        self.p1 = Paddle(self.paddle_dims, paddle_speeds[1], "right")

        self.agents = ["paddle_0", "paddle_1"]

        self.ball = Ball(
            self.randomizer,
            self.ball_dims,
            self.ball_speed,
            self.bounce_randomness,
            self.ball_direction_randomness,
        )

        self.reward_structure = reward_structure

        self.reinit()

    def step(self, actions):
        pass

    def reinit(self):
        self.rewards = dict(zip(self.agents, [0.0] * len(self.agents)))
        self.terminations = dict(zip(self.agents, [False] * len(self.agents)))
        self.truncations = dict(zip(self.agents, [False] * len(self.agents)))
        self.infos = dict(zip(self.agents, [{}] * len(self.agents)))
        self.score = 0

        self.score_0 = 0
        self.score_1 = 0

    def reset(self, seed=None, options=None):
        self.ball.reset(self.area.center)

        self.p0.reset(self.area.midleft, self.paddle_speeds[0])
        self.p1.reset(self.area.midright, self.paddle_speeds[1])

        self.terminate = False
        self.truncate = False

        self.num_frames = 0

        self.reinit()
        self.draw()

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False

    def enable_render(self):
        self.screen = pygame.display.set_mode(self.screen.get_size())
        self.renderOn = True
        self.draw()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if not self.renderOn and self.render_mode == "human":
            self.enable_render()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        if self.render_mode == "human":
            pygame.display.flip()

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def observe(self):
        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        observation = np.rot90(
            observation, k=3
        )  # now the obs is laid out as H, W as rows and cols
        observation = np.fliplr(observation)  # laid out in the correct order
        return observation

    def state(self):
        """Returns an observation of the global environment."""
        state = pygame.surfarray.pixels3d(self.screen).copy()
        state = np.rot90(state, k=3)
        state = np.fliplr(state)
        return state

    def draw(self):
        pygame.draw.rect(self.screen, (0, 0, 0), self.area)
        self.p0.draw(self.screen)
        self.p1.draw(self.screen)
        self.ball.draw(self.screen)

    def step(self, action, agent):
        self.rewards = {a: 0 for a in self.agents}

        if agent == self.agents[0]:
            self.p0.update(self.area, action)
            # print("rewards after agent ", agent, ":", self.rewards)
        elif agent == self.agents[1]:
            self.p1.update(self.area, action)
            # print("rewards after agent ", agent, ":", self.rewards)

            # do everything else after the last agent has moved (agent selector takes care of the order)

            # TODO self.terminate used for two purposes?
            if not self.terminate:
                terminate, wall, paddle_hit = self.ball.update2(
                    self.area, self.p0, self.p1
                )
                self.terminate = terminate

                if self.terminate:
                    if wall:
                        self.rewards = get_rewards(wall, self.reward_structure)
                        if wall == "left":
                            self.score_1 += 1
                        elif wall == "right":
                            self.score_0 += 1

                    self.ball.reset(self.area.center)
                    self.terminate = False

                if not self.terminate:
                    self.num_frames += 1
                    self.truncate = self.num_frames >= self.max_cycles

                for ag, score in zip(self.agents, [self.score_0, self.score_1]):
                    self.terminations[ag] = self.terminate
                    self.truncations[ag] = self.truncate
                    self.infos[ag] = score, wall, paddle_hit

        if self.renderOn:
            pygame.event.pump()
        self.draw()


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "pong",
        "is_parallelizable": True,
        "render_fps": FPS,
        "has_manual_policy": True,
    }

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()

        self.render_mode = self.env.render_mode
        self.agents = self.env.agents[:]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.action_spaces = dict(zip(self.agents, self.env.action_space))
        self.observation_spaces = dict(zip(self.agents, self.env.observation_space))
        self.state_space = self.env.state_space

        self.observations = {}
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

        self.score = self.env.score
        self.score_0 = self.env.score
        self.score_1 = self.env.score

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)
        self.env = Pong(self.randomizer, **self._kwargs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = self.env.rewards
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

    def observe(self, agent):
        obs = self.env.observe()
        return obs

    def state(self):
        state = self.env.state()
        return state

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(
                "Action for agent {} must be in Discrete({})."
                "It is currently {}".format(agent, self.action_spaces[agent].n, action)
            )

        self.env.step(action, agent)

        self.agent_selection = self._agent_selector.next()

        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos

        self.score = self.env.score
        self.score_0 = self.env.score_0
        self.score_1 = self.env.score_1

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()


# This was originally created, in full, by Ananth Hari in a different repo, and was added in by J K Terry (which is why they're shown as the creator in the git history)
