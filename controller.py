from gymnasium import wrappers
import os
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from worker3 import Worker
import logging
from pympler import asizeof
from scipy.special import softmax
from collections import OrderedDict
logger = logging.getLogger(__name__)

# Create directory if it doesn't exist
# os.makedirs("./tf_debug/manager2", exist_ok=True)

# # Enable the debugger
# # tf.debugging.experimental.enable_dump_debug_info("./tf_debug/manager2")
# tf.debugging.experimental.enable_dump_debug_info("/Volumes/ssdmac/ray/2", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# tf.debugging.enable_check_numerics()


class Spec(object):
    def __init__(self, id, max_episode_steps):
        self.id = id
        self.max_episode_steps = max_episode_steps


class Controller(gym.Env):
    def __init__(self, agents, initial_capital, intervention_interval=10):
        super(Controller, self).__init__()

        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.agents = agents
        self.initial_capital = initial_capital
        self.current_allocations = {
            agent_id: 1.0 / len(agents) for agent_id in self.agents.keys()}

        self.intervention_interval = intervention_interval
        self.observation_space = gym.spaces.Dict({
            "controller_state": gym.spaces.Dict({
                "annual_return": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                # "total_portfolio_value": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                # "total_free_cash": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                # Add other controller-specific metrics here if needed
            }),
            **{
                agent_id: gym.spaces.Dict({
                    "current_cash": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                    "total_trades": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                }) for agent_id in agents.keys()
            }
        })

        self.action_space = gym.spaces.Dict({
            worker_id: gym.spaces.Dict({
                "allocation_limit": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            }) for worker_id in self.agents.keys()
        })

        self.step_count = 0
        self.day = 0
        self.state = self._get_state()
        self.invalid_actions_count = 0

    def step(self, action_dict, agents, finalize_reward=True):
        done = False
        self.step_count += 1

        agent_obs = {}
        agent_rewards = {}
        agent_dones = {}
        agent_infos = {}
        action_dict_softmax = self.apply_softmax(action_dict)
        new_allocation_limits = self.calculate_allocation_limit(
            action_dict_softmax)

        self.day += 1
        obs = self.state = self._get_state()
        # new_resource_allocations = self.calculate_new_allocations(action_dict)
        # info = {'new_resource_allocations': new_resource_allocations}
        info = {"total_portfolio_value": self.calculate_total_pf_value(),
                "portfolio_return": self.calculate_portfolio_return()[0],
                "allocation_limit": new_allocation_limits,
                "sum_allocation_breaches": self.sum_allocation_breaches(),
                }

        truncated = False
        done = any(agent.done for agent in agents.values())

        if finalize_reward:
            reward = self.calculate_reward(action_dict_softmax)
            return obs, reward, done, truncated, info
        else:
            return obs, None, done, truncated, info

    def calculate_allocation_limit(self, action_dict_softmax):
        factor = self.calculate_total_pf_value()
        new_dict = {}
        for agent_id, action in action_dict_softmax.items():
            new_dict[agent_id] = np.float32(
                action["allocation_limit"][0] * factor)
        return new_dict

    def calculate_annual_return(self):
        annual_return = 0
        return np.array([annual_return], dtype=np.float32)

    def apply_softmax(self, action_dict):
        allocation_limits = [agent['allocation_limit'][0]
                             for agent in action_dict.values()]
        new_allocation_limits = softmax(allocation_limits)
        for i, (key, value_dict) in enumerate(action_dict.items()):
            value_dict['allocation_limit'] = np.array(
                [new_allocation_limits[i]])
        return action_dict

    def calculate_portfolio_return(self):

        total_assets = sum(agent._calculate_assets()
                           for _, agent in self.agents.items())
        portfolio_return = np.float32(
            (total_assets / self.initial_capital-1)*100)
        return np.array([portfolio_return], dtype=np.float32)

    def calculate_total_pf_value(self):
        total_assets = sum(agent._calculate_assets()
                           for _, agent in self.agents.items())
        return np.float32(total_assets)

    def sum_allocation_breaches(self):
        sum_allocation_breaches = sum(agent.allocation_limit_breach_count
                                      for _, agent in self.agents.items())
        return np.float32(sum_allocation_breaches)

    def calculate_reward(self, action_dict_softmax):
        # tallocation_limit_sum = sum(
        #     agent["allocation_limit"][0] for agent in action_dict_softmax.values())
        reward = np.float32(0)
        return reward

    def reset(self, *, seed=None, options=None):
        self._reset_to_initial_values()
        obs = self._get_state()
        obs = obs
        info = {}
        return obs, info

    def _reset_to_initial_values(self):
        self.step_count = 0
        self.day = 0
        self.invalid_actions = 0
        pass

    def _get_state(self):
        state = {
            "controller_state": {
                "annual_return": self.calculate_portfolio_return(),  # Replace with actual logic
                # Add other controller-specific metrics here
            }
        }
        for agent_id, agent in self.agents.items():
            agent_state = agent._get_state()
            state[agent_id] = {
                "current_cash": agent_state["current_cash"],
                "total_trades": agent_state["total_trades"],
            }
        # state = np.zeros(2, dtype=np.float32)
        return state
