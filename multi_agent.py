from ray.rllib.env import MultiAgentEnv
import gymnasium as gym
import numpy as np
from ray.rllib.evaluation import episode_v2
import pandas as pd
from pympler import asizeof
import tensorflow as tf
import os
from gymnasium import wrappers
from worker2 import Worker
from controller import Controller

# tf.debugging.experimental.enable_dump_debug_info("/Volumes/ssdmac/ray/2", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# tf.debugging.enable_check_numerics()


class MultiAgent(MultiAgentEnv):

    def __init__(self, env_config=None, print_verbosity=1, initial_capital=2e6):
        self.initial_capital = initial_capital
        self.print_verbosity = print_verbosity
        self.data = self.read_data()
        self.agents = {}
        self.episode = self.day = 0
        self.total_portfolio_return = 0
        self.total_portfolio_value = 0
        unique_tickers = self.data["tic"].unique()
        for ticker in unique_tickers:
            capital_allocation = self.initial_capital / len(unique_tickers)
            ticker_df = self.data[self.data["tic"] == ticker]
            self.agents[ticker] = Worker(
                worker_id=ticker, ticker_df=ticker_df, initial_capital=capital_allocation)
        self.controller = Controller(
            agents=self.agents, initial_capital=self.initial_capital)

        self._agent_ids = set(["controller"]+list(self.agents.keys()))

        # self.terminateds = set()
        # self.truncateds = set()
        self.observation_space = gym.spaces.Dict({
            **{tic: agent.observation_space for tic, agent in self.agents.items()},
            "controller": self.controller.observation_space
        })

        self.action_space = gym.spaces.Dict({
            **{tic: agent.action_space for tic, agent in self.agents.items()},
            "controller": self.controller.action_space
        })
        super(MultiAgent, self).__init__()

    def read_data(self):
        df = pd.read_csv("train3.csv").drop(labels="Unnamed: 0", axis=1)
        df.index = df["date"].factorize()[0]
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            df[column] = df[column].astype(np.float32)
        return df

    def reset(self, *, seed=None, options=None):
        # self.terminateds.clear()
        self.episode += 1
        self._reset_to_initial_values()
        # self.terminateds.clear()

        obs = {}
        info = {}
        controller_obs, controller_info = self.controller.reset()
        obs["controller"] = controller_obs
        info["controller"] = controller_info

        for agent_id, agent in self.agents.items():
            agent_obs, agent_info = agent.reset()
            obs[agent_id] = agent_obs
            info[agent_id] = agent_info

        return obs, info

    def _reset_to_initial_values(self):
        self.day = 0
        self.total_portfolio_return = 0
        self.total_portfolio_value = 0
        self.accumulated_worker_rewards_dict = {}

    def step(self, action_dict):
        obs, reward, terminateds, truncateds, info = {}, {}, {}, {}, {}
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}

        fully_done = False

        # Step the controller
        controller_action = action_dict.get("controller")
        if controller_action is not None:
            obs["controller"], _, controller_done, truncateds["controller"], info["controller"] = \
                self.controller.step(
                    controller_action, self.agents, finalize_reward=False)
            new_allocation_limit = info["controller"].get(
                'allocation_limit', {})

        # Step each worker agent
        for agent_id, agent_action in action_dict.items():
            if agent_id == "controller":
                continue  # Skip, already handled

            # Get new resource allocation for this agent from the controller
            agent_new_allocation_limit = new_allocation_limit.get(agent_id, {})
            # Step the agent
            obs[agent_id], reward[agent_id], terminateds[agent_id], truncateds[agent_id], info[agent_id] = self.agents[agent_id].step(
                agent_action, agent_new_allocation_limit)
            # if terminateds[agent_id]:
            #     self.terminateds.add(agent_id)
        # Now finalize the reward for the controller
        _, reward["controller"], controller_done, _, _ = self.controller.step(
            controller_action, self.agents, finalize_reward=True)

        self.day += 1
        # Determine if controller should be done based on agents
        any_worker_done = any(terminateds.values())
        controller_done = controller_done or any_worker_done
        terminateds["controller"] = controller_done

        terminateds["__all__"] = terminateds["__all__"] or any_worker_done
        fully_done = terminateds["__all__"]
        self.total_portfolio_return = info["controller"]["portfolio_return"]
        self.total_portfolio_value = info["controller"]["total_portfolio_value"]

        if fully_done:
            self._handle_done()

        for agent in reward.keys():
            if agent != "controller":
                reward[agent] = 0

        # print(info)

        return obs, reward, terminateds, truncateds, info

    def _handle_done(self):
        if self.episode % self.print_verbosity == 0:
            print("=========HRL is done=============")
            print("HRL is done")
            print(f"day: {self.day}, episode: {self.episode}")
            print(
                f"Total Portfolio Velue (m): {self.total_portfolio_value / 1000000}")
            print(
                f"Total Portfolio Return: {int(self.total_portfolio_return)}%")
            print(f"Controller invalid actions: ")
            print("=================================")
