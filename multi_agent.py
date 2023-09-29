from ray.rllib.env import MultiAgentEnv
import gymnasium as gym
import numpy as np
from ray.rllib.evaluation import episode_v2
import pandas as pd
from pympler import asizeof
import tensorflow as tf
import os
from gymnasium import wrappers
from worker_standlone import WorkerStandAlone
from controller import Controller


class MultiAgent(MultiAgentEnv):
 
    def __init__(self, env_config=None, print_verbosity=1, initial_capital=2e6):
        self.initial_capital = initial_capital
        self.print_verbosity = print_verbosity
        self.data = self.read_data()
        self.controller = Controller()
        self.agents = {}
        self.episode = self.day = 0
        unique_tickers = self.data["tic"].unique()
        for ticker in unique_tickers:
            ticker_df = self.data[self.data["tic"]==ticker]
            self.agents[ticker] = WorkerStandAlone(worker_id=ticker, ticker_df=ticker_df)

        self._agent_ids = set(["controller"]+list(self.agents.keys()))

    

        self.terminateds = set()
        self.truncateds = set()
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
        df = pd.read_pickle("./train1.pkl")
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            df[column] = df[column].astype(np.float32)
        return df


    def reset(self, *, seed=None, options=None):
        self.episode += 1
        self._reset_to_initial_values()
        self.terminateds.clear()

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
        self.accumulated_worker_rewards_dict = {}




    def step(self, action_dict):
        obs, reward, terminateds, truncateds, info = {}, {}, {}, {}, {}

        fully_done = False


         # Handle controller separately
        controller_action = action_dict.get("controller", None)
        if controller_action is not None:
            controller_obs, controller_reward, _, controller_truncated, controller_info= self.controller.step(controller_action, self.agents)
            obs["controller"] = controller_obs
            reward["controller"] = controller_reward
            info["controller"] = controller_info
            truncateds["controller"] = controller_truncated

        # Handle other agents
        for agent_id, action in action_dict.items():
            if agent_id == "controller":
                continue  # Skip, already handled
            agent = self.agents[agent_id]
            agent_obs, agent_reward, agent_done, truncated, agent_info = agent.step(action)
            obs[agent_id] = agent_obs
            reward[agent_id] = agent_reward
            terminateds[agent_id] = agent_done
            truncateds[agent_id] = truncated  # Update based on your condition
            info[agent_id] = agent_info

            if agent_done:
                self.terminateds.add(agent_id)

        # Determine if controller should be done based on agents
        controller_done = len(self.terminateds) == len(self.agents)
        terminateds["controller"] = controller_done
        truncateds["__all__"] = all(truncateds.values())
        
        fully_done = terminateds["__all__"] = controller_done


        if fully_done:
            self._handle_done()
        self.day +=1
        return obs, reward, terminateds, truncateds, info
    
    

    def _handle_done(self):
        if self.episode % self.print_verbosity == 0:
            print("=========HRL is done=============")
            print("HRL is done")
            print(f"day: {self.day}, episode: {self.episode}")
            print("=================================")


