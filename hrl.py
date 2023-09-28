from ray.rllib.env import MultiAgentEnv

from manager import Manager
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.evaluation import episode_v2
import pandas as pd
from pympler import asizeof
import tensorflow as tf
import os
from gymnasium import wrappers

# Create directory if it doesn't exist
# os.makedirs("./tf_debug/hrl2", exist_ok=True)

# # Enable the debugger
# # tf.debugging.experimental.enable_dump_debug_info("./tf_debug/hrl2")
# tf.debugging.experimental.enable_dump_debug_info("./tf_debug/hrl2", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

#! TODO
#? 1) gather accumulated rewards from worker --> solely on trading revenue


class HRL(MultiAgentEnv):
 
    def __init__(self, env_config=None, print_verbosity=1, initial_capital=2e6):
        super(HRL, self).__init__()
        self.initial_capital = initial_capital
        self.data = self.read_data()
        # self.data = pd.read_pickle("/Users/floriankockler/Documents/GitHub.nosync/quantumai3/train1.pkl")
        # self.manager = Manager(env_config=env_config["manager_config"], initial_capital=initial_capital)
        self.manager = Manager(data=self.data, initial_capital=initial_capital)
        self._agent_ids = set(["manager"] + list(self.manager.workers.keys()))
        self.workers = self.manager.workers
        self.print_verbosity = print_verbosity
        self.episode = self.day = 0
        self.total_capital = initial_capital
        # Rewards
        self.target_annualized_return = 0.15  # Example target annualized return of 15%
        self.target_sharpe_ratio = 1.0   


        self.observation_space = gym.spaces.Dict({
            **{"manager": self.manager.observation_space},
            **{tic: worker.observation_space for tic, worker in self.workers.items()}
        })



        self.action_space = gym.spaces.Dict({
            **{"manager": self.manager.action_space},
            **{tic: worker.action_space for tic, worker in self.workers.items()}
        })

    def read_data(self):
        df = pd.read_pickle("./train1.pkl")
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for column in numerical_columns:
            df[column] = df[column].astype(np.float32)
        return df


    def reset(self, *, seed=None, options=None):
        manager_obs, manager_info = self.manager.reset()  # This will now only contain manager-specific states
        self.episode += 1
        self._reset_to_initial_values()

        obs = {"manager": manager_obs}
        info = {"manager": manager_info}
        
        # Loop over all workers and gather their observations
        for worker_id, worker in self.workers.items():
            worker_obs, worker_info = worker.reset()
            obs[worker_id] = worker_obs  # This should be a dictionary, as per your worker reset method
            info[worker_id] = worker_info

        # print("hrl",obs)
      
        return obs, info
    
    def check_nan_in_obs(self,obs, path=""):
            if isinstance(obs, dict):
                for key, value in obs.items():
                    self.check_nan_in_obs(value, path + f".{key}")
            elif isinstance(obs, (list, tuple, np.ndarray)):
                for idx, value in enumerate(obs):
                    self.check_nan_in_obs(value, path + f"[{idx}]")
            else:
                if np.isnan(obs).any():
                    print(f"NaN value in hrl obs reset at step: {self.day}, path: {path}")
    
    def _reset_to_initial_values(self):
        self.day = 0
        self.accumulated_worker_rewards_dict = {}
        self.manager_rewards_array = []



    def step(self, action_dict):
        # print("HRL step being - self.day: ",self.day)

        fully_done = False

        
        obs = {"manager": None}  # We will update this later
        reward = {"manager": None}  # We will update this later
        terminateds = {"__all__": False}
        truncateds = {"__all__": False}
        info = {"manager": None}  # We will update this later

        worker_observations = []
        worker_rewards = []
        worker_dones = []
        worker_truncateds = []
       
    
        # Loop over all workers, apply actions, and gather results
        for worker_id, worker in self.workers.items():
            worker_action = action_dict[worker_id]
            worker_obs, worker_reward, worker_done, worker_truncated, worker_info = worker.step(worker_action)
            
            # worker_observations.append(worker_obs)
            worker_rewards.append(worker_reward)
            worker_dones.append(worker_done)
            worker_truncateds.append(worker_truncated)
            
            obs[worker_id] = worker_obs
            reward[worker_id] = worker_reward
            terminateds[worker_id] = worker_done
            truncateds[worker_id] = worker_truncated
            info[worker_id] = worker_info

  
        manager_action = action_dict["manager"]
        manager_obs, manager_reward, manager_done, manager_truncated, manager_info = self.manager.step(manager_action, worker_observations, worker_rewards, worker_dones, worker_truncateds)

     
        terminateds["__all__"] = terminateds["__all__"] or worker_done

        fully_done = terminateds["__all__"]

        if fully_done:
            self._handle_done()
        obs["manager"] = manager_obs
        reward["manager"] = manager_reward
        terminateds["manager"] = manager_done
        truncateds["manager"] = manager_truncated
        info["manager"] = manager_info


        self.day +=1
        print(reward)

        return obs, reward, terminateds, truncateds, info



    
    def find_float64(self, obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                self.find_float64(value, path + f".{key}")
        elif isinstance(obj, (list, tuple, np.ndarray)):
            for idx, value in enumerate(obj):
                self.find_float64(value, path + f"[{idx}]")
        elif isinstance(obj, np.float64):
            print(f"Found float64 at {path}")
        else:
            # This branch will handle single values and other data types
            if isinstance(obj, float):
                print(f"Found float (Python's native float type, which is typically float64) at {path}")
            elif isinstance(obj, np.float64):
                print(f"Found float32 at {path}")
        
    
    
    def _handle_done(self):
        if self.episode % self.print_verbosity == 0:
            print("=========HRL is done=============")
            print("HRL is done")
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"Total Cash Transfers: {self.manager.total_cash_transfers}")           
            print(f"total_portfolio_trades: {self.manager._calculate_total_portfolio_trades()[0]}")           
            print(f"Beginn_Portfolio_Value: {round(self.initial_capital)}")           
            print(f"End_Portfolio_Value: {self.manager._calculate_total_portfolio_value()[0]}")
            print(f"Annual Return: {self.manager._calculate_annualized_return()*100:0.2f} %")
            for worker_id, worker in self.workers.items():
                print(f"Worker ID: {worker_id} Current Stock Exposure: {round(worker._get_state().get('current_stock_exposure')[0])}")
            print(f"Free Cash: {self.manager._calculate_free_cash()}")
            print(f"Manager Common Pool: {self.manager.common_pool}")
            print("=================================")


