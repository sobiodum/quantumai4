import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from worker import Worker
import logging
from pympler import asizeof
logger = logging.getLogger(__name__)
import tensorflow as tf
import os
from gymnasium import wrappers

# Create directory if it doesn't exist
# os.makedirs("./tf_debug/manager2", exist_ok=True)

# # Enable the debugger
# # tf.debugging.experimental.enable_dump_debug_info("./tf_debug/manager2")
# tf.debugging.experimental.enable_dump_debug_info("./tf_debug/manager2", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)


class Spec(object):
    def __init__(self, id, max_episode_steps):
        self.id = id
        self.max_episode_steps = max_episode_steps


class Controller(gym.Env):
    def __init__(self):
        super(Controller, self).__init__()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(3)

        self.state = None
        self.step_count = 0
        self.day = 0

    
    def step(self, action, agents):
        done = False
        self.step_count += 1
        reward = 0
        print("controller: agents: ",agents)

   

        self.day += 1

 

        obs = self._get_state()
        info = {}
        truncated = False



  
        return obs, reward, done, truncated, info
    




    def reset(self, *, seed=None, options=None):
        self._reset_to_initial_values()
        obs= self._get_state()
        obs = obs
        info = {}
        return obs, info

    def _reset_to_initial_values(self):
        self.step_count = 0
        self.day = 0
        pass

    
    def _get_state(self):
        state = np.array([0.0, 0.0], dtype=np.float32)
        return state
    





