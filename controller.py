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
    def __init__(self, agents, intervention_interval=10):
        super(Controller, self).__init__()

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.agents = agents
        self.current_allocations = {agent_id: 1.0 / len(agents) for agent_id in self.agents.keys()}

        self.intervention_interval = intervention_interval
        # self.observation_space = gym.spaces.Dict({
        #     "controller_state": gym.spaces.Dict({
        #         "annual_return": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #         # Add other controller-specific metrics here if needed
        #     }),
        #     **{
        #         agent_id: gym.spaces.Dict({
        #             "current_cash": gym.spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
        #             "total_trades": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #         }) for agent_id in agents.keys()
        #     }
        # })


      
        # self.action_space = gym.spaces.Dict({
        #     worker_id: gym.spaces.Dict({
        #         "action_type": gym.spaces.Discrete(2), #0: nothing, 1: increase, 2: decrease
        #         "amount": gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        #     }) for worker_id in self.agents.keys()
        # })
        self.action_space = gym.spaces.Discrete(2)


        self.step_count = 0
        self.day = 0
        self.state = self._get_state()
        self.invalid_actions_count = 0

    
    def step(self, action_dict, agents):
        done = False
        self.step_count += 1
   
        agent_obs = {}
        agent_rewards = {}
        agent_dones = {}
        agent_infos = {}


        self.day += 1

        # reward = self.calculate_reward(action_dict)
        reward = 0
        obs = self.state = self._get_state()
        # new_resource_allocations = self.calculate_new_allocations(action_dict)
        # info = {'new_resource_allocations': new_resource_allocations}
        info = {}
        
        truncated = False
        done = any(agent.done for agent in agents.values())
        if done:
            print("Controller done")
  
        return obs, reward, done, truncated, info
    
    def calculate_new_allocations(self, action):
        # Translate the action into new resource allocations.
        # This is where your logic for interpreting the action goes.
        new_allocations = {}
        for agent_id in self.agents.keys():
            if action[agent_id]['action_type'] == 1:  # Increase allocation
                new_allocations[agent_id] = self.current_allocations[agent_id] + action[agent_id]['amount']
            elif action[agent_id]['action_type'] == 2:  # Decrease allocation
                new_allocations[agent_id] = self.current_allocations[agent_id] - action[agent_id]['amount']
            else:
                new_allocations[agent_id] = self.current_allocations[agent_id]  # No change
        return new_allocations
    
    def calculate_reward(self,action_dict):
        invalid_action_penalty = 0
        action_values = 0
        for action_type, action in action_dict.items():
            if action_type is not 0:
                action_values += action["amount"][0]
        if action_values > 1:
            invalid_action_penalty = -1
        return np.float32(0)

    def calculate_annual_return(self):
        annual_return = 0
        return np.array([annual_return], dtype=np.float32) 



    def reset(self, *, seed=None, options=None):
        self._reset_to_initial_values()
        obs= self._get_state()
        obs = obs
        info = {}
        return obs, info
    



    def _reset_to_initial_values(self):
        self.step_count = 0
        self.day = 0
        self.invalid_actions = 0
        pass

    
    def _get_state(self):
        # state = {
        #     "controller_state": {
        #         "annual_return": self.calculate_annual_return(),  # Replace with actual logic
        #         # Add other controller-specific metrics here
        #     }
        # }
        # for agent_id, agent in self.agents.items():
        #     agent_state = agent._get_state()
        #     state[agent_id] = {
        #         "current_cash": agent_state["current_cash"],
        #         "total_trades": agent_state["total_trades"],
        #     }
        state = np.zeros(2, dtype=np.float32)
        return state
        





