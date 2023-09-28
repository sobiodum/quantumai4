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


class Manager(gym.Env):
    def __init__(self, initial_capital=2e6,data=None,env_config=None):
        super(Manager, self).__init__()
        self.workers = {}
        self.total_capital = initial_capital

        self.common_pool = np.float32(initial_capital)
        # print("common pool after init: ",self.common_pool)
        if data is not None:
            df = data
            unique_tickers = df['tic'].unique()
            for ticker in unique_tickers:
                # capital_per_worker = self.total_capital / len(unique_tickers)
                capital_per_worker = np.float32(0)
                worker_allocation = np.float32(capital_per_worker / self.total_capital)
                ticker_df = df[df['tic'] == ticker]
                self.workers[ticker] = Worker(data=ticker_df, worker_id=ticker, initial_capital=capital_per_worker, worker_allocation_pct=worker_allocation, manager=self)
     
        self.unique_tickers = df['tic'].unique()
        self.day = 0
    
        self.step_count = 0
        self.decisions = {}
        self.total_cash_transfers = 0

         # Rewards
        self.worker_rewards_dict = {}
 
        self.manager_rewards_array = []
        # Trades
        self.worker_total_trades_dict = {}
        #Exposure
        self.worker_expsoure_dict = {}
        #PnL
        self.worker_accumulated_pnl_dict = {}
        # Sharpe Ratio
        self.worker_sharpe_ratio_dict = {}
        # Rewards
        self.return_worker_return_memory_dict = {}
        # Return per Volatility
        self.worker_return_per_volatility = {}
        # Free Cash
        self.worker_free_cash = {}
        # Directives
        self.directives = {
            "capital_allocation": None,
            # "risk_limit": None,
            # "position_limit": None
        }

        # self.observation_space = spaces.Dict({
        #     'capital_allocation': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'risk_limit': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'position_limit': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'total_free_cash': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'portfolio_sharpe_ratio': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'total_portfolio_trades': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        #     'total_portfolio_value': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),

        # })
        #!Potentially adding states like total trades, free cash from earlier states
        self.manager_observation_space = spaces.Dict({
            'capital_allocation': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            # ... (other state variables for the manager)
        })


        self.action_space = spaces.Dict({
            ticker: self.worker_directive_space() for ticker in unique_tickers
        })
    

        self.observation_space = spaces.Dict({
            ticker: self.worker_observation_space() for ticker in unique_tickers
        })
    def worker_observation_space(self):
        return spaces.Dict({
            "portfolio_value": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "return_mean": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "return_std": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "Sharpe_Ratio": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),

        })


    def worker_directive_space(self):
        return spaces.Dict({
            "capital_allocation": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # "risk_limit": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            # "position_limit": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
    
    def step(self, action, worker_observations, worker_rewards, worker_dones, worker_truncateds):

        self.step_count += 1

        # Initializing dictionaries to store various metrics
        metrics = [
            "total_trades", "current_stock_exposure", "pnl", "sharpe_ratios",
            "return_memory", "current_cash"
        ]
        worker_metrics = {metric: {} for metric in metrics}

        for worker_id, worker in self.workers.items():
            worker_state = worker._get_state()
            worker_sharpe_ratios = worker._get_sharpe_ratio()
            worker_return_memory = worker._get_return_memory()
            
            worker_metrics["total_trades"][worker_id] = worker_state["total_trades"]
            worker_metrics["current_stock_exposure"][worker_id] = worker_state["current_stock_exposure"]
            worker_metrics["pnl"][worker_id] = worker_state["pnl"]
            worker_metrics["sharpe_ratios"][worker_id] = worker_sharpe_ratios
            worker_metrics["return_memory"][worker_id] = worker_return_memory
    
            worker_metrics["current_cash"][worker_id] = worker.current_cash

        # Updating the respective class attributes with the calculated metrics
        self.worker_total_trades_dict = worker_metrics["total_trades"]
        self.worker_rewards_dict = worker_metrics["current_stock_exposure"]
        self.worker_expsoure_dict = worker_metrics["current_stock_exposure"]
        self.worker_accumulated_pnl_dict = worker_metrics["pnl"]
        self.worker_sharpe_ratio_dict = worker_metrics["sharpe_ratios"]
        self.return_worker_return_memory_dict = worker_metrics["return_memory"]
   
        self.worker_free_cash = worker_metrics["current_cash"]
        
        #! NEW action space to give directives --> das wirft aktuell noch error, muss action checken
        # if True:
        #     for worker_id, normalized_directive in normalized_directives_dict.items():               
        #         self.workers[worker_id].set_directives({'capital_allocation': normalized_directive})

        directives_dict = {worker_id: directives['capital_allocation'] for worker_id, directives in action.items()}
        normalized_directives_dict = self._normalize_directives(directives_dict)

        for worker_id, normalized_directive in normalized_directives_dict.items():
            # Determine the amount of capital to allocate (it could be a percentage of the common pool)
            capital_to_allocate = np.float32(normalized_directive * self.common_pool)
            
            # Set directives for workers
            self.workers[worker_id].set_directives({'capital_allocation': capital_to_allocate})

        all_workers_done = all(worker_dones)
        all_workers_truncated = any(worker_truncateds) 

        # Updating the manager's state based on worker observations or other criteria
        self.day += 1
        reward = self._calculate_reward()
 

        obs = self._get_state()
        info = {}
        print("manager action", action)


        return obs, reward, all_workers_done, all_workers_truncated, info
    
    def _normalize_directives(self, directives):
        total = sum(directives.values())
        total = max(total, 1e-10)

        if total == 0:
            raise ValueError("The total sum of directives cannot be zero")
        
        normalized_directives = {}
        for key in directives:
            normalized_directives[key] = directives[key] / total
        
        return normalized_directives
    
    def check_for_nan(self,obs, path=""):
        if isinstance(obs, dict):
            for key, value in obs.items():
                self.check_for_nan(value, path + f".{key}")
        elif isinstance(obs, (list, tuple, np.ndarray)):
            for idx, value in enumerate(obs):
                self.check_for_nan(value, path + f"[{idx}]")
        else:
            if np.isnan(obs).any():
                print(f"NaN value in manager obs reset at step: {self.day}, path: {path}")

    def update_common_pool(self, worker_id, change_in_value):
        self.common_pool -= np.float32(change_in_value)
        # print("common_pool",self.common_pool)
        # print("change_in_value",change_in_value)

    def make_decision(self, worker_id, observation):
        # ... (logic to make decisions based on the current observations)
        self.decisions[worker_id] = None  # store the decision
        pass

    def reset(self, *, seed=None, options=None):
        # self.state = self._initiate_state()
        self._reset_to_initial_values()
        obs= self._get_state()
        manager_obs = obs
        manager_info = {}



        return manager_obs, manager_info

    def _reset_to_initial_values(self):
        self.step_count = self.day = 0
        self.decisions = {}
        self.worker_total_trades_dict = {}
        self.worker_rewards_dict = {}
        self.worker_expsoure_dict = {}
        self.worker_accumulated_pnl_dict = {}
        self.worker_sharpe_ratio_dict = {}
        self.return_worker_return_memory_dict = {}
        self.worker_free_cash = {}
        self.total_cash_transfers = 0
        self.common_pool = np.float32(self.total_capital)

    
    def _get_state(self):
        state = {}
        
        for ticker in self.unique_tickers:
            worker = self.get_worker_by_ticker(ticker)
            
            # Create a nested dictionary for this ticker with the calculated metrics
            if worker:
                state[ticker] = {
                    "portfolio_value": np.array([worker._calculate_assets()], dtype=np.float32),
                    "return_mean": np.array([worker._calculate_return_mean()], dtype=np.float32),
                    "return_std": np.array([worker._calculate_return_std()], dtype=np.float32),
                    "Sharpe_Ratio": np.array([worker._get_sharpe_ratio()], dtype=np.float32),
                }
            else:
                print("no worker found for ticker")
        
        return state
    
    def get_worker_by_ticker(self, ticker):
        return self.workers.get(ticker)
    
    def update_total_capital(self):
        # Calculate the total value of all assets held by the workers
        total_assets_value = self._calculate_total_portfolio_value()
        # Update the total capital to be the sum of the current common pool and the total assets value
        self.total_capital = np.float32(self.common_pool + total_assets_value[0])
        # print("self.total_capital - update capital Manager",self.total_capital)


    
    def _calculate_total_portfolio_value(self):
        portfolio_value = 0
        for worker in self.workers.values():
            portfolio_value += worker._calculate_assets()
        return np.array([portfolio_value], dtype=np.float32)
    
    def _calculate_free_cash(self):
        free_cash = 0
        for worker in self.workers.values():
            free_cash += worker.current_cash
        return np.float32(free_cash)
    
    def _calculate_share_portfolio(self):
        stock_value = 0
        for worker in self.workers.values():
            stock_value += worker._calculate_stock_exposure()
        return np.float32(stock_value)

    def _calculate_total_portfolio_trades(self):
        if not self.worker_total_trades_dict:
            return np.array([0.0], dtype=np.float32)
        total_trades = sum(self.worker_total_trades_dict.values())
        return np.array(total_trades, dtype=np.float32)

            
    
    def _calculate_sharpe_ratio(self):
        """ Calculate Sharpe Ratio for all workers combined """
        # Calculate Sharpe Ratio for all workers combined
        total_returns = sum(self.worker_rewards_dict.values())
        total_volatility = np.std(list(self.worker_rewards_dict.values()))      
        sharpe_ratio = total_returns / total_volatility if total_volatility != 0 else 0
        if np.isnan(sharpe_ratio):
            sharpe_ratio = 0
        return np.array([sharpe_ratio], dtype=np.float32)
    


    
    def calculate_cash_allocation(self):
        return np.array([0.0], dtype=np.float32)
    def _calculate_risk_limit(self):
        return np.array([0.0], dtype=np.float32)
    def _calculate_position_limit(self):
        return np.array([0.0], dtype=np.float32)

    


    def pre_approve_trade(self, worker_id, proposed_trade):
        # Here, proposed_trade could be a dictionary with details of the proposed trade
        # such as {'action': 'buy', 'amount': 1000, 'price': 50}
        
        # Apply risk management rules to decide whether to approve the trade
        # For simplicity, let's just use a dummy rule here. You can replace this with real rules
        if proposed_trade['amount'] * proposed_trade['price'] <= self.capital_allocation[worker_id]:
            return True  # Approve the trade
        else:
            return False  # Deny the trade
    def manage_transaction_costs(self, worker_id, proposed_trade):
        # Here, you can implement logic to estimate the transaction costs of the proposed trade
        # and advise the worker to delay the trade if the costs are too high
        # For simplicity, let's just use a dummy rule here. You can replace this with real rules
        estimated_cost = proposed_trade['amount'] * proposed_trade['price'] * 0.001  # Just a dummy estimation
        if estimated_cost <= 10:  # Dummy threshold
            return True  # Proceed with the trade
        else:
            return False  # Advise to delay the trade
        
    
    def _calculate_reward(self):
        #  reward = self._calculate_portfolio_sharpe_ratio()
        #  reward = reward[0]
         annual_return = self._calculate_annualized_return()
         if np.isnan(annual_return):
            print("reward in worker is NaN")
            annual_return = 0
         return np.float32(annual_return)





    def _calculate_annualized_return(self):
        """
        Calculate the annualized rate of return based on the number of days and total assets among all workers.
        """
        if self.day == 0:
            return np.float32(0.0)
        total_assets = sum(worker._calculate_assets() for worker in self.workers.values())
        # print("Total assets - calc annual return:", total_assets)
        start_value = self.total_capital
        # print("start_value - calc annual return:", start_value)
        return_value = total_assets / start_value
        # print("return_value - calc annual return:", return_value)
        annualized_return = (return_value ** (365.0 / self.day)) - 1
        # print("self.day",self.day)
        # print("annual return",np.float32(annualized_return))
        return np.float32(annualized_return)
    
    def _calculate_entropy_bonus(self, weights):
        entropy = -np.sum(weights * np.log(weights + 1e-6))
        return np.float32(entropy)
    
    def _calculate_portfolio_sharpe_ratio(self):
        """
        Calculate the overall portfolio Sharpe ratio.

        The method calculates the annualized Sharpe ratio based on the daily returns of the entire portfolio,
        which is aggregated from the daily returns of individual workers. If the standard deviation of the 
        returns is zero or if any non-finite values are encountered, the Sharpe ratio is returned as zero.

        Returns:
            float: The calculated Sharpe ratio.
        """
        if not self.return_worker_return_memory_dict:
            return np.array([0.0], dtype=np.float32)

        portfolio_returns = np.concatenate([returns for returns in self.return_worker_return_memory_dict.values() if returns.size > 0])

        # If concatenation results in an empty array, return a default Sharpe ratio value of 0
        if portfolio_returns.size == 0:
            return np.array([0.0], dtype=np.float32)

        # Step 2: Calculate Mean and Standard Deviation
        # Ensure that there are no non-finite values in the returns array
        if not np.all(np.isfinite(portfolio_returns)):
            return np.array([0.0], dtype=np.float32)

        # Calculate the mean and standard deviation of the portfolio returns
        mean_returns = np.mean(portfolio_returns)
        std_returns = np.std(portfolio_returns)

        # Step 3: Calculate Sharpe Ratio
        # Calculate the annualized Sharpe ratio
        sharpe_ratio = (mean_returns / std_returns) * np.sqrt(252) if std_returns != 0 else 0.0

        # Check for non-finite values in the calculated Sharpe ratio
        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = np.array([0.0], dtype=np.float32)

        return np.array([sharpe_ratio], dtype=np.float32)
    

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
        # def step(self, action, worker_observations, worker_rewards, worker_dones, worker_truncateds):
    #     self.step_count += 1
    #     self.worker_total_trades_dict = {}
    #     self.worker_rewards_dict = {}
    #     self.worker_expsoure_dict = {}
    #     self.worker_accumulated_pnl_dict = {}
    #     self.worker_sharpe_ratio_dict = {}
    #     self.return_worker_return_memory_dict = {}
    #     self.worker_return_per_volatility = {}
    #     for worker_id, worker in self.workers.items():
    #         worker_state = worker._get_state()
    #         worker_sharpe_ratios = worker._get_sharpe_ratio()
    #         worker_return_memory = worker._get_return_memory()
    #         worker_current_cash = worker.current_cash
    #         self.worker_total_trades_dict[worker_id] = worker_state["total_trades"]           
    #         self.worker_rewards_dict[worker_id] = worker_state["current_stock_exposure"]
    #         self.worker_expsoure_dict[worker_id] = worker_state["current_stock_exposure"]
    #         self.worker_accumulated_pnl_dict[worker_id] = worker_state["pnl"]
    #         self.worker_sharpe_ratio_dict[worker_id] = worker_sharpe_ratios
    #         self.return_worker_return_memory_dict[worker_id] = worker_return_memory
    #         self.worker_return_per_volatility[worker_id] = worker_state["return_per_volatility"]
    #         self.worker_free_cash[worker_id] = worker_current_cash

    #     # Generate high-level orders for each worker
    #     if self.step_count % 50 == 0:
    #         self.reallocate_capital()

    #     all_workers_done = all(worker_dones)
    #     all_workers_truncated = any(worker_truncateds)
    
    #     # for ticker, worker in self.workers.items():
    #     #     worker.set_directives(action[ticker])
        
    #     # Update the manager's state based on worker observations or other criteria
    #     self.day += 1
    #     reward = self._calculate_reward()


    #     obs = self._get_state()
    #     info = {}
    #     # print("Manager - size: obs: ",asizeof.asizeof(obs))
    

    #     return obs, reward, all_workers_done, all_workers_truncated, info