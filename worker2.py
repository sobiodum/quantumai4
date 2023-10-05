import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
import warnings
import traceback
from pympler import asizeof
import tensorflow as tf
import os
from gymnasium import wrappers

# Create directory if it doesn't exist
# os.makedirs("./tf_debug/worker2", exist_ok=True)

# # Enable the debugger
# # tf.debugging.experimental.enable_dump_debug_info("./tf_debug/worker2")
# tf.debugging.experimental.enable_dump_debug_info("./tf_debug/worker2", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

#! CATCH EROR WITH the following
# with warnings.catch_warnings():
#     warnings.filterwarnings('error')
#     try:
#         # Your suspected code here, e.g., np.mean(empty_array)
#     except RuntimeWarning:
#         traceback.print_exc()


class Spec(object):
    def __init__(self, id, max_episode_steps):
        self.id = id
        self.max_episode_steps = max_episode_steps


class Worker(gym.Env):
    def __init__(self,  initial_capital=1e6, trading_cost=0.001, initial_shares_held = 0,
                 invalid_action_penalty=-0, print_verbosity=1, worker_id=None, ticker_df=None, tic=None, **kwargs):
        super(Worker, self).__init__()

        # stock_data=pd.read_pickle("/Users/floriankockler/Documents/GitHub.nosync/quantumai4/train1.pkl")
        # filted_df = stock_data[stock_data["tic"]=="ABT.US"]
        self.df = ticker_df

        self.tic = tic
        self.worker_id = worker_id
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.trading_cost = trading_cost
        self.initial_shares_held = initial_shares_held
        self.episode = 0
        self.print_verbosity = print_verbosity
        self.done = False

        #State Info
        self.cash_initial = np.float32(initial_capital)
        self.current_cash = np.float32(initial_capital)
        self.tech_indicator_list = [ 'avgvol_50',
       'sma_10', 'sma_50', 'sma_100', 'sma_200', 'wma_50', 'rsi_14',
       'volatility_30', 'volatility_100', 'stddev_30', 'dmi_14', 'adx_14',
       'macd', 'atr_14']

        self.current_price = np.float32(self.df.iloc[self.day]['close'])
        #Cash Info
        self.cash_spent = 0
        self.cash_from_sales = 0
        #Stock Info
        self.shares_held = 0
        #Memory & Reward related Info

        self.reward = 0
        self.total_pnl_history = [0]
      
        self.actions_memory = [0]
        self.action_type = []
        self.trading_memory = [0]
        self.return_memory = [0]

        self.invalid_action_penalty = invalid_action_penalty

        self.previous_portfolio_value = self._calculate_assets() 

        #Track position change
     
        self.stock_holding_memory = np.array([initial_shares_held], dtype=np.float32)  # To store the net position after each trade
        self.position_memory = np.array([0], dtype=np.float32)  # To store the net position after each trade
        #Debugging info
        self.invalid_action_count = 0
        self.penalty = 0
        #Trading related
        self.trading_pentalty = 0
        self.total_costs = 0
        self.total_trades= 0
        self.current_step_cost = 0

        self.spec = Spec(id="worker-single-stock",
                         max_episode_steps=len(self.df.index.unique()) - 1)

        self.observation_space = spaces.Dict({
            'current_cash': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'shares_held': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'current_price': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'total_costs': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'day': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'tech_indicators': spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
            'pnl': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            # 'return_per_volatility': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'total_trades': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'current_stock_exposure': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        
        })


        self.action_space = spaces.Dict({
            'type': spaces.Discrete(3),  # 0: hold, 1: buy, 2: sell
            'amount': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Percentage of cash/shares to use
        })


  

    def step(self, action, controller_action):
        done = self.day >= len(self.df.index.unique()) - 1
        if done:
            self.done = True
            #!Leave prints out for testing purposes
            # self._handle_done()
        else:
            self.current_step_cost = 0
            begin_adj_portfolio_value = self._calculate_assets()
            assert np.isfinite(begin_adj_portfolio_value), f"begin_adj_portfolio_value is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"

            action_type = action['type']
            action_amount = action['amount'][0]
            

            if action_type == 0:  # hold
                pass
            elif action_type in [1, 2]:  # buy or sell
                trade = self._handle_trading(action_type,action_amount)
                self.actions_memory.append(action_amount)
                self.trading_memory.append(trade)

            self.action_type.append(action_type)
            self.day += 1
            self.data = self.df.loc[self.day, :]

            end_adj_portfolio_value = self._calculate_assets()
            assert np.isfinite(end_adj_portfolio_value), f"end_adj_portfolio_value is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
            # print("end_adj_portfolio_value", self._calculate_assets()/1000000)
            portfolio_return = np.float32(0) if begin_adj_portfolio_value == 0 else np.float32((end_adj_portfolio_value / begin_adj_portfolio_value) - 1)
            assert np.isfinite(portfolio_return), f"portfolio_return is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
            if isinstance(portfolio_return, np.ndarray) and portfolio_return.size == 1:
                portfolio_return = portfolio_return.item()
            self.return_memory.append(portfolio_return)
            total_pnl = self._calculate_pnl()
            assert np.isfinite(total_pnl), f"total_pnl is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
            self.total_pnl_history.append(total_pnl)
            self.reward = self._calculate_reward(begin_adj_portfolio_value, end_adj_portfolio_value)
            assert np.isfinite(self.reward ), f"self.reward  is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
            self.trading_pentalty = 0
            self.current_step_cost = 0
        truncated = False
        info = {}
        obs = self._get_state() 
        return obs, self.reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        self._reset_to_initial_values()
        self.episode += 1
        obs = self._get_state()
        info = {}
        return obs, info
   

    
    def _calculate_reward(self, begin_adj_portfolio_value, end_adj_portfolio_value):
        reward = 0
        pnl = end_adj_portfolio_value - begin_adj_portfolio_value - self.current_step_cost 
        assert np.isfinite(pnl), f"pnl  is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
        if self.invalid_action_count > 0:
            reward += self.invalid_action_penalty
        reward = 0.01 * pnl - self.trading_pentalty
        if np.isnan(reward):
            print("reward in worker is NaN")
            reward = np.float32(0)
        return np.float32(reward)
        
    
    def _handle_trading(self, action_type, action_amount):
        if self.current_price == 0:
            return 0

        shares_to_sell = shares_to_buy = 0

        if action_type == 2: #Selling
            if self.shares_held <= 0.0001:
                self.trading_pentalty += 1
                self.invalid_action_count += 1
                return 0
            action_nominal_value = self.current_price * self.shares_held * action_amount
            shares_to_sell = min(self.shares_held, int(action_nominal_value / self.current_price))
            assert np.isfinite(shares_to_sell), f"shares_to_sell  is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
            sell_amount = self.current_price * shares_to_sell * (1 - self.trading_cost)
            assert np.isfinite(sell_amount), f"sell_amount  is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"

            
            self.cash_from_sales += sell_amount
            self.current_cash += np.float32(sell_amount)
            self.total_trades += 1
           
        elif action_type == 1:
            if self.current_cash <= 0.9:
                self.trading_pentalty += 1
                self.invalid_action_count += 1
                return 0
            max_affordable_shares = self.current_cash / (self.current_price * (1 + self.trading_cost))
            assert np.isfinite(max_affordable_shares), f"max_affordable_shares  is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
            shares_to_buy = int(min(max_affordable_shares, action_amount * self.current_cash / self.current_price))    
            buy_amount = self.current_price * shares_to_buy * (1 + self.trading_cost)
            assert np.isfinite(buy_amount), f"buy_amount  is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
            self.cash_spent += buy_amount 
            self.current_cash -= np.float32(buy_amount)
            self.total_trades +=1


        net_position = np.float32(self.shares_held * self.current_price)
        assert np.isfinite(net_position), f"net_position  is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
        self.position_memory.append(net_position)
        self.shares_held += shares_to_buy - shares_to_sell
        assert np.isfinite(self.shares_held ), f"self.shares_held   is not finite (could be NaN, inf, or -inf) at day {self.day} and worker {self.worker_id}"
        self.current_step_cost += self.current_price * (abs(shares_to_sell) + abs(shares_to_buy)) * self.trading_cost
        self.total_costs += np.float32(self.current_step_cost)
        self.stock_holding_memory.append(self.shares_held)

        return np.float32(shares_to_buy - shares_to_sell)


    def _calculate_assets(self):
        """Calcualtes total assets for each worker - called by worker and manager"""
        return np.float32(self.current_cash + self.shares_held * self.current_price)
    


    def _calculate_pnl(self):
        """Calculated accumulated pnl and appends to self.total_pnl_history """
        pnl = (self.cash_from_sales - self.cash_spent)  + self.current_price * self.shares_held
        return np.float32(pnl)
    

    def _handle_done(self):
            if self.episode % self.print_verbosity == 0:
                print(f"========={self.worker_id} is done=============")
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"Beginn_Portfolio_Value: {self.cash_initial}")           
                print(f"End Total Assets: {self._calculate_assets()}")           
                print(f"Total PnL: {self._calculate_pnl()}")           
                print("=================================")

    def render(self, mode='human'):
        pass

    def close(self):
        pass


    
    def _get_state(self): 
        self.current_price = np.float32(self.data["close"])
        stock_data = self.data
        tech_indicators = [np.float32(stock_data[tech]) for tech in self.tech_indicator_list]

        state = {
            'current_cash': np.array([self.current_cash], dtype=np.float32),
            'shares_held': np.array([self.shares_held], dtype=np.float32),
            'current_price': np.array([self.current_price], dtype=np.float32),
            'total_costs': np.array([self.total_costs], dtype=np.float32),  # Total trading costs
            'day': np.array([self.day], dtype=np.float32),  # Current step in the episode
            'tech_indicators': np.array(tech_indicators, dtype=np.float32),
            'pnl': np.array([self._calculate_pnl()], dtype=np.float32),
            'total_trades': np.array([self.total_trades], dtype=np.float32),
            'current_stock_exposure': np.array([self.current_price * self.shares_held], dtype=np.float32),

        }
  
        return state

    def _reset_to_initial_values(self):
        self.done = False
        self.shares_held = self.reward = 0
        self.cash_spent = self.cash_from_sales = 0
        self.current_price = np.float32(self.df.iloc[self.day]['close'])
        self.invalid_action_count = 0
        self.day = 0
        self.current_step_cost = 0
        self.penalty = 0
        self.total_costs = np.float32(0)
        self.total_trades = 0
        self.actions_memory = [0]
        self.trading_memory = [0]
        self.stock_holding_memory = [self.initial_shares_held]  # To store the net position after each trade
        self.position_memory = [0]  # To store the net position after each trade
        self.action_type = []
        self.total_pnl_history
        self.current_cash = self.cash_initial
        self.return_memory = [0]

################----------------------------------------------------------------##############


#?Currently not used

    def _calculate_return_std(self):
        """ Calculate the standard deviation of all returns for manager"""
        returns = np.array(self.return_memory)
        if returns.size == 0:
            return np.float32(0.0)
        else:
            return np.float32(np.std(returns))
        

    def _get_return_memory(self):
        """
        Access the return memory for the manager
        
        Returns:
            np.ndarray: A numpy array containing all returns
            shorten to 400 obs to save memory
        """
        return np.array([self.return_memory[-400:]], dtype=np.float32)
        


    def _get_sharpe_ratio(self):
        """
        Calculate the Sharpe Ratio to be accessed by the manager.
        
        The method calculates the annualized Sharpe ratio based on the daily returns stored in self.return_memory.
        If the standard deviation of the returns is zero or if any non-finite values are encountered,
        the Sharpe ratio is returned as zero.
        
        Returns:
            np.ndarray: A numpy array containing the single Sharpe ratio value.
        """
        returns = np.array(self.return_memory)

        # Ensure that there are no non-finite values in the returns array
        if not np.all(np.isfinite(returns)):
            return np.float32(0.0)

        # Calculate the Sharpe ratio
        sharpe_ratio = ((np.mean(returns))  / (np.std(returns))) * np.sqrt(252) if np.std(returns) != 0 else 0.0

        # Check for non-finite values in the calculated Sharpe ratio
        if not np.isfinite(sharpe_ratio):
            sharpe_ratio = np.float32(0.0)

        return np.float32(sharpe_ratio)
    
    def _calculate_return_mean(self):
        """ Calculate the mean of all returns"""
        returns = np.array(self.return_memory)
        if returns.size == 0:
            return np.float32(0.0)
        mean_return = np.mean(returns)
        return np.float32(mean_return)
    
    def _calculate_stock_exposure(self):
        """Calculated total stock holdings per worker """
        stock_holdings = self.current_price * self.shares_held
        return np.float32(stock_holdings)
     
