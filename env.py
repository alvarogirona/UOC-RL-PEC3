import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, Box
from collections import namedtuple, deque
from copy import deepcopy
import math
import time
import torch
import torch.nn.functional as F
from tabulate import tabulate
import pandas as pd
import yfinance as yf
import warnings
import csv
import os

start = "2019-01-01"
end = "2021-01-01"
ticker = "SPY"
initial_balance = 10_000

class StockMarketEnv(gym.Env):
    def __init__(
            self,
            ticker=ticker,
            initial_balance=initial_balance,
            is_eval=False,
            start=start,
            end=end,
            save_to_csv=False,
            csv_filename="stock_trading_log.csv"
    ):
        super(StockMarketEnv, self).__init__()

        # Descargar los datos históricos de la acción
        self.df = yf.download(ticker, start, end)
        self.num_trading_days = len(self.df)
        self.prices = self.df.Close.values
        self.n_steps = len(self.prices) - 1

        # Parámetros del entorno
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.previus_net_worth = initial_balance

        # Espacio de acciones: 0 -> mantener, 1 -> comprar, 2 -> vender
        self.action_space = gym.spaces.Discrete(3)

        # Calculamos los indicadores técnicos
        self.rsi = calculate_rsi(self.df.Close).values
        self.ema = calculate_ema(self.df.Close).values

        # Espacio de observaciones: [precio_actual, balance, acciones, rsi, ema, sma, upper_band, lower_band]
        self.observation_space = gym.spaces.Box(low=-1, high=1, dtype=np.int8)
        self.is_eval = is_eval

        # Valores para normalización (obtenemos mínimos y máximos)
        self.min_price = self.prices.min()
        self.max_price = self.prices.max()
        self.min_rsi = self.rsi.min()
        self.max_rsi = self.rsi.max()
        self.min_ema = self.ema.min()
        self.max_ema = self.ema.max()

        # Parámetros adicionales para el CSV
        self.save_to_csv = save_to_csv
        self.csv_filename = csv_filename

        # Si la opción de almacenar en CSV está activada, crea o sobreescribe el archivo
        if self.save_to_csv:
            pass

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.previus_net_worth = self.initial_balance

        if seed is not None:
            super().reset(seed=seed)

        return self._next_observation(), {}

    def _normalize(self, value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    def _next_observation(self):
        # Normalizamos los valores
        current_price = self.prices[self.current_step][0]
        norm_price = self._normalize(current_price, self.min_price, self.max_price)
        norm_balance = self._normalize(self.balance, self.initial_balance * 0.85, self.initial_balance * 1.25)
        norm_shares_held = self._normalize(self.shares_held, 0, 100)  # Máximo de 100 acciones
        norm_rsi = self._normalize(self.rsi[self.current_step], self.min_rsi, self.max_rsi)
        norm_ema = self._normalize(self.ema[self.current_step], self.min_ema, self.max_ema)

        return np.array([
            norm_price,
            norm_balance,
            norm_shares_held,
            norm_rsi,
            norm_ema,
        ])

    def step(self, action):
        current_price = self.prices[self.current_step][0]
        self.previus_net_worth = self.net_worth

        if action == 1:  # Buy
            max_shares_possible = self.balance // current_price
            shares_to_buy = max_shares_possible

            purchase_cost = shares_to_buy * current_price
            self.balance -= purchase_cost
            self.shares_held += shares_to_buy

        elif action == 2:  # Sell
            sale_value = self.shares_held * current_price
            self.balance += sale_value
            self.shares_held = 0

        self.net_worth = self.balance + (self.shares_held * current_price)

        step_reward = self.net_worth - self.previus_net_worth

        self.current_step += 1

        is_terminated = self.current_step >= self.n_steps
        is_truncated = self.net_worth < (self.initial_balance * 0.85)

        if self.save_to_csv:
            self.save_to_csv_file()

        obs = self._next_observation()

        info = {
            'current_price': current_price,
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
        }

        return obs, step_reward, is_terminated, is_truncated, info

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')
        print('-' * 30)

    # La función save_to_csv_file guarda los datos actuales en un archivo CSV.
    # 1. Primero calcula el beneficio como la diferencia entre el valor neto
    # actual y el balance inicial.
    # 2. Luego, abre (o crea) el archivo CSV en modo 'append' para agregar una
    # nueva fila de datos sin sobrescribir las anteriores.
    # 3. Escribe una nueva fila en el CSV con los valores del paso actual,
    # balance, acciones mantenidas, valor neto y el beneficio.
    # Step,Balance,Shares Held,Net Worth,Profit
    # 1,12000,50,13000,3000
    def save_to_csv_file(self):
        pass
        """Guarda los datos actuales en el archivo CSV."""
        profit = self.net_worth - self.initial_balance
        data_to_write = {
            'Step': self.current_step,
            'Balance': round(self.balance, 2),
            'Shares Held': self.shares_held,
            'Net Worth': round(self.net_worth, 2),
            'Profit': round(profit, 2)
        }

        file_exists = os.path.isfile(self.csv_filename)

        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data_to_write.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(data_to_write)

class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = namedtuple('Buffer',
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in


class NeuralNetStockMarket(torch.nn.Module):

    ###################################
    ###inicialización y modelo###
    def __init__(self, env, learning_rate=1e-3, optimizer = None, device=None):

        """
        Params
        ======
        n_inputs: tamaño del espacio de estados
        n_outputs: tamaño del espacio de acciones
        actions: array de acciones posibles
        """
        super(NeuralNetStockMarket, self).__init__()
        
        # Initialize parameters
        self.n_inputs = env.observation_space.shape[0]  # Size of state space
        self.n_outputs = env.action_space.n  # Size of action space (3 actions)
        self.actions = np.arange(self.n_outputs)
        self.learning_rate = learning_rate

        # Define device (CPU or GPU)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build neural network
        self.model = torch.nn.Sequential(
            # First hidden layer
            torch.nn.Linear(self.n_inputs, 256, bias=True),
            torch.nn.ReLU(),
            
            # Second hidden layer
            torch.nn.Linear(256, 128, bias=True),
            torch.nn.ReLU(),
            
            # Third hidden layer
            torch.nn.Linear(128, 64, bias=True),
            torch.nn.ReLU(),
            
            # Output layer
            torch.nn.Linear(64, self.n_outputs, bias=True)
        ).to(self.device)

        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optimizer

    def get_action(self, state, epsilon=0.05):
        """
        Select action using epsilon-greedy policy
        """
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)  # Random action
        else:
            qvals = self.get_qvals(state)  # Get Q-values
            action = torch.max(qvals, dim=-1)[1].item()  # Select action with highest Q-value
        return action

    def get_qvals(self, state):
        """
        Get Q-values for all actions given a state
        """
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(self.device)
        return self.model(state_t)

class DQNAgent:
    def __init__(self, env, main_network, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32, min_episodes=300, device=None):
        # Initialize variables
        self.env = env
        self.main_network = main_network
        self.target_network = deepcopy(main_network)  # Create copy of main network
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.min_episodes = min_episodes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tracking variables
        self.step_count = 0
        self.episode_rewards = []  # Store rewards for each episode
        self.mean_rewards_history = []  # Store mean rewards every 100 episodes
        self.epsilon_history = []  # Store epsilon values
        self.loss_history = []  # Store loss values
        self.episodes_train_dqn = 0  # Track number of training episodes
        
        self.initialize()

    def initialize(self):
        # Set target network weights equal to main network weights initially
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.state0 = self.env.reset()[0]
        self.total_reward = 0
        self.update_loss = []

    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            # Random action for exploration
            action = self.env.action_space.sample()
        else:
            # Epsilon-greedy action selection
            action = self.main_network.get_action(self.state0, eps)
            self.step_count += 1

        # Take step in environment
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        # Store experience in buffer
        self.buffer.append(self.state0, action, reward, done, next_state)
        
        # Update state and accumulate reward
        self.state0 = next_state
        self.total_reward += reward

        if done:
            self.episode_rewards.append(self.total_reward)
            return True
        return False

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(self.device)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Get Q-values from main network
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        
        # Get next Q-values from target network
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                              dim=-1)[0].detach()
        qvals_next[dones_t] = 0

        # Calculate expected Q-values using Bellman equation
        expected_qvals = rewards_vals + self.gamma * qvals_next
        
        # Calculate MSE loss
        loss = F.mse_loss(qvals.squeeze(), expected_qvals)
        return loss

    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32, dnn_update_frequency=4,
              dnn_sync_frequency=2000, REWARD_THRESHOLD=9000):
        
        self.gamma = gamma
        
        # Fill replay buffer
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        print("Training...")
        while training:
            self.state0 = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            
            while not gamedone:
                # Take action
                gamedone = self.take_step(self.epsilon, mode='train')
                
                # Update main network according to frequency
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                
                # Sync target network according to frequency
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.main_network.state_dict())
                
                if gamedone:
                    episode += 1
                    self.episodes_train_dqn = episode
                    
                    # Calculate mean reward over last 100 episodes
                    mean_rewards = np.mean(self.episode_rewards[-100:])
                    self.mean_rewards_history.append(mean_rewards)
                    
                    # Store epsilon
                    self.epsilon_history.append(self.epsilon)
                    
                    # Reset loss tracking for next episode
                    self.update_loss = []
                    
                    print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")
                    
                    # Check episode limits
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    if mean_rewards >= REWARD_THRESHOLD and episode >= self.min_episodes:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(episode))
                        break
                    
                    # Update epsilon
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)

    def update(self):
        self.main_network.optimizer.zero_grad()  # eliminamos cualquier gradiente pasado
        batch = self.buffer.sample_batch(batch_size=self.batch_size) # seleccionamos un conjunto del buffer
        loss = self.calculate_loss(batch) # calculamos la pérdida
        loss.backward() # hacemos la diferencia para obtener los gradientes
        self.main_network.optimizer.step() # aplicamos los gradientes a la red neuronal
        # Guardamos los valores de pérdida
        self.update_loss.append(loss.detach().cpu().numpy())


