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
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.model(state_t)

        
class DQNAgent:
    def __init__(self, env, main_network, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32, min_episodes= 300, device = None):
        ######################################
        ##TODO 1: Declarar variables
        self.env = env
        self.main_network = main_network
        self.target_network = deepcopy(main_network) # red objetivo (copia de la principal)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = 100 # bloque de los X últimos episodios de los que se calculará la media de recompensa
        self.initialize()
        self.min_episodes = min_episodes
        self.device = torch.device("mps")
        # self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Configurar el dispositivo (CPU o GPU)


    def initialize(self):
        ######################################
        ##TODO 3: Inicializa lo necesario
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.epsilon_history = []  # Store epsilon values
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = self.env.reset()[0]

    ## Tomar nueva acción
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            # acción aleatoria en el burn-in y en la fase de exploración (epsilon)>
            action = self.env.action_space.sample() 
        else:
            # acción a partir del valor de Q (elección de la acción con mejor Q)
            action = self.main_network.get_action(self.state0, eps)
            self.step_count += 1
        #TODO: tomar 'step' i obtener nuevo estado y recompensa. Guardar la experiencia en el buffer
        new_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        # new_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, done, new_state) # guardem experiència en el buffer
        self.state0 = new_state.copy()
        
        #TODO: resetear entorno 'if done'
        if done:
            self.training_rewards.append(self.total_reward)
            self.total_reward = 0
            self.state0 = env.reset()[0]
        return done



    ## Entrenamiento
    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000, REWARD_THRESHOLD = 9000):

        self.gamma = gamma

        # Rellenamos el buffer con N experiencias aleatorias ()
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
            while gamedone == False:
                # El agente toma una acción
                gamedone = self.take_step(self.epsilon, mode='train')
                ##################################################################################
                #####TODO 4:  Actualizar la red principal según la frecuencia establecida #######
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                ########################################################################################
                ###TODO 6: Sincronizar red principal y red objetivo según la frecuencia establecida#####
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.main_network.state_dict())
                    self.sync_eps.append(episode)


                if gamedone:
                    episode += 1
                    #######################################################################################
                    ###TODO 7: calcular la media de recompensa de los últimos X episodios, y almacenar#####

                    self.update_loss = []
                    mean_rewards = np.mean(   # calculem la mitjana de recompensa dels últims X episodis
                        self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)
                    
                    # Store epsilon
                    self.epsilon_history.append(self.epsilon)

                    #######################################################################################
                    ### TODO 8: Comprobar que todavía quedan episodios. Parar el aprendizaje si se llega al límite

                    print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")

                    # Comprobamos que todavía quedan episodios
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        break

                    #######################################################################################
                    ### TODO 9: Añadir min episodes
                    if mean_rewards >= REWARD_THRESHOLD and self.min_episodes < episode :
                        training = False
                        print('\nEpisode limit reached.')
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        break


                    #################################################################################
                    ######TODO 9: Actualizar epsilon según la velocidad de decaimiento fijada########
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)

    ## Cálculo de la pérdida
    ## Càlcul de la pèrdua                   
    def calculate_loss(self, batch):
      #  print('loss')
        # Separamos las variables de la experiencia y las convertimos a tensores
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_vals = torch.FloatTensor(rewards).to(self.device)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Obtenemos los valores de Q de la red principal
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals).to(self.device)
        # Obtenemos los valores de Q objetivo. El parámetro detach() evita que estos valores actualicen la red objetivo
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach().to(self.device)
        qvals_next[dones_t.bool()] = 0

        #################################################################################
        ### TODO: Calcular ecuación de Bellman
        expected_q_values = rewards_vals + (self.gamma * qvals_next)  # Shape: (batch_size,)
        # expected_qvals = None

        #################################################################################
        ### TODO: Calcular la pérdida (MSE)
        loss = torch.nn.MSELoss()(qvals, expected_q_values.reshape(-1,1))
        return loss


    def update(self):
        self.main_network.optimizer.zero_grad()  # eliminamos cualquier gradiente pasado
        batch = self.buffer.sample_batch(batch_size=self.batch_size) # seleccionamos un conjunto del buffer
        loss = self.calculate_loss(batch) # calculamos la pérdida
        loss.backward() # hacemos la diferencia para obtener los gradientes
        self.main_network.optimizer.step() # aplicamos los gradientes a la red neuronal
        # Guardamos los valores de pérdida
        if self.device.type == 'mps':
            loss_value = loss.detach().to('cpu').item()
        else:
            loss_value = loss.detach().cpu().numpy()
        self.update_loss.append(loss_value)



# Define hyperparameters
LEARNING_RATE = 0.0005
BATCH_SIZE = 128
MAX_EPISODES = 4000
BURN_IN = 1000
UPDATE_FREQ = 6
SYNC_FREQ = 15
MEMORY_SIZE = 50000
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995

# Calculate REWARD_THRESHOLD
ticker = 'SPY'
start = '2019-01-01'
end = '2021-01-01'

# Download data to get number of trading days
temp_data = yf.download(ticker, start, end)
num_days = len(temp_data)
print(f"Numero de dias de trading para {ticker} desde {start} hasta {end}: {num_days}")
print(f"Nuestro objetivo ganar el 50 por ciento de los dias: {round(num_days/2)}")
REWARD_THRESHOLD = round(num_days/2)

# Training
print("Starting training...")
start_time = time.time()

# Create environment
env = StockMarketEnv(ticker=ticker, start=start, end=end)

# Create experience replay buffer
buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)

# Create neural network
main_network = NeuralNetStockMarket(
    env=env,
    learning_rate=LEARNING_RATE
)

# Create DQN agent
agent = DQNAgent(
    env=env,
    main_network=main_network,
    buffer=buffer,
    epsilon=EPSILON,
    eps_decay=EPSILON_DECAY,
    batch_size=BATCH_SIZE,
    min_episodes=300
)

# Train the agent
agent.train(
    gamma=GAMMA,
    max_episodes=MAX_EPISODES,
    batch_size=BATCH_SIZE,
    dnn_update_frequency=UPDATE_FREQ,
    dnn_sync_frequency=SYNC_FREQ,
    REWARD_THRESHOLD=REWARD_THRESHOLD
)

# Calculate training time
end_time = time.time()
training_time = (end_time - start_time) / 60  # Convert to minutes
print(f"\nTraining time: {training_time:.2f} minutes")

# Print final results
print(f"Episodes completed: {agent.episodes_train_dqn}")
print(f"Final mean reward: {agent.mean_training_rewards[-1]:.2f}")


#Generar un fichero CSV que registre los resultados de las interacciones
#del agente con el mercado en cada episodio y muestra por pantalla las últimas 30 acciones.
file_path = 'stock_trading_agent_dqn.csv'
#Reproducir una partida completa del agente entrenado y mostrar el resultado final,
#incluyendo el valor total del portafolio al final del episodio.
env = ''#TODO


def read_csv_and_show_last_30(file_path):
    try:
        df = pd.read_csv(env.csv_filename)
        print(f"\nLast 30 rows of {env.csv_filename}:")
        print(df.tail(30))

    except FileNotFoundError:
        print(f"El archivo {file_path} no fue encontrado.")
    except Exception as e:
        print(f"Se produjo un error al leer el archivo: {e}")

# Ejemplo de uso
read_csv_and_show_last_30(file_path)

import random
import pandas as pd
import numpy as np

# Generar calendario de trading usando días hábiles
def generate_random_trading_dates(start_range, end_range, trading_days_target=505):
    """
    Genera un par de fechas (start, end) que tengan exactamente trading_days_target días hábiles.
    """
    start_date = pd.to_datetime(start_range)
    end_date = pd.to_datetime(end_range)

    while True:
        # Seleccionar una fecha de inicio aleatoria
        random_start = start_date + pd.DateOffset(days=random.randint(0, (end_date - start_date).days - trading_days_target))

        # Generar un rango de fechas de trading usando solo los días hábiles
        trading_days = pd.bdate_range(random_start, random_start + pd.DateOffset(days=2 * trading_days_target)).tolist()

        # Filtrar las fechas para obtener exactamente el número de días objetivo
        if len(trading_days) >= trading_days_target:
            random_end = trading_days[trading_days_target - 1]  # Último día de trading en el rango deseado
            return random_start.strftime("%Y-%m-%d"), random_end.strftime("%Y-%m-%d")

def test_model(ag, base_env, start_range, end_range, trading_days_target=505, win_days_target=252):
    all_rewards = []
    win_days_count = []
    
    for i_episode in range(100):
        # Generate random start and end dates
        start_date, end_date = generate_random_trading_dates(start_range, end_range, trading_days_target)
        
        # Create new environment with random dates
        env = StockMarketEnv(
            ticker='SPY',
            start=start_date,
            end=end_date,
            save_to_csv=(i_episode == 0)  # Save CSV only for first episode
        )
        
        # Reset environment and initialize variables
        state = env.reset()[0]
        episode_reward = 0
        positive_days = 0
        done = False
        
        # Run episode
        while not done:
            # Get action from trained agent
            action = ag.main_network.get_action(state, epsilon=0.01)  # Small epsilon for some exploration
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Count positive days (when reward > 0)
            if reward > 0:
                positive_days += 1
        
        # Store results
        all_rewards.append(episode_reward)
        win_days_count.append(positive_days)
        
        # Print progress
        print(f"\rTest Episode {i_episode + 1}/100: Reward = {episode_reward:.2f}, Positive Days = {positive_days}", end="")
        
        env.close()

    # Calculate success rate (episodes with more than win_days_target positive days)
    success_rate = sum(days >= win_days_target for days in win_days_count) / len(win_days_count)
    
    return all_rewards, success_rate

def plot_test(rewards, th):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Test Rewards')
    plt.axhline(y=th, color='r', linestyle='--', label='Reward Threshold')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Test Performance Across Random Time Periods')
    plt.legend()
    plt.grid(True)
    plt.show()

# Test the trained agent
print("\nTesting agent across random time periods...")
test_rewards, success_rate = test_model(
    agent, 
    env,
    start_range="2015-01-01",
    end_range="2024-01-01",
    trading_days_target=505,
    win_days_target=252
)

# Store metrics
mean_reward_dqn = np.mean(test_rewards)
mean_reward_dqn_last = agent.mean_training_rewards[-1]

# Plot results
plot_test(test_rewards, REWARD_THRESHOLD)

# Print results
print(f"\nTest Results:")
print(f"Mean reward over 100 test episodes: {mean_reward_dqn:.2f}")
print(f"Final training reward: {mean_reward_dqn_last:.2f}")
print(f"Success rate (>252 positive days): {success_rate*100:.2f}%")

# Show last 30 actions from CSV
print("\nLast 30 trading actions:")
read_csv_and_show_last_30('stock_trading_agent_dqn.csv')


import torch.autograd as autograd

class duelingDQN(torch.nn.Module):

    def __init__(self, env, device=None, learning_rate=1e-3):

        """
        Parámetros
        ==========
        n_inputs: tamaño del espacio de estados
        n_outputs: tamaño del espacio de acciones
        actions: array de acciones posibles
        """

        ###################################
        ####TODO: Inicializar variables####
        super(duelingDQN, self).__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_inputs = #TODO
        self.n_outputs = #TODO
        self.actions = #TODO

        ######

        #######################################
        ##TODO: Construcción de la red neuronal
        # Red común
        ##Construcción de la red neuronal

        self.model_common = #TODO

        # Subred de la función de Valor
        self.fc_layer_inputs = self.feature_size()


        self.advantage  = #TODO

        # Recordad adaptarlas a cpu o gpu

        # Subred de la Ventaja A(s,a)
        self.value = #TODO

        #######
        #######################################
        ##TODO: Inicializar el optimizador
        self.optimizer = #TODO


    #######################################
    #####TODO: función forward#############
    def forward(self, state):
        # Conexión entre capas de la red común
        common_out = #TODO

        # Conexión entre capas de la Subred de Valor
        advantage = #TODO

        # Conexión entre capas de la Subred de Ventaja
        value = #TODO


        ## Agregar las dos subredes:
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
        action = #TODO

        return action
    #######



    ### MéTODO e-greedy
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            qvals = self.get_qvals(state)
            action = torch.max(qvals, dim=-1)[1].item()
        return action


    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(self.device)
        return self.forward(state_t)

    def feature_size(self):
        dummy_input = torch.zeros(1, *env.observation_space.shape).to(self.device)
        return self.model_common(autograd.Variable(dummy_input)).view(1, -1).size(1)