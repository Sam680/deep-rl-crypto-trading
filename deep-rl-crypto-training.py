import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from binance.client import Client
import random
from sklearn.preprocessing import MinMaxScaler

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        data = self.data[data_idx]
        return idx, self.tree[idx], data

    def __len__(self):
        return self.n_entries


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = SumTree(2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Need to reshape input for LSTM layers.
        input_layer = layers.Input(shape=(self.state_size, 1))  # Adjust this based on your state representation

        # Value Stream
        value_lstm = layers.LSTM(50, return_sequences=True)(input_layer)
        value_lstm = layers.LSTM(30, return_sequences=False)(value_lstm)
        value = layers.Dense(1)(value_lstm)

        # Advantage Stream
        advantage_lstm = layers.LSTM(50, return_sequences=True)(input_layer)
        advantage_lstm = layers.LSTM(30, return_sequences=False)(advantage_lstm)
        advantage = layers.Dense(self.action_size)(advantage_lstm)

        # Aggregating Layer
        output = layers.Add()([value, layers.Subtract()([advantage, layers.Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)])])
        model = models.Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=Adam())
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)  # Store data as a tuple
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = 1
        self.memory.add(max_priority, (experience,))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = []
        idxs = []
        segment = self.memory.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.memory.get(s)
            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        # Filter out None values from minibatch
        minibatch = [x for x in minibatch if x is not None]
        if not minibatch:
            return  # Skip replay if minibatch is empty

        states = np.stack([np.reshape(x[0][0], (self.state_size, 1)) for x in minibatch])
        actions = np.array([x[0][1] for x in minibatch])
        rewards = np.array([x[0][2] for x in minibatch])
        next_states = np.array([x[0][3] for x in minibatch])
        dones = np.array([x[0][4] for x in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = self.model.predict_on_batch(states)
        target_next = self.model.predict_on_batch(next_states)  # Selecting action here
        target_val = self.target_model.predict_on_batch(next_states)  # Evaluating action here

        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                a = np.argmax(target_next[i])  # Select action from model network
                targets[i][actions[i]] = rewards[i] + self.gamma * (target_val[i][a])  # Get value from target network

            self.memory.update(idxs[i], abs(targets[i][actions[i]] - priorities[i]))

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            # Adjusted epsilon decay
            if self.epsilon > 0.1:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon *= self.epsilon_decay * 1.01


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.position_value = 0
        self.position_held = False
        self.position_duration = 0
        self.inactivity_duration = 0
        self.profits = 0
        self.trade_count = 0
        self.day_count = 0
        self.t = 0
        self.done = False
        self.stop_loss = 0.02
        self.take_profit = 0.02
        self.features = ['Close', 'Volume', 'EMA5', 'EMA8', 'EMA13', 'EMA21', 'EMA50', 'EMA100', 'EMA200', 'RSI', 'MACD', 'Signal', '%K', '%D']
        self.history_t = 90
        self.history = np.zeros((self.history_t, len(self.features)))
        self.reset()

    def reset(self):
        self.t = 0
        self.done = False
        self.profits = 0
        self.position_value = 0
        self.history = np.zeros((self.history_t, len(self.features)))
        self.position_held = False  # Track if a position is currently held
        self.position_duration = 0  # Track how long a position is held
        self.inactivity_duration = 0  # Track how long a position is not held
        self.trade_count = 0  # Track the number of trades
        self.day_count = 0  # Add this line to keep track of the number of days

        history_flat = self.history.flatten()
        return np.concatenate(([np.array(self.position_value)], history_flat))

    def step(self, act):
        reward = 0
        commission_fee = 0.001  # 0.1% commission fee
        percentage_profit = 0
        potential_percentage_profit = 0
        position_held = self.position_held

        # Calculate the current profit/loss percentage
        if self.position_held:
            price = self.data.iloc[self.t, :]['Close']
            sell_price = price * (1 - commission_fee)
            current_profit_loss = (sell_price - self.position_value) / self.position_value

            # Sell if the stop loss or take profit level is hit
            if current_profit_loss <= -self.stop_loss or current_profit_loss >= self.take_profit:
                percentage_profit = current_profit_loss
                if percentage_profit > 0:
                    reward = percentage_profit + 1
                else:
                    reward = percentage_profit - 1
                self.profits += percentage_profit
                self.position_held = False

        if act == 1:  # buy
            if position_held:
                reward = -1  # Penalty for buying when a position is already held
            else:
                price = self.data.iloc[self.t, :]['Close']
                reward += 0.003
                self.position_value = price * (1 - commission_fee)  # Account for commission fee
                self.trade_count += 1  # Increment the trade count when a trade is made
                self.position_held = True

        # Calculate the potential profit if a position is currently held (reward for half the value)
        if self.position_held:
            price = self.data.iloc[self.t, :]['Close']
            potential_sell_price = price * (1 - commission_fee)
            potential_percentage_profit = (potential_sell_price - self.position_value) / self.position_value
            if potential_percentage_profit > 0:
                reward += potential_percentage_profit / 2
            else:
                reward -= potential_percentage_profit / 2

        # Set next time
        self.t += 1
        if self.t == len(self.data):
            self.done = True
            return (
                np.concatenate(([self.position_value], self.history.flatten())),
                reward,
                self.done,
                percentage_profit,  # obs, reward, done
            )
        self.history = self.history[1:]  # Remove the first element
        new_history = [
            self.data.iloc[self.t, :][feature] - self.data.iloc[self.t - 1, :][feature]
            for feature in self.features
        ]
        new_history = np.array(new_history, dtype=float).reshape(1, -1)  # Reshape to match self.history
        self.history = np.vstack((self.history, new_history))

        # Penalty for long holding
        if self.position_held and self.position_duration > 3:
            reward -= 0.0005 * (self.position_duration - 3)  # Increase penalty the longer the position is held

        # Penalty for long inactivity
        if not self.position_held and self.inactivity_duration > 3:
            reward -= 0.001 * (self.inactivity_duration - 3)  # Increase penalty the longer the inactivity

        # Reward for making > 10 trades in a day
        self.day_count += 1

        # Check if a day has passed. 288 * 5 minute intervals = 1 day.
        if self.day_count >= 288:
            self.day_count = 0  # Reset the day count
            if self.trade_count >= 10:  # If 10 or more trades were made, add a reward
                reward += 1  # Modify this reward value as needed
            self.trade_count = 0  # Reset the trade count

        # Update position holding time
        if self.position_held:
            self.position_duration += 1
            self.inactivity_duration = 0
        else:
            self.position_duration = 0
            self.inactivity_duration += 1

        # Update done status
        self.done = True if self.t == len(self.data) - 1 else False

        return (
            np.concatenate(([self.position_value], self.history.flatten())),
            reward,
            self.done,
            percentage_profit,
            potential_percentage_profit,
        )


def calculate_macd(data, short_window, long_window):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal'] = data['MACD'].ewm(span=4, adjust=False).mean()
    data['MACD'] = data['MACD'].astype(float)
    data['Signal'] = data['Signal'].astype(float)
    return data


def calculate_rsi(data, period):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    average_gain = up.rolling(window=period).mean()
    average_loss = abs(down.rolling(window=period).mean())
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_stochastic_oscillator(data, period=14):
    high14 = data['High'].rolling(period).max()
    low14 = data['Low'].rolling(period).min()
    data['%K'] = (100 * ((data['Close'] - low14) / (high14 - low14))).astype(float)
    data['%D'] = (data['%K'].rolling(3).mean()).astype(float)
    return data


if __name__ == "__main__":
    episodes = 1000
    batch_size = 32

    api_key = ''
    api_secret = ''
    client = Client(api_key, api_secret)

    # Download historical data from Binance
    candles = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, "24 May, 2023")
    data = pd.DataFrame(
        candles,
        columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
            'Number of trades', 'Taker buy base', 'Taker buy quote', 'Ignore'
        ]
    )
    data['Date'] = pd.to_datetime(data['Date'], unit='ms')

    data = data.sort_values('Date').reset_index()
    data = data.drop(
        [
            'Date', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base',
            'Taker buy quote', 'Ignore'
        ],
        axis=1
    )
    data['Close'] = data['Close'].astype(float)
    data['High'] = data['High'].astype(float)
    data['Low'] = data['Low'].astype(float)
    data['Open'] = data['Open'].astype(float)
    data['Volume'] = data['Volume'].astype(float)

    data['EMA5'] = (data['Close'].ewm(span=5, adjust=False).mean()).astype(float)
    data['EMA8'] = (data['Close'].ewm(span=8, adjust=False).mean()).astype(float)
    data['EMA13'] = (data['Close'].ewm(span=13, adjust=False).mean()).astype(float)
    data['EMA21'] = (data['Close'].ewm(span=21, adjust=False).mean()).astype(float)
    data['EMA50'] = (data['Close'].ewm(span=50, adjust=False).mean()).astype(float)
    data['EMA100'] = (data['Close'].ewm(span=100, adjust=False).mean()).astype(float)
    data['EMA200'] = (data['Close'].ewm(span=200, adjust=False).mean()).astype(float)
    data['RSI'] = calculate_rsi(data['Close'], 14)
    data = calculate_stochastic_oscillator(data)
    data = calculate_macd(data, 6, 13)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['RSI', 'MACD', 'Signal', '%K', '%D']] = scaler.fit_transform(data[['RSI', 'MACD', 'Signal', '%K', '%D']])

    print(data)
    print(len(data))
    state_size = 1 + (90 * 14)
    action_size = 2
    agent = DQNAgent(state_size, action_size)
    env = TradingEnvironment(data)

    for e in range(episodes):
        print("Episode " + str(e) + "/" + str(episodes))
        state = env.reset()
        state = np.reshape(state, [1, state_size, 1])  # Reshape state
        i = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, percentage_profit, potential_percentage_profit = env.step(action)
            next_state = np.reshape(next_state, [1, state_size, 1])  # Reshape next_state
            agent.remember(state, action, reward, next_state, done)
            print(
                f"e{e}: {i}/{len(data)}\taction: {action}\tprofit: {percentage_profit:.3f}"
                f"\tpot_profit: {potential_percentage_profit:.3f}\treward: {reward:.3f}\ttotal_profit: {env.profits:.4f}"
            )
            i += 1
            state = next_state
            if done:
                output = "episode: {}/{}, score: {}, e: {:.2}\n".format(e, episodes, env.profits, agent.epsilon)
                print(output)
                with open('output.txt', 'a') as f:
                    f.write(output)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            agent.save("model_ep" + str(e) + ".h5")
    agent.save("model.h5")
