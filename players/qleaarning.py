import numpy as np
import random


class TTTQLearningPlayer:
    def __init__(self, symbol, learning_rate=0.1, discount_factor=0.9):
        self.symbol = symbol
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.q_table = {}  # Initialize Q-table as an empty dictionary
        self.last_action = None  # Store the last action taken
    
    @classmethod
    def to_string(self) -> str:
        return "qlearning"

    def get_state(self, game):
        """Returns the current state as a tuple, which is hashable and can be used as a key in the Q-table."""
        return tuple([tuple(row) for row in game.board])

    def update_q_table(self, state, action, next_state, done, reward):
        """Update the Q-table using the Q-learning algorithm."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)  # 9 actions - for a 3x3 board
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)  # 9 actions - for a 3x3 board
        
        action_index = action[0] * 3 + action[1]  # Convert 2D action to 1D index
        
        if done:
            qsa_observed = reward  # If the game has ended, the reward is the final outcome
        else:
            qsa_observed = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        td_error = qsa_observed - self.q_table[state][action_index]
        self.q_table[state][action_index] += self.learning_rate * td_error

    def choose_action(self, state, available_actions):
         # Convert available actions to indices
        action_indices = [action[0] * 3 + action[1] for action in available_actions]
        q_values = self.q_table.get(state, np.zeros(9))
        
        # Filter Q-values for available actions only
        available_q_values = q_values[action_indices]
        max_q_value = np.max(available_q_values)
        
        # Filter actions with the max Q-value
        actions_with_max_q_value = [available_actions[i] for i, q_value in enumerate(available_q_values) if q_value == max_q_value]
        
        action = random.choice(actions_with_max_q_value)
        
        self.last_action = action
        return action

    def input(self, game):
        """Determine the best move using the current Q-table."""
        if game.beginning:
            game.beginning = False
        state = self.get_state(game)
        available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
        action = self.choose_action(state, available_actions)
        return action

    def get_last_action(self):
        """Returns the last action taken by this player."""
        return self.last_action


class Trainer:
    def __init__(self, game_class, learning_rate=0.1, discount_factor=0.9):
        self.game_class = game_class
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    def train(self, games_to_play=10000, max_exploration_rate=1.0, min_exploration_rate=0.05, decay_rate=0.00005):
        player_x = TTTQLearningPlayer(symbol="X", learning_rate=self.learning_rate, discount_factor=self.discount_factor)
        player_o = TTTQLearningPlayer(symbol="O", learning_rate=self.learning_rate, discount_factor=self.discount_factor)
        
        for game_index in range(games_to_play):
            game = self.game_class()
            exploration_rate = max(min_exploration_rate, max_exploration_rate - decay_rate * game_index)  # Decaying exploration rate
            
            while True:
                current_player = player_x if game.current_player == "X" else player_o
                state = current_player.get_state(game)
                available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
                
                if random.uniform(0, 1) < exploration_rate:
                    action = random.choice(available_actions)
                else:
                    action = current_player.input(game)
                
                game_over, winner = game.user_input(*action)
                
                next_state = current_player.get_state(game)  # End state
                reward = self.determine_reward(current_player.symbol, winner, game_over)
                current_player.update_q_table(state, action, next_state, game_over, reward)

                if game_over:
                    break
                    
            if game_index % 1000 == 0:
                print(f"Game {game_index}: Exploration rate = {exploration_rate}")
        
        return player_x, player_o

    def determine_reward(self, player_symbol, winner, done):
        if not done:
            return -0.01  # Reward for non-terminal moves
        if winner is None:
            return 0.5  # Draw
        if winner == player_symbol:
            return 1  # Win
        return -1  # Loss


class Connect4QLearningPlayer:
    def __init__(self, symbol, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.symbol = symbol
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon for epsilon-greedy strategy
        self.q_table = {}  # Initialize Q-table as an empty dictionary
        self.last_action = None  # Store the last action taken
    
    @classmethod
    def to_string(cls) -> str:
        return "qlearning"

    def get_state(self, game):
        """Returns the current state as a tuple, which is hashable and can be used as a key in the Q-table."""
        return tuple([tuple(row) for row in game.board])

    def update_q_table(self, state, action, next_state, reward, done):
        """Update the Q-table using the Q-learning algorithm."""
        if state not in self.q_table:
            self.q_table[state] = [0] * 7  # Initialize Q-values for each column
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * 7  # Initialize Q-values for each column
        
        if done:
            target = reward  # If the game has ended, the reward is the final outcome
        else:
            target = reward + self.discount_factor * max(self.q_table[next_state])
        
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + \
                                      self.learning_rate * target

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(available_actions)
        else:
            q_values = self.q_table.get(state, [0] * 7)  # Initialize Q-values for each column
            max_q_value = max(q_values)
            actions_with_max_q_value = [action for action in available_actions if q_values[action] == max_q_value]
            action = random.choice(actions_with_max_q_value)
        self.last_action = action  # Store the last action
        return action

    def input(self, game):
        """Determine the best move using the current Q-table."""
        if game.beginning:
            game.beginning = False
        state = self.get_state(game)
        available_actions = [col for col in range(7) if game.is_valid_move(col)]  # Adjust for Connect4
        action = self.choose_action(state, available_actions)
        return action

    def get_last_action(self):
        """Returns the last action taken by this player."""
        return self.last_action


class TTTRandomPlayer:
    """A simple random player for Tic Tac Toe."""
    def __init__(self, symbol):
        self.symbol = symbol
        self.last_action = None
    
    def get_state(self, game):
        """Returns the current state as a tuple, which is hashable and can be used as a key in the Q-table."""
        return tuple([tuple(row) for row in game.board])

    def input(self, game):
        if game.beginning:
            game.beginning = False
        available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
        action = random.choice(available_actions)
        self.last_action = action
        return action

    def get_last_action(self):
        """Returns the last action taken by this player."""
        return self.last_action


class Connect4RandomPlayer:
    """A simple random player for Connect4."""
    def __init__(self, symbol):
        self.symbol = symbol
        self.last_action = None
    
    def get_state(self, game):
        """Returns the current state as a tuple, which is hashable and can be used as a key in the Q-table."""
        return tuple([tuple(row) for row in game.board])

    def input(self, game):
        if game.beginning:
            game.beginning = False
        available_actions = [col for col in range(7) if game.is_valid_move(col)]  # Adjust for Connect4
        action = random.choice(available_actions)
        self.last_action = action
        return action

    def get_last_action(self):
        """Returns the last action taken by this player."""
        return self.last_action
