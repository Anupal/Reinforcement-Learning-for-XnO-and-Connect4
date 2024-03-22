import numpy as np
import random

class TTTQLearningPlayer:
    def __init__(self, symbol, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.symbol = symbol
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon for epsilon-greedy strategy
        self.q_table = {}  # Initialize Q-table as an empty dictionary

    def get_state(self, game):
        """Returns the current state as a tuple, which is hashable and can be used as a key in the Q-table."""
        return tuple([tuple(row) for row in game.board])

    def update_q_table(self, state, action, next_state, reward, done):
        """Update the Q-table using the Q-learning algorithm."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros((3, 3))
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros((3, 3))
        
        if done:
            target = reward  # If the game has ended, the reward is the final outcome
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + \
                                      self.learning_rate * target

    def choose_action(self, state, available_actions):
        """Choose an action based on the epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_actions)  # Explore
        else:
            q_values = self.q_table.get(state, np.zeros((3, 3)))
            max_q_value = np.max(q_values[available_actions])
            actions_with_max_q_value = [action for action in available_actions if q_values[action] == max_q_value]
            return random.choice(actions_with_max_q_value)  # Exploit

    def input(self, game):
        """Determine the best move using the current Q-table."""
        state = self.get_state(game)
        available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
        action = self.choose_action(state, available_actions)
        return action
