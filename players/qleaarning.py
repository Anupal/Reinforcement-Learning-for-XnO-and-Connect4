import numpy as np
import random


class TTTQLearningPlayer:
    def __init__(self, symbol, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.symbol = symbol
        self.learning_rate = learning_rate  # Alpha
        self.discount_factor = discount_factor  # Gamma
        self.exploration_rate = exploration_rate  # Epsilon for epsilon-greedy strategy
        self.q_table = {}  # Initialize Q-table as an empty dictionary
        self.last_action = None  # Store the last action taken
    
    @classmethod
    def to_string(self) -> str:
        return "qlearning"

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
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(available_actions)
        else:
            q_values = self.q_table.get(state, np.zeros((3, 3)))
            max_q_value = np.max(q_values[available_actions])
            actions_with_max_q_value = [action for action in available_actions if q_values[action] == max_q_value]
            action = random.choice(actions_with_max_q_value)
        self.last_action = action  # Store the last action
        return action

    def input(self, game):
        """Determine the best move using the current Q-table."""
        state = self.get_state(game)
        available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
        action = self.choose_action(state, available_actions)
        return action

    def get_last_action(self):
        """Returns the last action taken by this player."""
        return self.last_action


class RandomPlayer:
    """A simple random player for Tic Tac Toe."""
    def __init__(self, symbol):
        self.symbol = symbol

    def input(self, game):
        available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
        return random.choice(available_actions)


def train_q_learning_players(num_episodes, ql_player_x, ql_player_o, game_class):
    print("Traning Qlearning player x and o")
    win_count = {"X": 0, "O": 0, "Draw": 0}

    for episode in range(num_episodes):
        game = game_class()
        
        # Alternate starting player each episode
        if episode % 2 == 0:
            player_x, player_o = ql_player_x, ql_player_o
        else:
            player_x, player_o = ql_player_o, ql_player_x
            
        player_x.symbol, player_o.symbol = "X", "O"  # Ensure symbols are set correctly for this episode
        
        current_player = player_x  # X starts the game

        while True:
            state = current_player.get_state(game)
            row, col = current_player.input(game)
            game_over, winner = game.user_input(row, col)

            if game_over:
                if winner == "X":
                    win_count["X"] += 1
                    reward_x = 1
                    reward_o = -1
                elif winner == "O":
                    win_count["O"] += 1
                    reward_x = -1
                    reward_o = 1
                else:  # Draw
                    win_count["Draw"] += 1
                    reward_x = reward_o = 0.5

                # Update Q-table for both players at the end of the game
                next_state_x = next_state_o = None  # No next state since the game is over
                action_x = player_x.get_last_action()
                action_o = player_o.get_last_action()
                state_x = player_x.get_state(game)
                state_o = player_o.get_state(game)
                ql_player_x.update_q_table(state_x, action_x, next_state_x, reward_x, True)
                ql_player_o.update_q_table(state_o, action_o, next_state_o, reward_o, True)
                break

            # Switch players
            current_player = player_x if current_player == player_o else player_o

    print(f"Training complete. Win counts: {win_count}")
    return ql_player_x, ql_player_o