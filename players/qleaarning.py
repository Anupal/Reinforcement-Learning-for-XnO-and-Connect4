import numpy as np
import random


class TTTQLearningPlayer:
    def __init__(self, symbol, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
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
        if game.beginning:
            game.beginning = False
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


def train_q_learning_players(num_episodes, ql_player_x, ql_player_o, game_class):
    print("Training Q-learning players X and O with Random player.")
    win_count = {"X": 0, "O": 0, "Draw": 0}

    for episode in range(num_episodes):
        game = game_class()
        
        # Alternate starting player each episode
        if episode % 2 == 0:
            player_x, player_o = ql_player_x, RandomPlayer("O")
        else:
            player_x, player_o = RandomPlayer("X"), ql_player_o
            
        current_player = player_x  # X starts the game

        while True:
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


def evaluate_players(player_x, player_o, game_class, num_games=100):
    win_count = {"X": 0, "O": 0, "Draw": 0}
    
    for _ in range(num_games):
        game = game_class()
        current_player = player_x
        
        while True:
            row, col = current_player.input(game)
            game_over, winner = game.user_input(row, col)

            if game_over:
                if winner == "X":
                    win_count["X"] += 1
                elif winner == "O":
                    win_count["O"] += 1
                else:
                    win_count["Draw"] += 1
                break

            current_player = player_x if current_player == player_o else player_o
    
    return win_count


def tune_parameters(game_class, num_episodes, param_grid):
    best_params = {}
    best_avg_win_rate = -1
    
    for learning_rate in param_grid['learning_rate']:
        for discount_factor in param_grid['discount_factor']:
            for exploration_rate in param_grid['exploration_rate']:
                ql_player_x = TTTQLearningPlayer("X", learning_rate, discount_factor, exploration_rate)
                ql_player_o = TTTQLearningPlayer("O", learning_rate, discount_factor, exploration_rate)
                
                ql_player_x, ql_player_o = train_q_learning_players(num_episodes, ql_player_x, ql_player_o, game_class)
                
                avg_win_rate = (evaluate_players(ql_player_x, ql_player_o, game_class)['X'] + evaluate_players(ql_player_x, ql_player_o, game_class)['O']) / (2 * num_episodes)
                
                if avg_win_rate > best_avg_win_rate:
                    best_avg_win_rate = avg_win_rate
                    best_params = {'learning_rate': learning_rate, 'discount_factor': discount_factor, 'exploration_rate': exploration_rate}
    
    return best_params, best_avg_win_rate