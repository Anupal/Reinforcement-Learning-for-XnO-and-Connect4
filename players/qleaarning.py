import numpy as np
import random
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.default import TTTDefaultPlayer
from tqdm import tqdm


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


def train_q_learning_players(num_episodes, ql_player_x, ql_player_o, game_class):
    print("Training Q-learning players X and O with Default player.")
    win_count = {"X": 0, "O": 0, "Draw": 0}

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        game = game_class()

        if game_class.to_string() == "ttt":
            # random_player_x = TTTRandomPlayer("X")
            # random_player_o = TTTRandomPlayer("O")
            random_player_x = TTTDefaultPlayer("X")
            random_player_o = TTTDefaultPlayer("O")
        else:
            random_player_x = Connect4RandomPlayer("X")
            random_player_o = Connect4RandomPlayer("O")
        
        # Alternate starting player each episode
        if episode % 2 == 0:
            player_x, player_o = ql_player_x, random_player_o
        else:
            player_x, player_o = random_player_x, ql_player_o
            
        current_player = player_x  # X starts the game

        while True:
            state_before = game.get_state()

            if game_class.to_string() == "ttt":
                row, col = current_player.input(game)
                game_over, winner = game.user_input(row, col)
            else:
                col = current_player.input(game)
                game_over, winner = game.user_input(col)

            state_after = game.get_state()

            # Determine rewards and update Q-tables accordingly
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
            else:
                # If the game is not over, no immediate reward
                reward_x = reward_o = 0

            # Update Q-table for the current player
            action_current_player = current_player.get_last_action()
            if current_player == ql_player_x:
                ql_player_x.update_q_table(state_before, action_current_player, state_after, reward_x, game_over)
            else:
                ql_player_o.update_q_table(state_before, action_current_player, state_after, reward_o, game_over)

            if game_over:
                # Update Q-table for the other player with terminal state
                next_state_x = next_state_o = None  # No next state since the game is over
                action_x = player_x.get_last_action()
                action_o = player_o.get_last_action()
                state_x = game.get_state()
                state_o = game.get_state()
                if current_player != ql_player_x:
                    ql_player_x.update_q_table(state_x, action_x, next_state_x, reward_x, True)
                if current_player != ql_player_o:
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
    best_score = -1  # Initialize with a score that will be updated
    
    for learning_rate in param_grid['learning_rate']:
        for discount_factor in param_grid['discount_factor']:
            for exploration_rate in param_grid['exploration_rate']:
                print(f"Combination: learning_rate={learning_rate} discount_factor={discount_factor} exploration_rate={exploration_rate}")
                # Initialize players with current parameters
                ql_player_x = TTTQLearningPlayer("X", learning_rate, discount_factor, exploration_rate)
                ql_player_o = TTTQLearningPlayer("O", learning_rate, discount_factor, exploration_rate)
                
                # Train the Q-learning players with the current set of parameters
                ql_player_x, ql_player_o = train_q_learning_players(num_episodes, ql_player_x, ql_player_o, game_class)
                
                # Evaluate the players
                win_counts = evaluate_players(ql_player_x, ql_player_o, game_class, num_games=num_episodes)
                
                # Calculate score by considering wins and draws (minimizing losses)
                score = win_counts['X'] + win_counts['O'] + win_counts['Draw']
                
                # Update best parameters if current score is better
                if score > best_score:
                    best_score = score
                    best_params = {
                        'learning_rate': learning_rate, 
                        'discount_factor': discount_factor, 
                        'exploration_rate': exploration_rate
                    }
    
    # Calculate adjusted win rate considering the total number of games
    best_avg_score = best_score / (2 * num_episodes)  # Adjusted for two players over num_episodes
    
    return best_params, best_avg_score


# Combination: learning_rate=0.1 discount_factor=0.99 exploration_rate=0.01
# Training Q-learning players X and O with Random player.
# Training Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 5272.72it/s]
# Training complete. Win counts: {'X': 2868, 'O': 3744, 'Draw': 3388}

# Combination: learning_rate=0.9 discount_factor=0.9 exploration_rate=0.01
# Training Q-learning players X and O with Random player.
# Training Progress: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:01<00:00, 5020.04it/s]
# Training complete. Win counts: {'X': 2764, 'O': 3823, 'Draw': 3413}