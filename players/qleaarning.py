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
    
    def train(self, opponent, game_class, games_to_play=10000, max_exploration_rate=1.0, min_exploration_rate=0.05, decay_rate=0.0005):
        for game_index in range(games_to_play):
            # exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate)*np.exp(-decay_rate*game_index)
            # exploration_rate = max(min_exploration_rate, exploration_rate)

            exploration_rate = max_exploration_rate
            game = game_class()
            
            while True:
                state = self.get_state(game)
                available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
                if game.current_player == self.symbol:
                    # Exploration vs Exploitation for the training agent
                    if random.uniform(0, 1) < exploration_rate:
                        if game.beginning:
                            game.beginning = False
                        action = random.choice(available_actions)
                    else:
                        action = self.input(game)
                    agent_moved = True
                else:
                    # Opponent's move
                    action = opponent.input(game)
                    agent_moved = False

                game_over, winner = game.user_input(*action)
                
                if agent_moved:
                    next_state = self.get_state(game)
                    reward = self.determine_reward(winner, game_over)
                    self.update_q_table(state, action, next_state, game_over, reward)
        
                if game_over:
                    break
             
            if game_index % 1000 == 0:
                print(f"Game {game_index}: Exploration rate = {exploration_rate}")
    
    def determine_reward(self, winner, done):
        """Determines the reward for the agent based on game outcome."""
        if not done:
            return 0  # Penalize non-final moves slightly to encourage winning in fewer steps
        if winner is None:
            return 0.5  # Draw
        if winner == self.symbol:
            return 1  # Win
        return -1  # Loss


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
    print("Training Q-learning players X and O with Random player.")
    win_count = {"X": 0, "O": 0, "Draw": 0}

    for episode in range(num_episodes):
        game = game_class()

        if game_class.to_string() == "ttt":
            random_player_x = TTTRandomPlayer("X")
            random_player_o = TTTRandomPlayer("O")
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
            if game_class.to_string() == "ttt":
                row, col = current_player.input(game)
                game_over, winner = game.user_input(row, col)
            else:
                col = current_player.input(game)
                game_over, winner = game.user_input(col)

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