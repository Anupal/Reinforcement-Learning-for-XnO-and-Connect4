import itertools
import numpy as np
from game.ttt import play_tic_tac_toe, TicTacToe
from players.qleaarning import TTTQLearningPlayer
from players.default import TTTDefaultPlayer



def grid_search(game_class, opponent, parameter_grid, trials_per_combination=100):
    best_parameters = None
    best_performance = -np.inf  # Initialize with a very low win rate
    
    for learning_rate, discount_factor, exploration_rate in itertools.product(*parameter_grid.values()):
        total_wins = 0
        for trial in range(trials_per_combination):
            player = TTTQLearningPlayer(symbol="X", learning_rate=learning_rate, discount_factor=discount_factor)
            player.train(opponent, game_class, games_to_play=500, max_exploration_rate=exploration_rate)
            
            # Evaluate the trained agent
            wins, losses, draws = evaluate_agent(player, opponent, game_class, num_games=100)
            total_wins += wins
        
        win_rate = total_wins / (trials_per_combination * 100)
        
        if win_rate > best_performance:
            best_performance = win_rate
            best_parameters = (learning_rate, discount_factor, exploration_rate)
        
        print(f"LR: {learning_rate}, DF: {discount_factor}, ER: {exploration_rate}, Win Rate: {win_rate}")

    print(f"Best Parameters: {best_parameters}, Best Win Rate: {best_performance}")
    return best_parameters

def evaluate_agent(player, opponent, game_class, num_games=100):
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        game = game_class()
        while True:
            if game.current_player == player.symbol:
                action = player.input(game)
            else:
                action = opponent.input(game)
            game_over, winner = game.user_input(*action)
            if game_over:
                if winner == player.symbol:
                    wins += 1
                elif winner == opponent.symbol:
                    losses += 1
                else:
                    draws += 1
                break
    return wins, losses, draws

# Example usage:
parameter_grid = {
    'learning_rate': [0.1, 0.3, 0.5],
    'discount_factor': [0.9, 0.95, 0.99],
    'exploration_rate': [1.0]
}

opponent = TTTDefaultPlayer(symbol="O")
game_class = TicTacToe
best_parameters = grid_search(game_class, opponent, parameter_grid)
