import numpy as np
from players.qleaarning import TTTQLearningPlayer
from game.ttt import play_tic_tac_toe, TicTacToe
from players.default import TTTDefaultPlayer
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer

def evaluate_performance(player, opponent, games_to_play=100):
    player_wins = 0
    player_draws = 0
    player_losses = 0
    for _ in range(games_to_play):
        winner = play_tic_tac_toe(player, opponent, False)
        if winner == "X":
            player_wins += 1
        elif winner == "O":
            player_draws += 1
        else:
            player_draws += 1

    
    return player_wins, player_losses, player_draws


def tune_hyperparameters(symbol, training_opponent_class, opponent_class, learning_rate_values, discount_factor_values, exploration_rate_values, games_to_play=100):
    best_performance = 0
    best_hyperparameters = {}

    for alpha in learning_rate_values:
        for gamma in discount_factor_values:
            for epsilon in exploration_rate_values:
                player = TTTQLearningPlayer(symbol=symbol, learning_rate=alpha, discount_factor=gamma)
                opponent = opponent_class(symbol="O" if symbol == "X" else "X")
                player.train(training_opponent_class(symbol="O" if symbol == "X" else "X"), TicTacToe, 10000, epsilon, epsilon, 0)
                
                player_wins, player_losses, player_draws = evaluate_performance(player, opponent, games_to_play)
                win_rate = player_wins / games_to_play
                
                print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}, Win Rate: {win_rate}")
                
                if win_rate > best_performance:
                    best_performance = win_rate
                    best_hyperparameters = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}

    print(f"Best Performance: {best_performance}")
    print(f"Best Hyperparameters: {best_hyperparameters}")


print("Training with Default player")
tune_hyperparameters(
    symbol="X",
    training_opponent_class=TTTDefaultPlayer,
    opponent_class=TTTDefaultPlayer, 
    learning_rate_values=[0.1, 0.5, 0.9], 
    discount_factor_values=[0.9, 0.95, 0.99], 
    exploration_rate_values=[0.1, 0.5, 1.0], 
    games_to_play=100
)

print("Training with MinMax ABP")
tune_hyperparameters(
    symbol="X",
    training_opponent_class=TTTMinimaxABPPlayer,
    opponent_class=TTTDefaultPlayer, 
    learning_rate_values=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9], 
    discount_factor_values=[0.8, 0.9, 0.95, 0.99], 
    exploration_rate_values=[0.1, 0.5, 1.0], 
    games_to_play=100
)

# Best Performance: 0.21
# Best Hyperparameters: {'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.1}