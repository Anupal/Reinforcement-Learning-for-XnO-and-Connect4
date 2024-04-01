import numpy as np
from players.qleaarning import tune_parameters
from game.ttt import TicTacToe


# Define the parameter grid
param_grid = {
    'learning_rate': [0.1, 0.3, 0.5],
    'discount_factor': [0.9, 0.95, 0.99],
    'exploration_rate': [0.1, 0.2, 0.3]
}

# Tune the parameters
best_params, best_avg_win_rate = tune_parameters(TicTacToe, 40_000, param_grid)

print("Best Parameters:", best_params)
print("Best Average Win Rate:", best_avg_win_rate)