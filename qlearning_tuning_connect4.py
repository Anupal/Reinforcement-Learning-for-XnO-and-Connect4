import numpy as np
from players.qleaarning import tune_parameters
from game.connect4 import Connect4


# Define the parameter grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5, 0.9],
    'discount_factor': [0.9, 0.95, 0.99],
    'exploration_rate': [0.01, 0.1, 0.3]
}

# Tune the parameters
best_params, best_avg_win_rate = tune_parameters(Connect4, 1_000, param_grid)

print("Best Parameters:", best_params)
print("Best Average Win Rate:", best_avg_win_rate)