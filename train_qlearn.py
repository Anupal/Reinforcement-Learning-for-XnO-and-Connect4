import ast
import os
import numpy as np

from game.ttt import play_tic_tac_toe, TicTacToe
from players.qleaarning import TTTQLearningPlayer
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.default import TTTDefaultPlayer

# ql_trainer = Trainer(TicTacToe, 0.5, 0.95)
ql_player_x = TTTQLearningPlayer("X", 0.5, 0.999)
ql_player_o = TTTQLearningPlayer("O", 0.5, 0.999)
ql_player_x.train(TTTDefaultPlayer("O"), TicTacToe, 100_000, 1, 0.01, 0.005)
ql_player_o.train(TTTDefaultPlayer("X"), TicTacToe, 100_000, 1, 0.01, 0.005)
ql_player_x.train(TTTMinimaxABPPlayer("O"), TicTacToe, 100_000, 1, 0.01, 0.005)
ql_player_o.train(TTTMinimaxABPPlayer("X"), TicTacToe, 100_000, 1, 0.01, 0.005)
ql_player_x.train(TTTMinimaxPlayer("O"), TicTacToe, 100_000, 1, 0.01, 0.005)
ql_player_o.train(TTTMinimaxPlayer("X"), TicTacToe, 100_000, 1, 0.01, 0.005)

# ql_player_x, ql_player_o = ql_trainer.train(20_000, 1, 0.2, 0.05)
    
np.savez("ql_x_table.npz", **{str(key): value for key, value in ql_player_x.q_table.items()})
# np.savez("ql_o_table.npz", **{str(key): value for key, value in ql_player_o.q_table.items()})


winner = play_tic_tac_toe(ql_player_x, TTTDefaultPlayer("O"))
print(winner)
print("-------------------------------------------------------------")
winner = play_tic_tac_toe(TTTDefaultPlayer("X"), ql_player_o)
print(winner)