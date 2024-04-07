import ast
import os
import numpy as np

from game.ttt import play_tic_tac_toe, TicTacToe
from players.human import TTTHumanPlayer
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.qleaarning import TTTQLearningPlayer
from players.default import TTTDefaultPlayer


# player_classes = [TTTMinimaxPlayer, TTTMinimaxABPPlayer, TTTQLearningPlayer]

# QLEARNING_EPISODES = 1_00_000
# trained_ql_player_x, trained_ql_player_o = train_q_learning_players(
#     QLEARNING_EPISODES,
#     TTTQLearningPlayer("X", 0.7, 0.95, 0.1),
#     TTTQLearningPlayer("O", 0.7, 0.95, 0.1),
#     TicTacToe
#     )

# ql_player_o = TTTQLearningPlayer("O")
# ql_player_o.train(TTTHeuristicOpponent("X"), TicTacToe, 50_000, 1, 0.05, 0.0005)

ql_player_x = TTTQLearningPlayer("X", 0.7, 0.95)

ql_table_data = np.load('ql_x_table.npz')
ql_table = {ast.literal_eval(key): ql_table_data[key].copy() for key in ql_table_data}
ql_player_x.q_table = ql_table
ql_table_data.close()

# winner = play_tic_tac_toe(TTTMinimaxPlayer("X"), ql_player_o)

# winner = play_tic_tac_toe(TTTMinimaxPlayer("X"), TTTDefaultPlayer("O"))


# print(list(ql_player_x.q_table.keys()))


winner = play_tic_tac_toe(ql_player_x, TTTDefaultPlayer("O"))

# print(winner)