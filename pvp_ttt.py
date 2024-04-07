import ast
import os
import numpy as np

from game.ttt import play_tic_tac_toe, TicTacToe
from players.human import TTTHumanPlayer
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.qleaarning import TTTQLearningPlayer, train_q_learning_players
from players.default import TTTDefaultPlayer

QLEARNING_EPISODES = 200_000


ql_player_x, ql_player_o = TTTQLearningPlayer("X", 0.1, 0.99, 0.01), TTTQLearningPlayer("O", 0.1, 0.99, 0.01),

ql_player_x.load_q_table("ttt_ql_player_x.pkl")
ql_player_o.load_q_table("ttt_ql_player_o.pkl")

if not ql_player_x.q_table or not ql_player_o.q_table:
    trained_ql_player_x, trained_ql_player_o = train_q_learning_players(
        QLEARNING_EPISODES,
        ql_player_x,
        ql_player_o,
        TicTacToe
    )
    trained_ql_player_x.save_q_table("ttt_ql_player_x.pkl")
    trained_ql_player_o.save_q_table("ttt_ql_player_o.pkl")
else:
    trained_ql_player_x = ql_player_x
    trained_ql_player_o = ql_player_o

winner = play_tic_tac_toe(ql_player_x, TTTDefaultPlayer("O"))

print("Winner is", winner)