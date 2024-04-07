import ast
import os
import numpy as np

from game.connect4 import play_connect4, Connect4
from players.human import TTTHumanPlayer
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.qleaarning import Connect4QLearningPlayer, train_q_learning_players
from players.default import Connect4DefaultPlayer

QLEARNING_EPISODES = 200_000


ql_player_x, ql_player_o = Connect4QLearningPlayer("X", 0.1, 0.99, 0.01), Connect4QLearningPlayer("O", 0.1, 0.99, 0.01),

ql_player_x.load_q_table("c4_ql_player_x.pkl")
ql_player_o.load_q_table("c4_ql_player_o.pkl")

if not ql_player_x.q_table or not ql_player_o.q_table:
    trained_ql_player_x, trained_ql_player_o = train_q_learning_players(
        QLEARNING_EPISODES,
        ql_player_x,
        ql_player_o,
        Connect4
    )
    trained_ql_player_x.save_q_table("c4_ql_player_x.pkl")
    trained_ql_player_o.save_q_table("c4_ql_player_o.pkl")
else:
    trained_ql_player_x = ql_player_x
    trained_ql_player_o = ql_player_o

winner = play_connect4(ql_player_x, Connect4DefaultPlayer("O"))

print("Winner is", winner)