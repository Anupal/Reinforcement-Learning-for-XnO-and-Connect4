from tabulate import tabulate
import time

from game.ttt import TicTacToe
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.qleaarning import TTTQLearningPlayer
from players.default import TTTDefaultPlayer


def time_game(player_x, player_o):
    turn_times = [[],[]]
    recursions = [[], []]
    game = TicTacToe()
    while True:
        try:
            if game.current_player == "X":
                # get execution time
                start_time = time.time()
                row, col = player_x.input(game)
                end_time = time.time()
                turn_times[0].append(end_time-start_time)

                # get recusions in case of minimax
                if "minimax" in player_x.to_string():
                    recursions[0].append(player_x.metric_recursions)
            else:
                # get execution time
                start_time = time.time()
                row, col = player_o.input(game)
                end_time = time.time()
                turn_times[1].append(end_time-start_time)
                # get recusions in case of minimax
                if "minimax" in player_o.to_string():
                    recursions[1].append(player_o.metric_recursions)

            game_over, winner = game.user_input(row, col)
            if game_over:
                return turn_times, recursions
        except ValueError:
            print("Invalid input. Please enter numbers between 0 and 2.")

ql_player_x = TTTQLearningPlayer("X", 0.1, 0.99, 0.01)

model_load_start_time = time.time()
ql_player_x.load_q_table("ttt_ql_player_x.pkl")
model_load_end_time = time.time()
print(f"Takes {model_load_end_time-model_load_start_time} seconds to load model.")

players = [TTTMinimaxPlayer("X"), TTTMinimaxABPPlayer("X"), ql_player_x]


rows = []
for player_x in players:
    turn_times, recursions = time_game(player_x, TTTDefaultPlayer("O"))
    avg_turn_time = sum(turn_times[0])/len(turn_times[0])
    if "minimax" in player_x.to_string():
        avg_recursions = sum(recursions[0])/len(recursions[0])
    else:
        avg_recursions = 0
    rows.append(
        {
            "Algorithm": player_x.to_string(),
            "Average turn time (s)": avg_turn_time,
            "Average recursions": avg_recursions
        }
    )
print(tabulate(rows, headers="keys", tablefmt='fancy_grid'))