from game.ttt import play_tic_tac_toe
from players.human import TTTHumanPlayer
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.qleaarning import TTTQLearningPlayer


human_player_x = TTTHumanPlayer("X")
minimax_player_x = TTTMinimaxPlayer("X")
minimax_abp_player_x = TTTMinimaxABPPlayer("X")
qlearning_player_x = TTTQLearningPlayer("X")

human_player_o = TTTHumanPlayer("O")
minimax_player_o = TTTMinimaxPlayer("O")
minimax_abp_player_o = TTTMinimaxABPPlayer("O")
qlearning_player_o = TTTQLearningPlayer("O")

# play_tic_tac_toe(human_player_x, minimax_player_o)
# winner = play_tic_tac_toe(human_player_x, minimax_abp_player_o)

num_games = 50

# mp vs mp-abp
print("MM vs MM-ABP")
mm_wins, mm_abp_wins, draws = 0, 0, 0
for i in range(num_games):
    print(f"  Game {i + 1}")
    minimax_player_x = TTTMinimaxPlayer("X")
    minimax_abp_player_o = TTTMinimaxABPPlayer("O")
    winner = play_tic_tac_toe(minimax_player_x, minimax_abp_player_o, False)
    if winner == "X":
        mm_wins += 1
    elif winner == "O":
        mm_abp_wins += 1
    else:
        draws += 1

print(f"MM wins={mm_wins} MM-ABP wins={mm_abp_wins} Draws={draws}")

# mp-abp vs mp
print("\nMM-ABP vs MM")
mm_wins, mm_abp_wins, draws = 0, 0, 0
for i in range(num_games):
    print(f"  Game {i + 1}")
    minimax_abp_player_x = TTTMinimaxABPPlayer("X")
    minimax_player_o = TTTMinimaxPlayer("O")
    
    winner = play_tic_tac_toe(minimax_abp_player_x, minimax_player_o, False)
    if winner == "X":
        mm_abp_wins += 1
    elif winner == "O":
        mm_wins += 1
    else:
        draws += 1

print(f"\nMM wins={mm_wins} MM-ABP wins={mm_abp_wins} Draws={draws}")