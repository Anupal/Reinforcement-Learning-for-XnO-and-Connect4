import pandas as pd
import json

from game.connect4 import play_connect4, Connect4
from players.minimax import Connect4MinimaxPlayer, Connect4MinimaxABPPlayer
from players.qleaarning import Connect4QLearningPlayer, train_q_learning_players
from players.default import Connect4DefaultPlayer
from multiprocessing import Process, Queue


NUM_GAMES = 5
QLEARNING_EPISODES = 30_000


def match(player_x_class, player_o_class, num_games, trained_ql_player_x, trained_ql_player_o, results_queue):
    game_stats = {
        player_x_class.to_string() + " wins": 0,
        player_o_class.to_string() + " wins": 0,
        player_x_class.to_string() + " draws": 0,
        player_o_class.to_string() + " draws": 0,
        player_x_class.to_string() + " win rate (%)": 0,
        player_o_class.to_string() + " win rate (%)": 0
    }
    print(player_x_class.to_string(), "vs", player_o_class.to_string())
    for i in range(num_games):
        if player_x_class == Connect4QLearningPlayer:
            player_x = trained_ql_player_x
        else:
            player_x = player_x_class("X")
        
        if player_o_class == Connect4QLearningPlayer:
            player_o = trained_ql_player_o
        else:
            player_o = player_o_class("O")

        # print(f"  Game {i + 1}")
        winner = play_connect4(player_x, player_o, False)
        if winner == "X":
            game_stats[player_x.to_string() + " wins"] += 1
        elif winner == "O":
            game_stats[player_o.to_string() + " wins"] += 1
        else:
            game_stats[player_x.to_string() + " draws"] += 1
            game_stats[player_o.to_string() + " draws"] += 1
    
    total_games = num_games * 2  # Total games played by both players
    game_stats[player_x_class.to_string() + " win rate (%)"] = (game_stats[player_x_class.to_string() + " wins"] / total_games) * 100
    game_stats[player_o_class.to_string() + " win rate (%)"] = (game_stats[player_o_class.to_string() + " wins"] / total_games) * 100
    
    results_queue.put({player_x_class.to_string() + "," + player_o_class.to_string(): game_stats})


def main():
    player_classes = [Connect4MinimaxPlayer, Connect4MinimaxABPPlayer, Connect4DefaultPlayer, Connect4QLearningPlayer]
    trained_ql_player_x, trained_ql_player_o = train_q_learning_players(QLEARNING_EPISODES, Connect4QLearningPlayer("X"), Connect4QLearningPlayer("O"), Connect4)


    print("\nMatches:")
    results_queue, processes = Queue(), []
    for player_x_class in player_classes:
        for player_o_class in player_classes:
            if player_x_class == player_o_class:
                continue
            processes.append(
                Process(target=match, args=(player_x_class, player_o_class, NUM_GAMES, trained_ql_player_x, trained_ql_player_o, results_queue))
            )
            
    for process in processes:
        process.start()
    
    for process in processes:
        process.join()

    result = {}
    while not results_queue.empty():
        result |= results_queue.get()
    
    # print(json.dumps(result, indent=2))

   # Display pairing results
    print("\nPairing Results:")
    pairing_df = pd.DataFrame(result).T
    print(pairing_df.fillna("-"))
    pairing_df = pairing_df.fillna(0).astype(int)

    # Calculate total wins, draws, and losses for each player
    total_results = {}
    for player in player_classes:
        total_games = (len(player_classes) - 1) * 2 * NUM_GAMES
        player_name = player.to_string()
        total_wins = sum(pairing_df[player_name + " wins"])
        total_draws = sum(pairing_df[player_name + " draws"])
        total_losses = total_games - total_wins - total_draws
        total_win_rate = total_wins / total_games * 100
        total_results[player_name] = {"Games": total_games, "Wins": total_wins, "Draws": total_draws, "Losses": total_losses, "Win Rate (%)": f"{total_win_rate:.2f}"}

    # Display total results for each player
    print("\nTotal Results:")
    total_results_df = pd.DataFrame(total_results).T
    print(total_results_df)


if __name__ == "__main__":
    main()