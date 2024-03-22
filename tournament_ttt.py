import json

from game.ttt import play_tic_tac_toe, TicTacToe
from players.human import TTTHumanPlayer
from players.minimax import TTTMinimaxPlayer, TTTMinimaxABPPlayer
from players.qleaarning import TTTQLearningPlayer, train_q_learning_players
from multiprocessing import Process, Queue


NUM_GAMES = 100
QLEARNING_EPISODES = 50_000


def match(player_x_class, player_o_class, num_games, trained_ql_player_x, trained_ql_player_o, results_queue):
    game_stats = {
        player_x_class.to_string() + " wins": 0,
        player_o_class.to_string() + " wins": 0,
        "draws": 0
    }
    print(player_x_class.to_string(), "vs", player_o_class.to_string())
    for i in range(num_games):
        if player_x_class == TTTQLearningPlayer:
            player_x = trained_ql_player_x
        else:
            player_x = player_x_class("X")
        
        if player_o_class == TTTQLearningPlayer:
            player_o = trained_ql_player_o
        else:
            player_o = player_o_class("O")

        # print(f"  Game {i + 1}")
        winner = play_tic_tac_toe(player_x, player_o, False)
        if winner == "X":
            game_stats[player_x.to_string() + " wins"] += 1
        elif winner == "O":
            game_stats[player_o.to_string() + " wins"] += 1
        else:
            game_stats["draws"] += 1
    
    results_queue.put({player_x_class.to_string() + "," + player_o_class.to_string(): game_stats})


def main():
    
    player_classes = [TTTMinimaxPlayer, TTTMinimaxABPPlayer, TTTQLearningPlayer]
    trained_ql_player_x, trained_ql_player_o = train_q_learning_players(QLEARNING_EPISODES, TTTQLearningPlayer("X"), TTTQLearningPlayer("O"), TicTacToe)


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
    
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()