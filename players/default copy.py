import copy
import random


class TTTDefaultPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"

    def input(self, game):
        """Determine the best move using the default strategy."""
        if game.beginning:
            game.beginning = False
            return random.choice([(i, j) for i in range(3) for j in range(3)])  # Choose a random move
        else:
            # Check for a winning move first
            # winning_move = self.find_winning_move(game)
            # if winning_move:
            #     print("winning move")
            #     return winning_move

            # If no winning move, check for a blocking move
            block_move = self.block_opponent_win(game)
            if block_move:
                print("block move")
                return block_move

            # If no blocking move is necessary, choose a random move
            print("random move")
            return self.random_move(game)

    @classmethod
    def to_string(cls) -> str:
        return "default"
    
    def find_winning_move(self, game):
        """Identify and execute a winning move if possible, without modifying the actual game board."""
        for row in range(3):
            for col in range(3):
                if game.board[row][col] == " ":
                    # Create a deep copy of the board to test the move
                    temp_board = copy.deepcopy(game.board)
                    temp_board[row][col] = self.symbol
                    if self.check_win(temp_board, self.symbol):
                        return row, col
        return None

    def block_opponent_win(self, game):
        """Identify and execute a blocking move to prevent the opponent from winning, without modifying the actual game board."""
        for row in range(3):
            for col in range(3):
                if game.board[row][col] == " ":
                    # Create a deep copy of the board to test the move
                    temp_board = copy.deepcopy(game.board)
                    temp_board[row][col] = self.opponent_symbol
                    if self.check_win(temp_board, self.opponent_symbol):
                        return row, col
        return None

    def random_move(self, game):
        """Make a random move if no blocking move is available."""
        available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
        return random.choice(available_actions)
    
    def check_win(self, board, player_symbol):
        """Checks if the specified player has won on the given board."""
        # Check rows, columns, and diagonals
        for i in range(3):
            if all(board[i][j] == player_symbol for j in range(3)) or \
               all(board[j][i] == player_symbol for j in range(3)):
                return True
        if all(board[i][i] == player_symbol for i in range(3)) or \
           all(board[i][2-i] == player_symbol for i in range(3)):
            return True
        return False


class Connect4DefaultPlayer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.opponent_symbol = "O" if symbol == "X" else "X"

    def input(self, game):
        """Determine the best move using the default strategy."""
        if game.beginning:
            game.beginning = False
            return random.randint(0, 6)  # Choose a random column
        else:
            block_move = self.block_opponent_win(game)
            if block_move is not None:
                return block_move
            else:
                return self.random_move(game)

    @classmethod
    def to_string(cls) -> str:
        return "default"

    def block_opponent_win(self, game):
        """Identify and execute a blocking move to prevent the opponent from winning."""
        for col in range(7):
            for row in range(5, -1, -1):
                if game.board[row][col] == " ":
                    game.board[row][col] = self.opponent_symbol
                    if game.check_win(self.opponent_symbol):
                        game.board[row][col] = " "
                        return col
                    game.board[row][col] = " "
        return None

    def random_move(self, game):
        """Make a random move if no blocking move is available."""
        available_columns = [col for col in range(7) if game.board[0][col] == " "]
        return random.choice(available_columns)
