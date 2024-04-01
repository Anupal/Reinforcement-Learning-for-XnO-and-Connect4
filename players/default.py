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
            block_move = self.block_opponent_win(game)
            if block_move:
                return block_move
            else:
                return self.random_move(game)

    @classmethod
    def to_string(cls) -> str:
        return "default"

    def block_opponent_win(self, game):
        """Identify and execute a blocking move to prevent the opponent from winning."""
        for row in range(3):
            for col in range(3):
                if game.board[row][col] == " ":
                    game.board[row][col] = self.opponent_symbol
                    if game.check_win(self.opponent_symbol):
                        game.board[row][col] = " "
                        return row, col
                    game.board[row][col] = " "
        return None

    def random_move(self, game):
        """Make a random move if no blocking move is available."""
        available_actions = [(row, col) for row in range(3) for col in range(3) if game.board[row][col] == " "]
        return random.choice(available_actions)
