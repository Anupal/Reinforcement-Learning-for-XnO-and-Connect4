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
            winning_move = self.find_winning_move(game)
            if winning_move:
                return winning_move

            # If no winning move, check for a blocking move
            block_move = self.block_opponent_win(game)
            if block_move:
                return block_move

            # If no blocking move is necessary, choose a random move
            return self.random_move(game)

    @classmethod
    def to_string(cls) -> str:
        return "default"
    
    def find_winning_move(self, game):
        """Identify and execute a winning move if possible."""
        for row in range(3):
            for col in range(3):
                if game.board[row][col] == " ":
                    game.board[row][col] = self.symbol  # Temporarily make the move
                    if game.check_win(self.symbol):  # Check if this move wins the game
                        game.board[row][col] = " "  # Reset the move
                        return row, col  # Return the winning move
                    game.board[row][col] = " "  # Reset the move if it's not a winning one
        return None

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
